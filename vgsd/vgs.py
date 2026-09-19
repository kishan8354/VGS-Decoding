"""Core of VGS-Decoding (Kolli et al., arXiv 2603.20314) plus a VCD baseline.

Key implementation idea ("paired rows")
---------------------------------------
Every question is put in the batch twice: row 2i holds the ORIGINAL image,
row 2i+1 the DISTORTED image, with identical text. At each decoding step a
LogitsProcessor sees both rows' logits, computes P_final from the pair, and
writes the same log P_final into BOTH rows. Greedy argmax therefore picks the
same token for both rows, so the distorted branch is always conditioned on the
same prefix y_<t -- exactly Algorithm 1 of the paper -- while HuggingFace
`generate` keeps handling KV caches, attention masks and image features.
"""
from __future__ import annotations

import math
from typing import Iterable, Optional

import torch
from transformers import LogitsProcessor

METHODS = ("vgs", "vcd", "orig")


def vgs_from_logprobs(logp_orig: torch.Tensor, logp_dist: torch.Tensor) -> torch.Tensor:
    """Eq. (4): VGS = (P_o - P_d) / (P_o + P_d).

    Computed through the exact identity (a-b)/(a+b) = tanh((ln a - ln b)/2),
    which stays finite for vocabulary entries whose probabilities underflow.
    Tokens with P_o = P_d = 0 get VGS = 0.
    """
    diff = logp_orig - logp_dist                      # nan only for (-inf) - (-inf)
    diff = torch.nan_to_num(diff, nan=0.0, posinf=float("inf"), neginf=float("-inf"))
    return torch.tanh(diff / 2.0)


def vgs_reweight(logits_orig, logits_dist, alpha: float = 1.0, delta: float = 0.01):
    """Eq. (5): P_final ∝ P_orig * max(1 + alpha * VGS, delta). Returns log P_final (normalised)."""
    lo = torch.log_softmax(logits_orig.float(), dim=-1)
    ld = torch.log_softmax(logits_dist.float(), dim=-1)
    vgs = vgs_from_logprobs(lo, ld)
    w = torch.clamp(1.0 + alpha * vgs, min=delta)
    lf = lo + torch.log(w)
    lf = lf - torch.logsumexp(lf, dim=-1, keepdim=True)
    return lf, vgs, lo, ld


def vcd_contrast(logits_orig, logits_dist, alpha: float = 1.0, beta: float = 0.1):
    """VCD (Leng et al., CVPR 2024), as in the official vcd_sample.py:
    (1+a)*l_o - a*l_d, with the adaptive plausibility cut-off on raw original logits."""
    lo, ld = logits_orig.float(), logits_dist.float()
    cd = (1.0 + alpha) * lo - alpha * ld
    cutoff = math.log(beta) + lo.max(dim=-1, keepdim=True).values
    cd = cd.masked_fill(lo < cutoff, float("-inf"))
    return torch.log_softmax(cd, dim=-1)


class PairedDecodingProcessor(LogitsProcessor):
    """LogitsProcessor for batches laid out as [orig_0, dist_0, orig_1, dist_1, ...].

    method="vgs"  : VGS-Decoding (paper)
    method="vcd"  : VCD baseline, same noisy image, greedy
    method="orig" : use only the original row (== greedy). Sanity check for the pairing machinery.

    Diagnostics (reset() before each generate call):
      steps[i]  number of decoding steps for pair i (until EOS)
      flips[i]  steps where the chosen token differs from plain greedy's argmax
      trace[i]  optional per-step top-k table (token id, P_orig, P_dist, VGS, P_final)
    """

    def __init__(self, method: str = "vgs", alpha: float = 1.0, delta: float = 0.01,
                 beta: float = 0.1, eos_token_ids: Optional[Iterable[int]] = None,
                 trace_topk: int = 0):
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}")
        self.method, self.alpha, self.delta, self.beta = method, alpha, delta, beta
        self.eos = torch.tensor(sorted(set(eos_token_ids or [])), dtype=torch.long)
        self.trace_topk = trace_topk
        self.reset()

    def reset(self):
        self._n = None
        self._prompt_len = None
        self.steps, self.flips, self.trace, self._done = None, None, None, None

    def _init_state(self, n_pairs: int):
        self._n = n_pairs
        self.steps = [0] * n_pairs
        self.flips = [0] * n_pairs
        self.trace = [[] for _ in range(n_pairs)]
        self._done = [False] * n_pairs

    @torch.no_grad()
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        n_rows, vocab = scores.shape
        if n_rows % 2:
            raise RuntimeError("Paired decoding needs an even number of rows [orig, dist, ...]")
        n_pairs = n_rows // 2
        if self._n != n_pairs:
            self._init_state(n_pairs)
            self._prompt_len = input_ids.shape[1]

        # stop counting diagnostics for pairs that already produced EOS
        if self.eos.numel() and input_ids.shape[1] > self._prompt_len:
            last = input_ids[0::2, -1].detach().cpu()
            for i, finished in enumerate(torch.isin(last, self.eos).tolist()):
                self._done[i] = self._done[i] or finished

        pair = scores.view(n_pairs, 2, vocab)
        raw_o, raw_d = pair[:, 0], pair[:, 1]

        vgs = ld = None
        if self.method == "vgs":
            lf, vgs, lo, ld = vgs_reweight(raw_o, raw_d, self.alpha, self.delta)
        elif self.method == "vcd":
            lf = vcd_contrast(raw_o, raw_d, self.alpha, self.beta)
            lo = torch.log_softmax(raw_o.float(), dim=-1)
        else:  # "orig"
            lo = torch.log_softmax(raw_o.float(), dim=-1)
            lf = lo

        greedy_tok = lo.argmax(-1)
        final_tok = lf.argmax(-1)
        for i in range(n_pairs):
            if self._done[i]:
                continue
            self.steps[i] += 1
            self.flips[i] += int(final_tok[i] != greedy_tok[i])
            if self.trace_topk:
                if ld is None:
                    ld = torch.log_softmax(raw_d.float(), dim=-1)
                if vgs is None:
                    vgs = vgs_from_logprobs(lo, ld)
                cand = torch.unique(torch.cat([lf[i].topk(self.trace_topk).indices,
                                               lo[i].topk(self.trace_topk).indices]))
                self.trace[i].append([
                    {"tok": int(t), "p_orig": float(lo[i, t].exp()), "p_dist": float(ld[i, t].exp()),
                     "vgs": float(vgs[i, t]), "p_final": float(lf[i, t].exp())}
                    for t in cand.tolist()])

        # identical scores in both rows -> both rows pick the same token
        return lf.unsqueeze(1).expand(n_pairs, 2, vocab).reshape(n_rows, vocab)


def interleave(xs_orig, xs_dist):
    """[a0, a1], [b0, b1] -> [a0, b0, a1, b1]"""
    out = []
    for a, b in zip(xs_orig, xs_dist):
        out.extend([a, b])
    return out
