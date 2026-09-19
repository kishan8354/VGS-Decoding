"""DoLA (Chuang et al., ICLR 2024) as a self-contained LogitsProcessor.

transformers >= 4.56 moved its built-in DoLA to a remote `custom_generate` repo that needs
trust_remote_code and network access, so this re-implements it locally:

  * forward hooks on the decoder layers capture each candidate layer's last-token hidden state
  * premature logits = output_embedding(h_layer)   (as in the old HF _dola_decoding: the final
    norm is NOT re-applied; matching that choice keeps the numbers comparable to published DoLA)
  * the premature layer is chosen per step by maximum Jensen-Shannon divergence from the mature
    (final-layer) distribution
  * scores = log_softmax(mature) - log_softmax(premature), restricted to the adaptive
    plausibility set {mature >= log(beta) + max(mature)}
  * a repetition penalty of 1.2 is applied, as the DoLA paper and HF default do

Use `--method dola`. This is a single-image baseline: no distorted branch, no pairing.
"""
from __future__ import annotations

import math
from typing import List

import torch
from transformers import LogitsProcessor, RepetitionPenaltyLogitsProcessor


def find_decoder_layers(model) -> torch.nn.ModuleList:
    """Locate the list of transformer blocks for the language model of a VLM."""
    candidates = [
        "model.language_model.layers",          # transformers>=4.52 Llava / Gemma3
        "language_model.model.layers",          # older VLM wrappers
        "model.model.layers",
        "model.layers",
        "transformer.h",
    ]
    for path in candidates:
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
        except AttributeError:
            continue
        if isinstance(obj, torch.nn.ModuleList) and len(obj) >= 1:
            return obj
    # fallback: the longest ModuleList that is not part of the vision tower
    best = None
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.ModuleList) and "vision" not in name and len(mod) >= 1:
            if best is None or len(mod) > len(best[1]):
                best = (name, mod)
    if best:
        return best[1]
    raise RuntimeError("Could not locate decoder layers for DoLA on this model class")


def default_candidate_layers(n_layers: int, which: str = "high") -> List[int]:
    """HF's dola_layers presets: even-indexed layers in the upper ('high') or lower ('low') half."""
    lo, hi = (n_layers // 2, n_layers) if which == "high" else (0, n_layers // 2)
    layers = list(range(lo, hi, 2))
    return layers or [max(0, n_layers // 2)]


def _jsd(logp: torch.Tensor, logq: torch.Tensor) -> torch.Tensor:
    """Jensen-Shannon divergence between two log-prob rows, per batch element."""
    p, q = logp.exp(), logq.exp()
    m = ((p + q) / 2).clamp_min(1e-12).log()
    kl_p = (p * (logp - m)).sum(-1)
    kl_q = (q * (logq - m)).sum(-1)
    return (kl_p + kl_q) / 2


class DoLaProcessor(LogitsProcessor):
    """Attach with `with DoLaProcessor(model) as proc:` so the hooks are always removed."""

    def __init__(self, model, which: str = "high", beta: float = 0.1,
                 repetition_penalty: float = 1.2, candidate_layers: List[int] | None = None):
        self.model = model
        self.layers = find_decoder_layers(model)
        self.candidates = candidate_layers or default_candidate_layers(len(self.layers), which)
        self.beta = beta
        self.rep = RepetitionPenaltyLogitsProcessor(repetition_penalty) if repetition_penalty and repetition_penalty != 1.0 else None
        self._cache: dict[int, torch.Tensor] = {}
        self._handles = []
        self.chosen_layers: List[int] = []

    def __enter__(self):
        def make_hook(idx):
            def hook(_module, _inp, out):
                h = out[0] if isinstance(out, tuple) else out
                self._cache[idx] = h[:, -1, :].detach()
            return hook
        for i in self.candidates:
            self._handles.append(self.layers[i].register_forward_hook(make_hook(i)))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        return False

    @torch.no_grad()
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        head = self.model.get_output_embeddings()
        mature = torch.log_softmax(scores.float(), dim=-1)

        best_jsd, best_logp, best_layer = None, None, None
        for i in self.candidates:
            h = self._cache.get(i)
            if h is None:
                continue
            prem = torch.log_softmax(head(h.to(head.weight.dtype)).float(), dim=-1)
            d = _jsd(mature, prem)
            if best_jsd is None:
                best_jsd, best_logp = d, prem
                best_layer = torch.full_like(d, i, dtype=torch.long)
            else:
                take = d > best_jsd
                best_jsd = torch.where(take, d, best_jsd)
                best_logp = torch.where(take[:, None], prem, best_logp)
                best_layer = torch.where(take, torch.full_like(best_layer, i), best_layer)
        if best_logp is None:
            return scores
        self.chosen_layers.append(int(best_layer[0]))

        out = mature - best_logp
        cutoff = math.log(self.beta) + mature.max(dim=-1, keepdim=True).values
        out = out.masked_fill(mature < cutoff, float("-inf"))
        if self.rep is not None:
            out = self.rep(input_ids, out)
        return out
