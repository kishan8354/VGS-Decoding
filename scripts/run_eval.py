#!/usr/bin/env python
"""Run greedy / VGS / VCD / orig_pair decoding on a medical VQA dataset.

Examples
  python scripts/run_eval.py --model medgemma --dataset vqa-rad --method greedy --out runs/rad_medgemma_greedy.jsonl
  python scripts/run_eval.py --model medgemma --dataset vqa-rad --method vgs --alpha 1.0 --out runs/rad_medgemma_vgs_a1.jsonl
  python scripts/run_eval.py --model medgemma --dataset vqa-rad --method orig_pair --out runs/rad_medgemma_origpair.jsonl   # sanity
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch  # noqa: E402
from tqdm import tqdm  # noqa: E402

from vgsd.data import iter_batches, load_mimic_diff_vqa, load_vqa_rad  # noqa: E402
from vgsd.dola import DoLaProcessor  # noqa: E402
from vgsd.models import DEFAULT_PROMPT, load_adapter  # noqa: E402
from vgsd.noise import prepare_pair  # noqa: E402
from vgsd.vgs import PairedDecodingProcessor, interleave  # noqa: E402


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=["llava-med", "llava-med-official", "medgemma", "chexagent"])
    p.add_argument("--model_id", default=None)
    p.add_argument("--dataset", required=True, choices=["vqa-rad", "mimic-diff-vqa"])
    p.add_argument("--method", required=True,
                   choices=["greedy", "vgs", "vcd", "orig_pair", "dola"])
    p.add_argument("--dola_layers", default="high", choices=["high", "low"])
    p.add_argument("--dola_rep_penalty", type=float, default=1.2)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--delta", type=float, default=0.01)
    p.add_argument("--vcd_beta", type=float, default=0.1)
    p.add_argument("--sigma", type=float, default=0.07)
    p.add_argument("--lam", type=float, default=70.0)
    p.add_argument("--noise_mode", default="gp", choices=["gp", "vase_full"])
    p.add_argument("--resize", type=int, default=512, help="square resize before noise (VASE); 0 = off")
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=1, help="questions per step (each becomes 2 rows)")
    p.add_argument("--pair_mode", default="batch", choices=["batch", "stepwise"])
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--load_in_4bit", action="store_true")
    p.add_argument("--trace_topk", type=int, default=0, help="store per-step top-k VGS tables")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    # MIMIC-Diff-VQA paths
    p.add_argument("--mimic_qa_csv"); p.add_argument("--mimic_metadata_csv"); p.add_argument("--mimic_jpg_root")
    p.add_argument("--mimic_sample_n", type=int, default=None)
    p.add_argument("--out", required=True)
    return p.parse_args()


def main():
    a = parse()
    torch.manual_seed(a.seed)
    if a.dataset == "vqa-rad":
        samples = load_vqa_rad("test", limit=a.limit)
    else:
        samples = load_mimic_diff_vqa(a.mimic_qa_csv, a.mimic_metadata_csv, a.mimic_jpg_root,
                                      limit=a.limit, sample_n=a.mimic_sample_n, seed=a.seed)

    done = set()
    if os.path.exists(a.out):
        with open(a.out) as f:
            done = {json.loads(l)["qid"] for l in f if l.strip()}
        print(f"[resume] {len(done)} already done")
    samples = [s for s in samples if s.qid not in done]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)

    adapter = load_adapter(a.model, a.model_id, dtype=a.dtype, device=a.device, load_in_4bit=a.load_in_4bit)
    if a.batch_size > 1 and not getattr(adapter, "supports_batch_padding", False):
        print("[warn] this adapter supports batch_size=1 only; forcing 1"); a.batch_size = 1

    paired = a.method in ("vgs", "vcd", "orig_pair")
    proc = None
    if paired:
        proc = PairedDecodingProcessor(method={"orig_pair": "orig"}.get(a.method, a.method),
                                       alpha=a.alpha, delta=a.delta, beta=a.vcd_beta,
                                       eos_token_ids=adapter.eos_token_ids, trace_topk=a.trace_topk)

    cfg = {k: v for k, v in vars(a).items() if k not in ("out",)}
    with open(a.out, "a") as fout:
        n_batches = (len(samples) + a.batch_size - 1) // a.batch_size
        for batch in tqdm(iter_batches(samples, a.batch_size), total=n_batches, desc=f"{a.model}/{a.method}"):
            origs, dists = zip(*[prepare_pair(s.load_image(), s.qid, size=a.resize or None, sigma=a.sigma,
                                              lam=a.lam, mode=a.noise_mode, base_seed=a.seed) for s in batch])
            prompts = [a.prompt.format(question=s.question) for s in batch]
            t0 = time.time()
            if a.method == "dola":
                inputs = adapter.build_inputs(list(origs), prompts)
                with DoLaProcessor(adapter.model, which=a.dola_layers,
                                   repetition_penalty=a.dola_rep_penalty) as dproc:
                    gen = adapter.generate(inputs, a.max_new_tokens, dproc)
            elif paired:
                proc.reset()
                inputs = adapter.build_inputs(interleave(origs, dists), interleave(prompts, prompts))
                gen = adapter.generate(inputs, a.max_new_tokens, proc, pair_mode=a.pair_mode)[0::2]
            else:
                inputs = adapter.build_inputs(list(origs), prompts)
                gen = adapter.generate(inputs, a.max_new_tokens)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            dt = (time.time() - t0) / len(batch)
            preds = adapter.decode(gen)
            for i, (s, pred) in enumerate(zip(batch, preds)):
                rec = {"qid": s.qid, "question": s.question, "answer": s.answer, "is_closed": s.is_closed,
                       "pred": pred, "sec": round(dt, 4), "meta": s.meta}
                if paired:
                    rec["steps"], rec["flips"] = proc.steps[i], proc.flips[i]
                    if a.trace_topk:
                        tok = adapter.tokenizer
                        rec["trace"] = [[dict(c, text=tok.decode([c["tok"]])) for c in step]
                                        for step in proc.trace[i]]
                fout.write(json.dumps(rec) + "\n")
            fout.flush()
    with open(a.out + ".config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"[done] wrote {a.out}")


if __name__ == "__main__":
    main()
