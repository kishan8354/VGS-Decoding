#!/usr/bin/env python
"""Find the prompt and answer-matching rule that reproduce the paper's greedy baseline.

  python scripts/calibrate_baseline.py --model llava-med --dtype bf16 --limit 150

Runs greedy decoding once per candidate prompt, scores each under all three closed-ended
matching rules, and prints the distance to the paper's published greedy numbers. Pick the
combination with the smallest distance and use it for EVERY run afterwards.

Paper greedy, full VQA-RAD test (open / closed / overall):
  LLaVA-Med 34.45 / 68.92 / 53.64   CheXagent 22.02 / 70.92 / 49.24   MedGemma 49.50 / 61.75 / 56.32
"""
import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from vgsd.metrics import summarize  # noqa: E402

PAPER = {"llava-med": (34.45, 68.92, 53.64),
         "chexagent": (22.02, 70.92, 49.24),
         "medgemma": (49.50, 61.75, 56.32)}

PROMPTS = {
    # VASE's prompt (current default) -- produces full sentences
    "vase": "Answer this question as concisely as possible based on the provide images: {question}",
    # the standard LLaVA / LLaVA-Med VQA instruction, used for VQA-RAD in their own eval
    "short": "{question}\nAnswer the question using a single word or phrase.",
    # explicit yes/no steer for closed questions
    "brief": "{question} Answer briefly.",
    # bare question, no instruction
    "bare": "{question}",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--limit", type=int, default=150, help="questions per prompt (0 = all 451)")
    ap.add_argument("--max_new_tokens", type=int, default=48)
    ap.add_argument("--out_dir", default="runs/calib")
    ap.add_argument("--prompts", nargs="*", default=list(PROMPTS))
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                    help="extra flags passed straight to run_eval.py, e.g. --load_in_4bit")
    a = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(a.out_dir, exist_ok=True)
    rows = []
    for key in a.prompts:
        out = os.path.join(a.out_dir, f"{a.model}_{key}.jsonl")
        cmd = [sys.executable, os.path.join(here, "run_eval.py"), "--model", a.model,
               "--dataset", "vqa-rad", "--method", "greedy", "--dtype", a.dtype,
               "--max_new_tokens", str(a.max_new_tokens), "--prompt", PROMPTS[key], "--out", out]
        if a.limit:
            cmd += ["--limit", str(a.limit)]
        cmd += [x for x in a.extra if x != "--"]
        print("+", " ".join(cmd[:8]), "...", flush=True)
        subprocess.run(cmd, check=True)

        recs = [json.loads(l) for l in open(out) if l.strip()]
        for mode in ("word", "substring", "first"):
            s = summarize(recs, mode)
            tgt = PAPER.get(a.model)
            dist = (abs(s["open"] - tgt[0]) + abs(s["closed"] - tgt[1])) / 2 if tgt else float("nan")
            rows.append((key, mode, s["open"], s["closed"], s["overall"], dist))

    rows.sort(key=lambda r: r[5])
    tgt = PAPER.get(a.model)
    print(f"\nn={a.limit or 451} per prompt. paper greedy for {a.model}: "
          f"open={tgt[0]} closed={tgt[1]} overall={tgt[2]}\n" if tgt else "")
    print(f"{'prompt':8s} {'match':10s} {'open':>7s} {'closed':>7s} {'overall':>8s} {'mean |Δ| vs paper':>18s}")
    for key, mode, o, c, ov, d in rows:
        print(f"{key:8s} {mode:10s} {o:7.2f} {c:7.2f} {ov:8.2f} {d:18.2f}")
    best = rows[0]
    print(f"\nclosest: --prompt \"{PROMPTS[best[0]]}\"  with --closed_mode {best[1]}")
    print("Sanity-check a few predictions by eye before adopting it; a matching rule that scores\n"
          "well on garbled answers is worse than one that scores lower on clean ones.")


if __name__ == "__main__":
    main()
