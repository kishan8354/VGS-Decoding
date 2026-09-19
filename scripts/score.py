#!/usr/bin/env python
"""Score one or more run files.

  python scripts/score.py runs/rad_medgemma_greedy.jsonl runs/rad_medgemma_vgs_a1.jsonl
  python scripts/score.py --baseline runs/rad_medgemma_greedy.jsonl runs/rad_medgemma_vgs_a1.jsonl
  python scripts/score.py --agreement runs/rad_medgemma_greedy.jsonl runs/rad_medgemma_origpair.jsonl
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np  # noqa: E402

from vgsd.metrics import mcnemar_exact, paired_bootstrap, per_sample_scores, summarize  # noqa: E402


def load(path):
    with open(path) as f:
        return {r["qid"]: r for r in map(json.loads, filter(str.strip, f))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+")
    p.add_argument("--baseline", default=None)
    p.add_argument("--agreement", action="store_true", help="exact-match rate between the two given runs")
    p.add_argument("--closed_mode", default="word", choices=["word", "substring", "first"])
    a = p.parse_args()

    runs = {r: load(r) for r in ([a.baseline] if a.baseline else []) + a.runs}
    common = sorted(set.intersection(*[set(v) for v in runs.values()]))
    print(f"{len(common)} questions common to all runs (closed_mode={a.closed_mode})\n")

    if a.agreement:
        x, y = [runs[r] for r in a.runs[:2]]
        same = np.mean([x[q]["pred"].strip() == y[q]["pred"].strip() for q in common])
        print(f"exact-match agreement: {100*same:.2f}%  (orig_pair vs greedy should be ~100%)")
        return

    base = None
    if a.baseline:
        base_recs = [runs[a.baseline][q] for q in common]
        base = summarize(base_recs, a.closed_mode)
        base_s = per_sample_scores(base_recs, a.closed_mode)

    print(f"{'run':55s} {'open':>7s} {'closed':>7s} {'overall':>8s} {'Δ':>7s} {'flip%':>6s} {'sec/q':>6s} {'p_boot':>7s} {'p_mcn':>7s}")
    for path in ([a.baseline] if a.baseline else []) + a.runs:
        recs = [runs[path][q] for q in common]
        s = summarize(recs, a.closed_mode)
        steps = sum(r.get("steps", 0) for r in recs)
        flips = sum(r.get("flips", 0) for r in recs)
        flip = f"{100*flips/steps:.1f}" if steps else "-"
        sec = np.mean([r.get("sec", float("nan")) for r in recs])
        d = pb = pm = "-"
        if base and path != a.baseline:
            d = f"{s['overall'] - base['overall']:+.2f}"
            sc = per_sample_scores(recs, a.closed_mode)
            pb = f"{paired_bootstrap(base_s, sc):.3f}"
            closed = np.array([r["is_closed"] for r in recs])
            pm = f"{mcnemar_exact(base_s[closed] > 0.5, sc[closed] > 0.5):.3f}"
        print(f"{os.path.basename(path)[:55]:55s} {s['open']:7.2f} {s['closed']:7.2f} {s['overall']:8.2f} "
              f"{d:>7s} {flip:>6s} {sec:6.2f} {pb:>7s} {pm:>7s}")
    print(f"\nopen={s['n_open']} closed={s['n_closed']}")


if __name__ == "__main__":
    main()
