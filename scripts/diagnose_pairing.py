#!/usr/bin/env python
"""Explain a greedy vs orig_pair disagreement.

  python scripts/diagnose_pairing.py runs/s/llava_greedy.jsonl runs/s/llava_origpair.jsonl

Prints every differing question with both answers, where they first diverge, and whether the
clinical verdict (yes/no) changed. A benign numerical difference looks like: same verdict,
divergence late in the sentence, wording only. A real bug looks like: empty output, prompt echo,
truncation at token 1, or flipped verdicts.
"""
import argparse
import json
import re
import sys


def load(p):
    with open(p) as f:
        return {r["qid"]: r for r in map(json.loads, filter(str.strip, f))}


def verdict(text):
    for w in re.sub(r"[^\w\s]", " ", text.lower()).split():
        if w in ("yes", "no"):
            return w
    return None


def common_prefix_words(a, b):
    aw, bw = a.split(), b.split()
    n = 0
    while n < min(len(aw), len(bw)) and aw[n] == bw[n]:
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("greedy")
    ap.add_argument("paired")
    ap.add_argument("--show", type=int, default=20)
    a = ap.parse_args()
    g, p = load(a.greedy), load(a.paired)
    qids = sorted(set(g) & set(p))
    diff = [q for q in qids if g[q]["pred"].strip() != p[q]["pred"].strip()]

    print(f"{len(qids)} common, {len(diff)} differ ({100*(1-len(diff)/max(len(qids),1)):.2f}% agreement)\n")
    flips, empties, early = 0, 0, 0
    for q in diff[:a.show]:
        gp, pp = g[q]["pred"].strip(), p[q]["pred"].strip()
        vg, vp = verdict(gp), verdict(pp)
        n = common_prefix_words(gp, pp)
        flips += int(vg != vp)
        empties += int(not gp or not pp)
        early += int(n == 0)
        print(f"Q  : {g[q]['question']}")
        print(f"GT : {g[q]['answer']}")
        print(f"G  : {gp}")
        print(f"P  : {pp}")
        print(f"   -> diverges after {n} identical words | verdict {vg} -> {vp}"
              f"{'  *** VERDICT CHANGED' if vg != vp else ''}\n")

    if diff:
        print(f"summary over shown cases: verdict changes {flips}, empty outputs {empties}, "
              f"diverging at the first word {early}")
        print("\nverdict changes and empty outputs point at a bug; late wording-only differences\n"
              "are bf16 near-ties and are expected. If in doubt, re-run both with --dtype fp32\n"
              "on a --limit 20 slice: the disagreement should disappear.")
    else:
        print("identical outputs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
