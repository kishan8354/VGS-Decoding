#!/usr/bin/env python
"""Build the paper's Table 1 / Figure 2 / Table 3 from a directory of run files.

  python scripts/make_figures.py --run_dir runs/vqarad/llava-med --model llava-med \
      --closed_mode first --out_dir figs

Expects files named <method>.jsonl, e.g. greedy.jsonl, vcd_a1.0.jsonl, dola.jsonl,
vgs_a0.5.jsonl ... Writes table1.tex, table3.tex, fig2_<model>.png and results.json.
Missing methods are skipped, so this works on a partial reproduction.
"""
import argparse
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from vgsd.metrics import mcnemar_exact, paired_bootstrap, per_sample_scores, summarize  # noqa: E402

PAPER_VQARAD = {   # Table 1, VQA-RAD: open, closed, overall
    "llava-med": {"greedy": (34.45, 68.92, 53.64), "vcd": (30.85, 61.20, 47.71),
                  "dola": (32.76, 58.96, 47.34), "opera": (33.22, 61.69, 49.05),
                  "vgs": (38.90, 72.91, 57.75)},
    "chexagent": {"greedy": (22.02, 70.92, 49.24), "vcd": (21.73, 68.53, 47.78),
                  "dola": (20.73, 68.92, 47.55), "opera": (20.50, 69.32, 47.67),
                  "vgs": (23.48, 71.31, 50.10)},
    "medgemma": {"greedy": (49.50, 61.75, 56.32), "vcd": (50.29, 57.77, 54.45),
                 "dola": (51.91, 72.51, 63.38), "opera": (48.90, 65.74, 58.27),
                 "vgs": (53.05, 73.58, 64.48)},
}
ORDER = ["greedy", "orig_pair", "vcd", "dola", "vgs"]


def label_of(path):
    return os.path.splitext(os.path.basename(path))[0]


def family(label):
    return re.split(r"_a[\d.]+$", label)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--closed_mode", default="first", choices=["word", "substring", "first"])
    ap.add_argument("--out_dir", default="figs")
    ap.add_argument("--baseline", default="greedy")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    runs = {}
    for p in sorted(glob.glob(os.path.join(a.run_dir, "*.jsonl"))):
        runs[label_of(p)] = {r["qid"]: r for r in map(json.loads, filter(str.strip, open(p)))}
    if a.baseline not in runs:
        sys.exit(f"no {a.baseline}.jsonl in {a.run_dir}")
    common = sorted(set.intersection(*[set(v) for v in runs.values()]))
    print(f"{len(common)} questions common to {len(runs)} runs")

    base_recs = [runs[a.baseline][q] for q in common]
    base_s = per_sample_scores(base_recs, a.closed_mode)
    results = {}
    for label, run in runs.items():
        recs = [run[q] for q in common]
        s = summarize(recs, a.closed_mode)
        sc = per_sample_scores(recs, a.closed_mode)
        closed = np.array([r["is_closed"] for r in recs])
        s["delta"] = s["overall"] - summarize(base_recs, a.closed_mode)["overall"]
        s["p_bootstrap"] = None if label == a.baseline else paired_bootstrap(base_s, sc)
        s["p_mcnemar"] = None if label == a.baseline else mcnemar_exact(base_s[closed] > 0.5, sc[closed] > 0.5)
        steps = sum(r.get("steps", 0) for r in recs)
        s["flip_rate"] = 100 * sum(r.get("flips", 0) for r in recs) / steps if steps else None
        s["sec_per_q"] = float(np.mean([r.get("sec", np.nan) for r in recs]))
        results[label] = s
    json.dump({"model": a.model, "closed_mode": a.closed_mode, "n": len(common),
               "results": results}, open(os.path.join(a.out_dir, "results.json"), "w"), indent=2)

    # ---- Table 1 (ours next to the paper's numbers)
    paper = PAPER_VQARAD.get(a.model, {})
    main_runs = [l for l in runs if family(l) in ORDER and not l.startswith("vgs_a")] + \
                (["vgs_a1.0"] if "vgs_a1.0" in runs else [])
    main_runs = sorted(set(main_runs), key=lambda l: ORDER.index(family(l)) if family(l) in ORDER else 99)
    with open(os.path.join(a.out_dir, "table1.tex"), "w") as f:
        f.write("\\begin{tabular}{llrrrr|rrr}\n\\toprule\n")
        f.write("Model & Method & Open & Closed & Overall & $\\Delta$ & Open$^p$ & Closed$^p$ & Overall$^p$ \\\\\n\\midrule\n")
        for label in main_runs:
            s = results[label]
            pp = paper.get(family(label))
            ours = f"{s['open']:.2f} & {s['closed']:.2f} & {s['overall']:.2f} & {s['delta']:+.2f}"
            theirs = f"{pp[0]:.2f} & {pp[1]:.2f} & {pp[2]:.2f}" if pp else "-- & -- & --"
            f.write(f"{a.model} & {label} & {ours} & {theirs} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")

    # ---- Table 3 (alpha sweep)
    alphas = sorted([(float(re.search(r"_a([\d.]+)", l).group(1)), l) for l in runs if l.startswith("vgs_a")])
    if alphas:
        with open(os.path.join(a.out_dir, "table3.tex"), "w") as f:
            f.write("\\begin{tabular}{lrrrr}\n\\toprule\n$\\alpha$ & Open & Closed & Overall & $p$ \\\\\n\\midrule\n")
            b = results[a.baseline]
            f.write(f"0.0 & {b['open']:.2f} & {b['closed']:.2f} & {b['overall']:.2f} & -- \\\\\n")
            for av, label in alphas:
                s = results[label]
                f.write(f"{av} & {s['open']:.2f} & {s['closed']:.2f} & {s['overall']:.2f} & {s['p_bootstrap']:.3f} \\\\\n")
            f.write("\\bottomrule\n\\end{tabular}\n")

    # ---- Figure 2
    labels = main_runs
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(1.6 * len(labels) + 2, 4))
    for k, (metric, marker) in enumerate([("open", "o"), ("closed", "s"), ("overall", "^")]):
        ax.plot(x, [results[l][metric] for l in labels], marker=marker, label=metric.capitalize())
        if paper:
            py = [paper.get(family(l), (np.nan,) * 3)[k] for l in labels]
            ax.plot(x, py, marker=marker, linestyle="--", alpha=0.45,
                    label=f"{metric.capitalize()} (paper)")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Score (%)"); ax.set_title(f"{a.model} on VQA-RAD (n={len(common)}, {a.closed_mode} matching)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out_dir, f"fig2_{a.model}.png"), dpi=200)

    if alphas:
        fig2, ax2 = plt.subplots(figsize=(5, 3.5))
        av = [0.0] + [v for v, _ in alphas]
        ov = [results[a.baseline]["overall"]] + [results[l]["overall"] for _, l in alphas]
        ax2.plot(av, ov, marker="o")
        ax2.set_xlabel(r"reweighting strength $\alpha$"); ax2.set_ylabel("Overall (%)")
        ax2.set_title(f"{a.model}: effect of $\\alpha$"); ax2.grid(alpha=0.3)
        fig2.tight_layout(); fig2.savefig(os.path.join(a.out_dir, f"alpha_{a.model}.png"), dpi=200)

    print(f"wrote {a.out_dir}/table1.tex, table3.tex, fig2_{a.model}.png, results.json")


if __name__ == "__main__":
    main()
