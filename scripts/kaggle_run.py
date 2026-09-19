"""Driver for Kaggle / Colab free GPUs. Paste into one notebook cell (or %run it).

Why this exists: free sessions die after ~12 h (Kaggle) or on idle (Colab), and the weekly
quota is ~30 GPU-h. run_eval.py appends to JSONL and skips qids already present, so a killed
session loses at most one question. Point OUT_ROOT at persistent storage (Kaggle Dataset output
/kaggle/working, or Google Drive on Colab) and just re-run the cell after a disconnect.

T4 and P100 have no bfloat16 -> DTYPE = "fp16".
"""
import os
import subprocess
import sys
import time

REPO = "/kaggle/working/vgs-decoding-repro"      # Colab: /content/vgs-decoding-repro
OUT_ROOT = "/kaggle/working/runs"                # Colab: /content/drive/MyDrive/vgs_runs
MODEL = "medgemma"
DTYPE = "fp16"                                   # T4/P100 have no bf16
TIME_BUDGET_H = 11.0                             # stop before Kaggle's 12 h cutoff
HF_TOKEN = os.environ.get("HF_TOKEN", "")        # MedGemma is gated: accept its licence first

JOBS = [                                          # order matters: baseline and sanity first
    ("greedy",    ["--method", "greedy"]),
    ("orig_pair", ["--method", "orig_pair", "--limit", "50"]),
    ("vcd_a1.0",  ["--method", "vcd", "--alpha", "1.0"]),
    ("vgs_a1.0",  ["--method", "vgs", "--alpha", "1.0"]),
    ("vgs_a1.5",  ["--method", "vgs", "--alpha", "1.5"]),
    ("vgs_a0.5",  ["--method", "vgs", "--alpha", "0.5"]),
    ("vgs_a2.0",  ["--method", "vgs", "--alpha", "2.0"]),
]


def sh(cmd):
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def setup():
    if not os.path.isdir(REPO):
        raise SystemExit(f"Upload/clone the repo to {REPO} first")
    sh([sys.executable, "-m", "pip", "install", "-q", "-U",
        "transformers==4.56.2", "accelerate", "datasets"])
    sh([sys.executable, "-m", "pytest", "-q", os.path.join(REPO, "tests")])
    if HF_TOKEN:
        from huggingface_hub import login
        login(HF_TOKEN)


def main():
    setup()
    out = os.path.join(OUT_ROOT, MODEL)
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    for name, args in JOBS:
        if (time.time() - t0) / 3600 > TIME_BUDGET_H:
            print("[budget] stopping; re-run this cell in the next session to continue")
            break
        sh([sys.executable, os.path.join(REPO, "scripts", "run_eval.py"),
            "--model", MODEL, "--dataset", "vqa-rad", "--dtype", DTYPE,
            "--out", os.path.join(out, name + ".jsonl")] + args)
    sh([sys.executable, os.path.join(REPO, "scripts", "score.py"), "--agreement",
        os.path.join(out, "greedy.jsonl"), os.path.join(out, "orig_pair.jsonl")])
    done = [os.path.join(out, n + ".jsonl") for n, _ in JOBS[2:]
            if os.path.exists(os.path.join(out, n + ".jsonl"))]
    if done:
        sh([sys.executable, os.path.join(REPO, "scripts", "score.py"),
            "--baseline", os.path.join(out, "greedy.jsonl")] + done)


if __name__ == "__main__":
    main()
