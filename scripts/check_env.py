#!/usr/bin/env python
"""Check the GPU and print the flags to use. Run this before anything else:

    python scripts/check_env.py
"""
import sys

import torch

# weights in GB at 16-bit, plus a rough working-set allowance for the vision tower,
# prefill activations and the KV cache of one orig/dist pair at ~600 tokens
MODELS = [("medgemma", "MedGemma-4B", 8.6, 2.5),
          ("chexagent", "CheXagent-2-3B", 6.4, 2.5),
          ("llava-med", "LLaVA-Med-7B", 14.2, 2.0)]


def main():
    if not torch.cuda.is_available():
        print("No CUDA device visible. Check the driver and that torch was installed with CUDA.")
        print(f"torch {torch.__version__}")
        sys.exit(1)

    i = torch.cuda.current_device()
    p = torch.cuda.get_device_properties(i)
    cap = f"{p.major}.{p.minor}"
    total = p.total_memory / 1024**3
    bf16 = torch.cuda.is_bf16_supported()
    dtype = "bf16" if bf16 else "fp16"
    print(f"GPU            : {p.name}  ({total:.1f} GB, compute {cap})")
    print(f"torch          : {torch.__version__}")
    print(f"bfloat16       : {'yes' if bf16 else 'no  (Turing/Volta/Pascal) -> use fp16'}")
    print(f"--dtype        : {dtype}\n")

    for key, label, w, overhead in MODELS:
        need = w + overhead
        if need < total - 1.0:
            verdict, flags = "fits", f"--dtype {dtype}"
        elif w + 1.0 < total:
            verdict, flags = "tight", f"--dtype {dtype} --max_new_tokens 48   (fall back to --load_in_4bit)"
        else:
            verdict, flags = "needs 4-bit", "--load_in_4bit"
        print(f"{label:16s} ~{need:4.1f} GB  {verdict:11s}  {flags}")

    print("""
Notes
  * bf16 and fp16 give the same answer on almost every question, but a handful of near-ties
    can flip. If greedy and orig_pair disagree, re-check a 50-question slice with --dtype fp32.
  * 4-bit changes the probability distributions that VGS reads, so if you quantize one run,
    quantize every run in the comparison (greedy, VCD and VGS) the same way, and say so
    in the paper. Never compare a 4-bit VGS run against a 16-bit greedy baseline.
  * fp32 does not fit any of these models on a 16 GB card; use it only with --limit for spot checks.""")


if __name__ == "__main__":
    main()
