#!/usr/bin/env bash
# Table 1 (VQA-RAD rows for greedy / VCD / VGS) + Table 3 alpha sweep.
# Usage: bash scripts/reproduce_vqarad.sh medgemma   (or llava-med)
set -euo pipefail
MODEL=${1:-medgemma}
OUT=runs/vqarad/$MODEL
mkdir -p "$OUT"
run() { python scripts/run_eval.py --model "$MODEL" --dataset vqa-rad "$@"; }

run --method greedy    --out $OUT/greedy.jsonl
run --method orig_pair --out $OUT/orig_pair.jsonl            # sanity: must match greedy
python scripts/score.py --agreement $OUT/greedy.jsonl $OUT/orig_pair.jsonl

run --method vcd --alpha 1.0 --out $OUT/vcd_a1.0.jsonl
for A in 0.5 1.0 1.5 2.0; do
  run --method vgs --alpha $A --out $OUT/vgs_a$A.jsonl
done
run --method vgs --alpha 1.0 --trace_topk 5 --limit 50 --out $OUT/vgs_a1.0_trace50.jsonl   # for qualitative figures

for MODE in word substring; do
  python scripts/score.py --closed_mode $MODE --baseline $OUT/greedy.jsonl \
    $OUT/vcd_a1.0.jsonl $OUT/vgs_a0.5.jsonl $OUT/vgs_a1.0.jsonl $OUT/vgs_a1.5.jsonl $OUT/vgs_a2.0.jsonl
done
