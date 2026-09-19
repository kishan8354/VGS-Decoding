#!/usr/bin/env bash
# 50-question sanity check on a 16 GB card. ~10 minutes.
#   bash scripts/smoke_16gb.sh            # medgemma
#   bash scripts/smoke_16gb.sh llava-med
set -euo pipefail
MODEL=${1:-medgemma}
OUT=runs/smoke/$MODEL
mkdir -p "$OUT"

# bf16 on Ampere and newer, fp16 on T4/V100/P100
DTYPE=$(python -c "import torch;print('bf16' if torch.cuda.is_bf16_supported() else 'fp16')")
EXTRA=""
[ "$MODEL" = "llava-med" ] && EXTRA="--max_new_tokens 48"    # 7B is tight in 16 GB
echo "model=$MODEL dtype=$DTYPE $EXTRA"

for M in greedy orig_pair; do
  python scripts/run_eval.py --model "$MODEL" --dataset vqa-rad --method $M \
    --dtype "$DTYPE" --batch_size 1 --limit 50 $EXTRA --out "$OUT/$M.jsonl"
done

# must be ~100%: pairing alone should not change greedy output
python scripts/score.py --agreement "$OUT/greedy.jsonl" "$OUT/orig_pair.jsonl"

python scripts/run_eval.py --model "$MODEL" --dataset vqa-rad --method vgs --alpha 1.0 \
  --dtype "$DTYPE" --batch_size 1 --limit 50 $EXTRA --out "$OUT/vgs_a1.0.jsonl"
python scripts/score.py --baseline "$OUT/greedy.jsonl" "$OUT/vgs_a1.0.jsonl"
