#!/usr/bin/env bash
# Full VQA-RAD reproduction for one model: Table 1 row, Table 3 alpha sweep, Figure 2.
#
#   bash scripts/run_all_vqarad.sh llava-med first "{question}\nAnswer the question using a single word or phrase."
#
# args: 1 model  2 closed-matching mode (word|substring|first)  3 prompt  4.. extra flags (e.g. --load_in_4bit)
set -euo pipefail
MODEL=${1:-llava-med}
MODE=${2:-first}
PROMPT=${3:-$'{question}\nAnswer the question using a single word or phrase.'}
shift 3 || true
EXTRA=("$@")

OUT=runs/vqarad/$MODEL
FIGS=figs/$MODEL
mkdir -p "$OUT" "$FIGS"
DTYPE=$(python -c "import torch;print('bf16' if torch.cuda.is_bf16_supported() else 'fp16')")
COMMON=(--model "$MODEL" --dataset vqa-rad --dtype "$DTYPE" --max_new_tokens 48 --prompt "$PROMPT" "${EXTRA[@]}")
echo "model=$MODEL dtype=$DTYPE mode=$MODE extra=${EXTRA[*]:-none}"

run () { local name=$1; shift; python scripts/run_eval.py "${COMMON[@]}" "$@" --out "$OUT/$name.jsonl"; }

# 1. baseline and sanity check (orig_pair must reproduce greedy)
run greedy    --method greedy
run orig_pair --method orig_pair
python scripts/score.py --agreement "$OUT/greedy.jsonl" "$OUT/orig_pair.jsonl"
python scripts/diagnose_pairing.py "$OUT/greedy.jsonl" "$OUT/orig_pair.jsonl" --show 5

# 2. baselines
run vcd  --method vcd --alpha 1.0
run dola --method dola --dola_layers high

# 3. VGS alpha sweep (Table 3); alpha=1.0 is the Table 1 entry
for A in 0.5 1.0 1.5 2.0; do run "vgs_a$A" --method vgs --alpha "$A"; done

# 4. a small traced run for qualitative figures
python scripts/run_eval.py "${COMMON[@]}" --method vgs --alpha 1.0 --trace_topk 5 --limit 50 \
  --out "$OUT/vgs_trace50.jsonl"

# 5. tables, figures, significance
python scripts/score.py --closed_mode "$MODE" --baseline "$OUT/greedy.jsonl" \
  "$OUT/vcd.jsonl" "$OUT/dola.jsonl" "$OUT"/vgs_a*.jsonl
python scripts/make_figures.py --run_dir "$OUT" --model "$MODEL" --closed_mode "$MODE" --out_dir "$FIGS"
echo "done -> $OUT and $FIGS"
