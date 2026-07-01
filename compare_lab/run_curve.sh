#!/bin/bash
# Learning-curve + MM-ceiling orchestration (runs INSIDE the tr1work container).
# Serial: train {1000,3047} (267 trained separately) -> 6 inference passes.
# Clean logs (TQDM off) under /work/compare_lab/eval150/logs/.
set -e
cd /work
export TQDM_DISABLE=1 TRANSFORMERS_VERBOSITY=error
L=/work/compare_lab/eval150/logs;  mkdir -p "$L" /work/compare_lab/eval150/preds
SFT="python /work/compare_lab/sft/train.py --completion-only --max-length 4096 --epochs 2 --batch 4 --grad-accum 4"
INF="python /work/compare_lab/infer_ic.py --batch 32 --max-new 200"
EMM=/work/compare_lab/eval150/eval_mm.jsonl
EPX=/work/compare_lab/eval150/eval_priceonly.jsonl
P=/work/compare_lab/eval150/preds
say(){ echo "[$(date +%H:%M:%S)] $*"; }

# ---- train (skip if adapter already present) ----
for N in 267 1000 3047; do
  D=/work/compare_lab/sft/data_top150_mm; [ "$N" = 3047 ] || D=${D}_${N}
  A=/work/data/sft_adapter_t150_${N}
  if [ -f "$A/adapter_model.safetensors" ]; then say "train $N: skip (exists)"; else
    say "train $N -> $A"; $SFT --data "$D" --out "$A" > "$L/train_${N}.log" 2>&1; say "train $N done"
  fi
done

# ---- inference matrix ----
# #1 learning curve on full-MM: base(0) + 267 + 1000 + 3047
say "infer mm_base";  $INF --eval $EMM --out $P/mm_base.jsonl                                   > "$L/inf_mm_base.log" 2>&1
say "infer mm_267";   $INF --eval $EMM --out $P/mm_267.jsonl  --adapter /work/data/sft_adapter_t150_267  > "$L/inf_mm_267.log" 2>&1
say "infer mm_1000";  $INF --eval $EMM --out $P/mm_1000.jsonl --adapter /work/data/sft_adapter_t150_1000 > "$L/inf_mm_1000.log" 2>&1
say "infer mm_3047";  $INF --eval $EMM --out $P/mm_3047.jsonl --adapter /work/data/sft_adapter_t150_3047 > "$L/inf_mm_3047.log" 2>&1
# #2 MM ceiling: base + best-SFT, price-only vs full-MM (mm_base/mm_3047 reused above)
say "infer px_base";  $INF --eval $EPX --out $P/px_base.jsonl                                   > "$L/inf_px_base.log" 2>&1
say "infer px_3047";  $INF --eval $EPX --out $P/px_3047.jsonl --adapter /work/data/sft_adapter_t150_3047 > "$L/inf_px_3047.log" 2>&1
say "ALL DONE"
