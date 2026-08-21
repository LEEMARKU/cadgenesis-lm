#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Tune the muTransfer BASE LR on the nano ladder, then run the full model
# with the SAME base LR.  Each step is a single quick run on 1 GPU.
#
#   bash scripts/gpu/run_mu_ladder.sh
#
# Pick the base LR where the medium run tracks the nano run's loss curve,
# then export LR=that_value when calling pretrain.sh.
# ---------------------------------------------------------------------------
set -euo pipefail

DATASET="${DATASET:-data/cad_programs.jsonl}"
LR="${LR:-5e-4}"
EPOCHS="${EPOCHS:-20}"

for D in 64 128 256; do
  echo ">> base LR ladder, d_model=$D (muTransfer, 1 GPU)"
  torchrun --standalone --nproc_per_node=1 train.py \
    --data "$DATASET" \
    --out-dir "checkpoints/mu-ladder/d${D}" \
    --model mini \
    --d-model "$D" \
    --mu-transfer --bf16 \
    --lr "$LR" \
    --epochs "$EPOCHS" \
    --packed
done

echo ">> compare final train loss across checkpoints/mu-ladder/d{64,128,256}"