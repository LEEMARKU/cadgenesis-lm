#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# FSDP multi-GPU pretraining for CADGenesis-LM.
#
#   GPU box (Linux, CUDA torch, >=1 GPU):
#     bash scripts/gpu/pretrain.sh --gpus 8 --out checkpoints/run-fsdp
#
#   All knobs are env-overridable:
#     GPUS=8  DATASET=data/cad_programs.jsonl  MODEL=small
#     LR=1e-3  EPOCHS=100  OUT=checkpoints/run-fsdp
#
# LR note: with --mu-transfer the LR here is the *base* LR (width-agnostic).
# Tune it once on the nano ladder (see run_mu_ladder.sh), then reuse the
# SAME value at full width.  If you do NOT pass --mu-transfer, tune LR at the
# full model width directly.
# ---------------------------------------------------------------------------
set -euo pipefail

GPUS="${GPUS:-8}"
DATASET="${DATASET:-data/cad_programs.jsonl}"
MODEL="${MODEL:-small}"
LR="${LR:-1e-3}"
EPOCHS="${EPOCHS:-100}"
OUT="${OUT:-checkpoints/run-fsdp}"
MU="${MU:-1}"

args=(
  --data "$DATASET"
  --out-dir "$OUT"
  --model "$MODEL"
  --fsdp --bf16
  --lr "$LR"
  --epochs "$EPOCHS"
  --packed
)
if [ "$MU" = "1" ]; then args+=(--mu-transfer); fi

echo ">> torchrun --nproc_per_node=$GPUS train.py ${args[*]}"
torchrun --standalone --nproc_per_node="$GPUS" train.py "${args[@]}"

echo ">> done.  checkpoints in $OUT; best = $OUT/best_checkpoint.pt"