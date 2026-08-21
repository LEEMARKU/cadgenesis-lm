#!/bin/bash
#SBATCH --job-name=cadgenesis-train
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=48:00:00
#SBATCH --output=logs/cadgenesis-%j.log
#SBATCH --error=logs/cadgenesis-%j.err
#SBATCH --partition=gpu
#SBATCH --exclusive

set -euo pipefail
mkdir -p logs

export PYTHONPATH="${PWD}/src:${PYTHONPATH}"
export CADGENESIS_OUTPUT_DIR="${PWD}/outputs"
export CADGENESIS_DEVICE=auto

# Distributed launcher (torchrun / DDP / FSDP)
torchrun \
    --nnodes="${SLURM_JOB_NUM_NODES}" \
    --nproc-per-node="${SLURM_GPUS_PER_NODE}" \
    --rdzv-backend=c10d \
    --rdzv-endpoint="${SLURM_JOB_HEAD_NODE}:29500" \
    -m cadgenesis.cli.train \
    --config configs/train.yaml \
    --distributed true
