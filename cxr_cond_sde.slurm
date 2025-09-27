#!/usr/bin/env bash
#SBATCH --job-name=cond-cxr-sde
#SBATCH --partition=bigbatch
#SBATCH --time=72:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# --- Pretty printing helpers ---
RED=$'\e[31m'; GRN=$'\e[32m'; YLW=$'\e[33m'; BLU=$'\e[34m'; RST=$'\e[0m'; BLD=$'\e[1m'
banner(){ printf "\n${BLU}${BLD}== %s ==${RST}\n" "$*"; }
kv(){ printf "  ${CYN}%-20s${RST} %s\n" "$1" "$2"; }
ok(){ printf "${GRN}%s${RST}\n" "$*"; }

# --- Activate Environment ---
source ~/.bashrc
mamba activate jax115 # Or your conda/mamba env
ok "Activated micromamba environment."

# --- Path and Script Configuration (from launcher) ---
WORKDIR="${WORKDIR:-$PWD}"
PYFILE="${PYFILE:-cxr.py}"
cd "$WORKDIR"
export PYTHONPATH="$WORKDIR"

# --- Training Defaults (can be overridden by launcher) ---
DATA_ROOT="${DATA_ROOT:?Must set DATA_ROOT}"
RUN_NAME="${RUN_NAME:-conditional-run}"
IMG_SIZE="${IMG_SIZE:-128}"
NUM_CLASSES="${NUM_CLASSES:-3}"
CHANNELS="${CHANNELS:-64,128,256}"
EPOCHS="${EPOCHS:-200}"
BATCH_PER_DEVICE="${BATCH_PER_DEVICE:-8}"
LR="${LR:-2e-4}"
SAMPLE_EVERY="${SAMPLE_EVERY:-10}"
CKPT_EVERY="${CKPT_EVERY:-20}"
WANDB_PROJECT="${WANDB_PROJECT:-conditional-cxr-sde}"
WANDB_TAGS="${WANDB_TAGS:-slurm,conditional}"

# Overfitting controls
OVERFIT_ONE="${OVERFIT_ONE:-0}"
OVERFIT_K="${OVERFIT_K:-0}"

# --- Build arguments for the Python script ---
ARGS=(
  --data_root "$DATA_ROOT"
  --img_size "$IMG_SIZE"
  --num_classes "$NUM_CLASSES"
  --channels "$CHANNELS"
  --epochs "$EPOCHS"
  --batch_per_device "$BATCH_PER_DEVICE"
  --lr "$LR"
  --sample_every "$SAMPLE_EVERY"
  --ckpt_every "$CKPT_EVERY"
  --exp_name "${RUN_NAME}"
  --run_name "${RUN_NAME}-${SLURM_JOB_ID}"
  --wandb --wandb_project "$WANDB_PROJECT" --wandb_tags "$WANDB_TAGS"
)

# --- Add optional flags ---
if [[ "$OVERFIT_ONE" == "1" ]]; then
  ARGS+=( --overfit_one )
  kv "OVERFITTING" "Single image"
elif (( OVERFIT_K > 0 )); then
  ARGS+=( --overfit_k "$OVERFIT_K" )
  kv "OVERFITTING" "$OVERFIT_K images"
fi


banner "Starting Conditional SDE Training"
kv "SLURM Job ID" "${SLURM_JOB_ID}"
kv "Python Script" "$PYFILE"
kv "Dataset Root" "$DATA_ROOT"
kv "Run Name" "${RUN_NAME}-${SLURM_JOB_ID}"
kv "Epochs" "$EPOCHS"
kv "Image Size" "$IMG_SIZE"
kv "Num Classes" "$NUM_CLASSES"
kv "Batch/Device" "$BATCH_PER_DEVICE"
kv "Sample Every" "$SAMPLE_EVERY epochs"
kv "Checkpoint Every" "$CKPT_EVERY epochs"
echo "----------------------------------------"
echo "Full command:"
echo "srun python -u $PYFILE ${ARGS[@]}"
echo "========================================="


srun python -u "$PYFILE" "${ARGS[@]}"

ok "Job finished successfully."