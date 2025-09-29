#!/usr/bin/env bash
#SBATCH --job-name=diag-cxr-sde      # Changed job name for clarity
#SBATCH --partition=bigbatch          # Consider using a smaller/debug partition if available
#SBATCH --time=00:30:00               # Reduced time limit to 30 minutes
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# --- Pretty printing helpers ---
RED=$'\e[31m'; GRN=$'\e[32m'; YLW=$'\e[33m'; BLU=$'\e[34m'; CYN=$'\e[36m'; RST=$'\e[0m'; BLD=$'\e[1m'
banner(){ printf "\n${BLU}${BLD}== %s ==${RST}\n" "$*"; }
kv(){ printf "  ${CYN}%-20s${RST} %s\n" "$1" "$2"; }
ok(){ printf "${GRN}%s${RST}\n" "$*"; }

# --- Activate Environment ---
source ~/.bashrc
mamba activate jax115 # Or your conda/mamba env
ok "Activated micromamba environment."

# --- Path and Script Configuration (from launcher) ---
WORKDIR="${WORKDIR:-$PWD}"
PYFILE="${PYFILE:?Must set PYFILE}"
cd "$WORKDIR"
export PYTHONPATH="$WORKDIR"

# --- Training Defaults (will be overridden by launcher) ---
DATA_ROOT="${DATA_ROOT:?Must set DATA_ROOT}"
RUN_NAME="${RUN_NAME:-diagnostic-run}"
IMG_SIZE="${IMG_SIZE:-32}"
NUM_CLASSES="${NUM_CLASSES:-3}"
CHANNELS="${CHANNELS:-64,128,256}"
EPOCHS="${EPOCHS:-10}"
BATCH_PER_DEVICE="${BATCH_PER_DEVICE:-4}"
LR="${LR:-2e-4}"
SAMPLE_EVERY="${SAMPLE_EVERY:-2}"
CKPT_EVERY="${CKPT_EVERY:-5}"
WANDB_PROJECT="${WANDB_PROJECT:-conditional-cxr-sde}"
WANDB_TAGS="${WANDB_TAGS:-slurm,diagnostic}"

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

banner "Starting DIAGNOSTIC Conditional SDE Training"
kv "SLURM Job ID" "${SLURM_JOB_ID}"
kv "Python Script" "$PYFILE"
kv "Dataset Root" "$DATA_ROOT"
kv "Run Name" "${RUN_NAME}-${SLURM_JOB_ID}"
# ... (other printouts)
echo "========================================="

srun python -u "$PYFILE" "${ARGS[@]}"
ok "Job finished successfully."