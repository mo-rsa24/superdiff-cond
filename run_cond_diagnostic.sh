#!/usr/bin/env bash

export PYFILE="cxr_cond.py" # Assumes the python script is in the same directory
export RUN_NAME="conditional-cxr-diagnostic"
export DATA_ROOT="/datasets/mmolefe/cleaned_2/TB" # Make sure this path is correct

# --- Diagnostic Settings (for speed) ---
export IMG_SIZE="32"
export NUM_CLASSES="3"
export CHANNELS="64,128,256"

export EPOCHS="10"
export BATCH_PER_DEVICE="4"
export LR="2e-4"
export SAMPLE_EVERY="2"
export CKPT_EVERY="5"

# --- Overfitting on 16 images ---
export OVERFIT_ONE="0"
export OVERFIT_K="16"

export WANDB_PROJECT="conditional-cxr-sde"
export WANDB_TAGS="slurm,diagnostic,overfit"

sbatch cxr_cond_diagnostic.slurm