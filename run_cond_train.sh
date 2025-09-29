#!/usr/bin/env bash

# Launcher for a full conditional model training run.
# Edit these variables to configure your run.

export PYFILE="run/cxr_cond.py"
export RUN_NAME="conditional-cxr-full-v1"
export DATA_ROOT="/datasets/mmolefe/cleaned_2/TB"

export IMG_SIZE="128"
export NUM_CLASSES="3" # NORMAL, PNEUMONIA, TB
export CHANNELS="64,128,256"

export EPOCHS="250"
export BATCH_PER_DEVICE="8"
export LR="2e-4"
export SAMPLE_EVERY="10"
export CKPT_EVERY="20"

# Set to 0 to disable overfitting
export OVERFIT_ONE="0"
export OVERFIT_K="0"

export WANDB_PROJECT="conditional-cxr-sde"
export WANDB_TAGS="slurm,conditional,full-run"

# Submit the job to SLURM
sbatch cxr_cond_sde.slurm