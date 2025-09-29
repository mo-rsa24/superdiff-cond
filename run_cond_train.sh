#!/usr/bin/env bash

# Launcher for a FULL conditional model training run.
# This is configured for performance to generate plausible images.

export PYFILE="run/cxr_cond.py" # Assumes the python script is in the same directory
export RUN_NAME="conditional-cxr-full-128px-v1"
export DATA_ROOT="/datasets/mmolefe/cleaned_2/TB" # Make sure this path is correct

# --- Performance Settings ---
export IMG_SIZE="128"
export NUM_CLASSES="3" # NORMAL, PNEUMONIA, TB
# A deeper U-Net for better capacity
export CHANNELS="64,128,256,512"

export EPOCHS="1000"
# Adjust batch size based on your GPU memory. 8 is a good start for 128px.
export BATCH_PER_DEVICE="8"
export LR="2e-4"
export SAMPLE_EVERY="25"
export CKPT_EVERY="50"

# --- Use the full dataset ---
export OVERFIT_ONE="0"
export OVERFIT_K="0"

export WANDB_PROJECT="conditional-cxr-sde"
export WANDB_TAGS="slurm,conditional,full-run,128px"

# Submit the job to SLURM using the full training slurm file
sbatch cxr_cond_full.slurm