#!/usr/bin/env bash
# Launcher for a SMOKE TEST. Overfits on a tiny dataset.
# GOAL: Finish in < 30 minutes to verify the pipeline.

export PYFILE="run/cxr_cond.py"
export RUN_NAME="smoke-test-64px"
export DATA_ROOT="/datasets/mmolefe/cleaned_2/TB"

# --- Drastically reduce everything for speed ---
export IMG_SIZE="64"
# Smaller model
export CHANNELS="32,64,128,128"
# Train for more epochs on the tiny dataset to ensure it can memorize
export EPOCHS="300"
# Can use a larger batch size due to small images/model
export BATCH_PER_DEVICE="16"
export LR="1e-4"

# --- Sample very frequently to see progress ---
export SAMPLE_EVERY="30"
export CKPT_EVERY="50"

# --- Key change: overfit on 32 examples ---
export OVERFIT_ONE="0"
export OVERFIT_K="32"

export WANDB_PROJECT="conditional-cxr-sde"
export WANDB_TAGS="slurm,smoke-test"

sbatch cxr_smoke_test.slurm