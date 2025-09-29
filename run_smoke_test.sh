#!/usr/bin/env bash
# Launcher for a FIDELITY TEST. Overfits on a tiny dataset.
# GOAL: Start to see anatomical structure to verify model capacity.

export PYFILE="run/cxr_cond.py"
export RUN_NAME="fidelity-test-128px" # Updated name
export DATA_ROOT="/datasets/mmolefe/cleaned_2/TB"

# --- Parameters increased for higher fidelity ---
# 💡 1. Increase image size for more detail.
export IMG_SIZE="128"

# 🧠 2. Increase model capacity to learn complex structures.
export CHANNELS="64,128,128,256"

# ⏳ 3. Increase epochs significantly. Diffusion needs time.
export EPOCHS="1500"

# ‼️ 4. Decrease batch size to prevent out-of-memory errors.
export BATCH_PER_DEVICE="4"
export LR="1e-4"

# --- Keep frequent sampling to monitor progress ---
export SAMPLE_EVERY="50"
export CKPT_EVERY="100"

# --- Continue overfitting on 32 examples ---
export OVERFIT_ONE="0"
export OVERFIT_K="32"

export WANDB_PROJECT="conditional-cxr-sde"
export WANDB_TAGS="slurm,fidelity-test"

sbatch cxr_smoke_test.slurm