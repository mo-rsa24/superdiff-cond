# cxr_cond.py (Updated and Unified)
import argparse
import functools
import json
import os
from datetime import datetime

import jax
import jax.numpy as jnp
import numpy as np
import optax
import torch
import tqdm
from flax.serialization import to_bytes
from flax.training.train_state import TrainState
from torch.utils.data import DataLoader
from torchvision.utils import save_image

# --- Local Project Modules ---
from datasets.ChestXRay import ChestXrayDataset
from diffusion.equations import (diffusion_coeff, marginal_prob_std,
                                 vpsde_diffusion_coeff, vpsde_marginal_prob_std)
from diffusion.sampling import Euler_Maruyama_sampler, ode_sampler, pc_sampler
from models.cxr_unet import ScoreNet
from train.train_score_sde import get_train_step_fn

# --- Optional: Weights & Biases ---
try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    _WANDB_AVAILABLE = False


# --- Helper Functions ---
def ensure_dir(p): os.makedirs(p, exist_ok=True); return p
def tree_ema_update(ema, new, decay): return jax.tree_map(lambda e, p: e * decay + (1.0 - decay) * p, ema, new)


def parse_args():
    """Parses command-line arguments for the training run."""
    p = argparse.ArgumentParser("JAX Conditional SDE Chest X-ray Trainer")
    # Data
    p.add_argument("--data_root", default="../datasets/cleaned_2/TB", help="Path to the dataset.")
    p.add_argument("--split", choices=["train", "val", "test"], default="train")
    p.add_argument("--img_size", type=int, default=128)
    p.add_argument("--num_classes", type=int, default=3, help="Number of classes for conditioning.")

    # ✨ ADDED: Overfit/Debug Arguments ✨
    p.add_argument("--overfit_one", action="store_true", help="Overfit on a single image.")
    p.add_argument("--overfit_k", type=int, default=0, help="Overfit on K images. 0 to disable.")
    p.add_argument("--repeat_len", type=int, default=8192, help="Dataset length for overfit_one mode.")

    # Model
    p.add_argument("--channels", type=str, default="64,128,256,512", help="U-Net channel progression.")
    p.add_argument("--embed_dim", type=int, default=256)

    # SDE/Sampler
    p.add_argument("--sde", choices=["VE", "VPSDE"], default="VE", help="Type of SDE to use.")
    p.add_argument("--sigma_max", type=float, default=25.0, help="Maximum sigma for VE SDE.")
    p.add_argument("--sampler", choices=["pc", "em", "ode"], default="em", help="Sampler to use for generation.")
    p.add_argument("--num_steps", type=int, default=500, help="Number of steps for the sampler.")
    p.add_argument("--snr", type=float, default=0.16, help="SNR for Langevin corrector (PC sampler).")
    p.add_argument("--eps", type=float, default=1e-3, help="Final time for samplers.")

    # Training
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_per_device", type=int, default=4)
    p.add_argument("--sample_batch_size", type=int, default=18, help="Must be a multiple of num_classes.")
    p.add_argument("--ema_decay", type=float, default=0.999)

    # Logistics
    p.add_argument("--output_root", default="runs", help="Root directory for experiment outputs.")
    p.add_argument("--exp_name", default="cxr_conditional_sde")
    p.add_argument("--run_name", default=None, help="Specific name for this run.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sample_every", type=int, default=5, help="Frequency of sampling (in epochs).")
    p.add_argument("--ckpt_every", type=int, default=10, help="Frequency of checkpointing (in epochs).")

    # WandB
    p.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    p.add_argument("--wandb_project", default="conditional-cxr-sde")
    p.add_argument("--wandb_entity", default=None)
    p.add_argument("--wandb_tags", default="", help="Comma-separated tags")

    return p.parse_args()


def main():
    args = parse_args()

    # --- 1. Setup & Configuration ---
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = args.run_name or f"{args.exp_name}_{ts}"
    run_dir = ensure_dir(os.path.join(args.output_root, run_name))
    ckpt_dir = ensure_dir(os.path.join(run_dir, "ckpts"))
    samples_dir = ensure_dir(os.path.join(run_dir, "samples"))
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    if args.sde == "VE":
        marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=args.sigma_max)
        diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=args.sigma_max)
    else:
        marginal_prob_std_fn = vpsde_marginal_prob_std
        diffusion_coeff_fn = vpsde_diffusion_coeff

    # --- 2. Dataset and DataLoader ---
    full_ds = ChestXrayDataset(root_dir=args.data_root, split=args.split, img_size=args.img_size)
    if args.num_classes != len(full_ds.class_map):
        raise ValueError(
            f"Mismatch: --num_classes={args.num_classes} but dataset has {len(full_ds.class_map)} classes.")

    # ✨ ADDED: Overfitting Logic ✨
    if args.overfit_one:
        first_img, first_label = full_ds[0]
        class RepeatOne(torch.utils.data.Dataset):
            def __len__(self): return args.repeat_len
            def __getitem__(self, i): return first_img, first_label
        train_ds = RepeatOne()
        print(f"✅ Overfitting on 1 image (label: {first_label}) for {args.repeat_len} steps.")
    elif args.overfit_k > 0:
        train_ds = torch.utils.data.Subset(full_ds, range(args.overfit_k))
        print(f"✅ Overfitting on first {args.overfit_k} images.")
    else:
        train_ds = full_ds

    batch_size = args.batch_per_device * jax.local_device_count()
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True, pin_memory=True)

    # --- 3. Model, Optimizer, and State Initialization ---
    rng = jax.random.PRNGKey(args.seed)
    H, W, C = args.img_size, args.img_size, 1
    channels = tuple(int(c.strip()) for c in args.channels.split(","))

    score_model = ScoreNet(
        marginal_prob_std=marginal_prob_std_fn,
        channels=channels,
        embed_dim=args.embed_dim,
        num_classes=args.num_classes,
    )

    rng, init_rng = jax.random.split(rng)
    fake_x = jnp.ones((1, H, W, C))
    fake_t = jnp.ones((1,))
    fake_y = jnp.zeros((1,), dtype=jnp.int32)
    params = score_model.init({'params': init_rng}, fake_x, fake_t, fake_y)['params']

    tx = optax.adamw(learning_rate=args.lr)
    state = TrainState.create(apply_fn=score_model.apply, params=params, tx=tx)
    ema_params = state.params
    replicated_state = jax.device_put_replicated(state, jax.local_devices())
    train_step_fn = get_train_step_fn(score_model, marginal_prob_std_fn)

    # --- 4. W&B Integration ---
    if args.wandb and _WANDB_AVAILABLE:
        wandb_tags = [t.strip() for t in args.wandb_tags.split(",") if t.strip()]
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            tags=wandb_tags,
            config=vars(args),
            dir=run_dir,
        )

    # --- 5. Training Loop ---
    for epoch in tqdm.trange(args.epochs, desc="Epochs"):
        inner_loop = tqdm.tqdm(loader, desc=f"Training Epoch {epoch + 1}", leave=False)
        for x_pt, y_pt in inner_loop:
            x_np = x_pt.permute(0, 2, 3, 1).numpy()
            y_np = y_pt.numpy()
            sharded_x = x_np.reshape(jax.local_device_count(), -1, H, W, C)
            sharded_y = y_np.reshape(jax.local_device_count(), -1)

            rng, *step_rngs = jax.random.split(rng, jax.local_device_count() + 1)
            step_rngs = jnp.asarray(step_rngs)

            loss, replicated_state = train_step_fn(step_rngs, sharded_x, sharded_y, replicated_state)
            loss_val = loss[0]
            inner_loop.set_postfix(loss=f"{loss_val:.4f}")
            if args.wandb and _WANDB_AVAILABLE:
                wandb.log({"train/loss": loss_val})

            params_host = jax.device_get(jax.tree_map(lambda x: x[0], replicated_state.params))
            ema_params = tree_ema_update(ema_params, params_host, args.ema_decay)

        if (epoch + 1) % args.sample_every == 0:
            sample_and_log(
                rng, score_model, ema_params, args, epoch + 1, samples_dir, full_ds.class_map, diffusion_coeff_fn
            )

        if (epoch + 1) % args.ckpt_every == 0:
            state_host = jax.device_get(jax.tree_map(lambda x: x[0], replicated_state))
            payload = to_bytes((state_host, ema_params))
            path = os.path.join(ckpt_dir, f"epoch_{epoch + 1}.flax")
            with open(path, "wb") as f: f.write(payload)
            print(f"\nSaved checkpoint: {path}")

    if args.wandb and _WANDB_AVAILABLE:
        wandb.finish()


def sample_and_log(rng, model, params, args, epoch, out_dir, class_map, diffusion_coeff_fn):
    """ ✨ REVISED: More robust sampling and logging function. ✨ """
    print(f"\n--- Starting sampling for epoch {epoch} ---")
    H, W = args.img_size, args.img_size
    if args.sample_batch_size % args.num_classes != 0:
        print(f"[Warning] sample_batch_size not divisible by num_classes.")
    samples_per_class = max(1, args.sample_batch_size // args.num_classes)

    all_samples_list = []
    class_names = sorted(class_map, key=class_map.get)
    sampler_fn = {'pc': pc_sampler, 'em': Euler_Maruyama_sampler, 'ode': ode_sampler}[args.sampler]
    # sampler_fn = {'em': Euler_Maruyama_sampler}[args.sampler]

    for i, class_name in enumerate(class_names):
        print(f"Sampling for class '{class_name}' (label {i})...")
        rng, sample_rng = jax.random.split(rng)
        y_cond = jnp.full((samples_per_class,), i, dtype=jnp.int32)

        # Call the appropriate sampler
        samples = sampler_fn(
            sample_rng, model, params,
            marginal_prob_std=model.marginal_prob_std,
            diffusion_coeff=diffusion_coeff_fn,
            batch_size=samples_per_class,
            img_size=args.img_size,
            num_steps=args.num_steps,
            y_cond=y_cond,
            eps=args.eps,
            # Pass sampler-specific args if they exist in `args`
            **({'snr': args.snr} if args.sampler == 'pc' else {})
        )
        all_samples_list.append(samples)

    final_samples = jnp.concatenate(all_samples_list, axis=0)
    final_samples = jnp.clip(final_samples.reshape(-1, H, W, 1), 0.0, 1.0)
    final_samples_pt = torch.tensor(np.asarray(final_samples)).permute(0, 3, 1, 2)

    grid_path = os.path.join(out_dir, f"grid_epoch_{epoch:04d}.png")
    save_image(final_samples_pt, grid_path, nrow=samples_per_class)
    print(f"Saved conditional sample grid to {grid_path}")
    if args.wandb and _WANDB_AVAILABLE:
        wandb.log({"samples/grid": wandb.Image(grid_path), "epoch": epoch})


if __name__ == "__main__":
    import tensorflow as tf
    tf.config.experimental.set_visible_devices([], "GPU")
    main()