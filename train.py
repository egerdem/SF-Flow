"""SF-Flow trainer: conditional Flow Matching for 3D ATF magnitude estimation.

Reproduces the paper's R1 models with, e.g.:
  python train.py --model_name SFFlow_r1_freq30 --freq_up_to 30 --num_iterations 400000 \
      --geo_conditioning --freq_ctx --validation_interval 200
(see README for the exact command per frequency range)
"""
import torch
# torch.set_num_threads(4)          # limit PyTorch CPU thread pool
# torch.set_num_interop_threads(2)  # limit inter-op parallelism
from torchvision import transforms
import os
import json
import time
import random
import numpy as np
import wandb
import argparse
from tqdm import tqdm

from fm_utils import (model_factory,
                      ATF3DSampler, GaussianConditionalProbabilityPath,
                      LinearAlpha, LinearBeta,
                      ATF3DTrainer
                      )


def set_all_seeds(seed: int = 42):
    """Seed all RNGs for reproducible training.
    NOTE: does NOT set cudnn.deterministic=True (expensive for conv models).
    cudnn.benchmark=False is enough to prevent non-deterministic algorithm selection.
    """
    torch.manual_seed(seed)          # seeds CPU and CUDA RNGs
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = True   # enable auto-tuner for faster conv kernels


def calculate_and_cache_coord_stats(train_sampler, dataset_version="r1"):
    """
    Calculates and caches the mean and std of relative coordinates for the training set.
    Cache file is specific to the dataset version to ensure correct statistics.
    """
    cache_path = f"coord_stats_{dataset_version}.pt"

    if os.path.exists(cache_path):
        print(f"Loading cached coordinate stats from {cache_path}")
        stats = torch.load(cache_path, map_location='cpu')
        return stats['mean'], stats['std']

    print(f"Calculating coordinate stats for dataset version {dataset_version}... (this may take a moment)")
    all_rel_coords = []

    # Iterate through all training sources to get all possible relative coordinates
    for i in tqdm(range(len(train_sampler.source_coords)), desc="Calculating Stats"):
        src_xyz = train_sampler.source_coords[i].unsqueeze(0).cpu()
        rel_coords = train_sampler.grid_xyz - src_xyz  # Shape [1331, 3]
        all_rel_coords.append(rel_coords)

    # Concatenate all relative coordinates into a single large tensor
    full_coords_tensor = torch.cat(all_rel_coords, dim=0)

    # Calculate mean and std along the sample dimension (dim=0)
    coord_mean = full_coords_tensor.mean(dim=0)
    coord_std = full_coords_tensor.std(dim=0)

    # Cache the results for future runs with dataset version info
    print(f"Saving coordinate stats to {cache_path}")
    print(f"Training set size: {len(train_sampler.source_coords)} sources")
    print(f"Coordinate mean: {coord_mean}")
    print(f"Coordinate std: {coord_std}")
    torch.save({'mean': coord_mean, 'std': coord_std, 'dataset_version': dataset_version,
                'num_sources': len(train_sampler.source_coords)}, cache_path)

    return coord_mean, coord_std


def main(args):
    # Seed all RNGs first, before any data loading or model construction.
    # This makes training batch order, t, x0 draws, and val metrics reproducible.
    set_all_seeds(seed=42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- 1. Initial Config from Arguments ---
    # This creates a baseline config that can be used immediately
    # Determine dataset version and src_splits from model name
    from fm_utils import get_dataset_version_from_model_name, get_src_splits_for_version
    dataset_version = args.dataset_version if args.dataset_version else get_dataset_version_from_model_name(args.model_name)
    src_splits_config = get_src_splits_for_version(dataset_version)

    config = {
        "data": {"data_dir": args.data_dir,
                 "src_splits": src_splits_config},
        "model": {"name": args.model_name, "channels": args.channels, "d_model": args.d_model, "nhead": args.nhead,
                  "num_encoder_layers": args.num_encoder_layers, "freq_up_to": args.freq_up_to, "freq_from": args.freq_from,
                  "architecture_version": args.version, "setencoder_version": args.setencoder_version,
                  "FM_vs_Diff": args.FM_vs_Diff,
                  "init_kernel_size": args.init_kernel_size,
                  "coord_dim": 9 if args.geo_conditioning else 3,
                  "num_ff_coord": args.num_ff_coord,
                  "freq_channel_bias": args.freq_channel_bias,
                  "freq_film": args.freq_film,
                  "freq_ctx": args.freq_ctx},
        "training": {"num_iterations": args.num_iterations, "batch_size": args.batch_size, "lr": args.lr,
                     "warmup_iterations": args.warmup_iterations, "decay_iterations": args.decay_iterations,
                     "min_lr": args.min_lr,
                     "M_range": args.M_range, "M_val_fixed": args.M_val_fixed, "eta": args.eta, "sigma": args.sigma,
                     "loss_type": args.loss_type, "freq_weight_max": args.freq_weight_max,
                     "validation_interval": args.validation_interval,
                     "lsd_validation_interval": args.lsd_validation_interval or args.validation_interval,
                     "idx_mes_pos_path": args.idx_mes_pos_path,
                     "geo_conditioning": args.geo_conditioning},
        "experiments_dir": args.experiments_dir
    }

    # --- 2. Handle Resuming ---
    start_iteration = 0
    resume_checkpoint_state = None
    experiment_dir = None
    experiment_name = ""

    if args.resume_from_checkpoint:
        if os.path.exists(args.resume_from_checkpoint):
            print(f"RESUMING training from: {args.resume_from_checkpoint}")
            resume_checkpoint_state = torch.load(args.resume_from_checkpoint, map_location=device)
            start_iteration = resume_checkpoint_state.get('iteration', 0)
            print(f"  → Resuming from iteration {start_iteration}")

            # Reuse the original experiment directory so model is saved in the same place.
            _ckpt_parent = os.path.dirname(args.resume_from_checkpoint)
            experiment_dir = (
                os.path.dirname(_ckpt_parent)
                if os.path.basename(_ckpt_parent) == 'checkpoints'
                else _ckpt_parent
            )
            experiment_name = os.path.basename(experiment_dir)
            print(f"  → Experiment dir: {experiment_dir}")

            # Resume W&B run (continues the existing run's graphs)
            if args.wandb:
                if args.wandb_key:
                    wandb.login(key=args.wandb_key)
                else:
                    wandb.login()  # uses WANDB_API_KEY env var or cached credentials
                run_id = resume_checkpoint_state.get('wandb_run_id')
                # Fallback: read run_id from config.json saved in the experiment dir
                if run_id is None:
                    _config_json = os.path.join(experiment_dir, 'config.json')
                    if os.path.exists(_config_json):
                        with open(_config_json) as _f:
                            run_id = json.load(_f).get('wandb_run_id')
                        if run_id:
                            print(f"  → W&B run ID loaded from config.json: {run_id}")
                wandb.init(project="sf-flow", name=experiment_name, id=run_id, resume="allow", config=config)
                config['wandb_run_id'] = wandb.run.id
                print(f"  → Resuming W&B run ID: {wandb.run.id}")
        else:
            print(f"⚠️  Warning: resume path does not exist: {args.resume_from_checkpoint}")

    if experiment_dir is None:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        experiment_name = f"{args.model_name}_{timestamp}_iter{args.num_iterations}"
        experiment_dir = os.path.join(args.experiments_dir, experiment_name)
        os.makedirs(experiment_dir, exist_ok=True)
        print("\n--- NEW EXPERIMENT ---")

        if args.wandb:
            if args.wandb_key:
                wandb.login(key=args.wandb_key)
            else:
                wandb.login()  # uses WANDB_API_KEY env var or cached credentials
            run = wandb.init(project="sf-flow", name=experiment_name, config=config)
            config['wandb_run_id'] = run.id

        # Config will be saved after data loading to include any corrections
        print(f"Experiment setup. Config will be saved after data loading.")

    MODEL_SAVE_PATH = os.path.join(experiment_dir, "model.pt")
    CHECKPOINT_DIR = os.path.join(experiment_dir, "checkpoints")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # --- Data Loading ---
    data_cfg = config['data']
    model_cfg = config['model']
    training_cfg = config['training']

    # 1. Create the training sampler. It will calculate and apply its own normalization.
    print("--- Loading Training Data ---")
    atf_train_sampler = ATF3DSampler(
        data_path=data_cfg['data_dir'],
        mode='train',
        src_splits=data_cfg['src_splits'],
        freq_up_to=model_cfg['freq_up_to'],
        freq_from=model_cfg.get('freq_from', 0),
        normalize=True,
        model_name=args.model_name
    )

    # 2. Create the validation sampler, but load the data RAW (normalize=False).
    print("\n--- Loading Validation Data ---")
    atf_valid_sampler = ATF3DSampler(
        data_path=data_cfg['data_dir'],
        mode='valid',
        src_splits=data_cfg['src_splits'],
        freq_up_to=model_cfg['freq_up_to'],
        freq_from=model_cfg.get('freq_from', 0),
        normalize=False,
        model_name=args.model_name
    )

    # 3. Save the corrected config after data loading (in case src_splits were updated)
    if experiment_dir:
        with open(os.path.join(experiment_dir, "config.json"), 'w') as f:
            json.dump(config, f, indent=4)
        print(f"Final config saved to {experiment_dir}/config.json")

    # 3. Manually apply the TRAINING stats to the VALIDATION data.
    print("Normalizing validation data using training set statistics...")
    atf_valid_sampler.cubes = (atf_valid_sampler.cubes - atf_train_sampler.mean) / (atf_train_sampler.std + 1e-8)

    # 4. Store the stats on the validation sampler object for consistency.
    atf_valid_sampler.mean = atf_train_sampler.mean
    atf_valid_sampler.std = atf_train_sampler.std

    # 5. Move data tensors to GPU — eliminates CPU→GPU copy every training step.
    #    cubes/source_coords are plain tensors so must be moved manually.
    #    .to(device) on the module moves the dummy Buffer so sample() returns GPU tensors.
    for sampler in [atf_train_sampler, atf_valid_sampler]:
        sampler.cubes = sampler.cubes.to(device)
        sampler.source_coords = sampler.source_coords.to(device)
        sampler.to(device)  # moves dummy Buffer so sample() returns GPU tensors

    # Calculate (or load) coordinate statistics
    # The cache file will be created in the same directory you run the script from.
    coord_mean, coord_std = calculate_and_cache_coord_stats(atf_train_sampler, dataset_version)
    print(f"Using Coordinate Stats -> Mean: {coord_mean.cpu().numpy()}, Std: {coord_std.cpu().numpy()}")

    # --- Model and Trainer Initialization ---

    # Get cube shape from the sampler for the probability path
    cube_shape = atf_train_sampler.cubes.shape[1:]

    path = GaussianConditionalProbabilityPath(
        p_data=atf_train_sampler,
        p_simple_shape=list(cube_shape),
        alpha=LinearAlpha(),
        beta=LinearBeta()
    ).to(device)

    set_encoder, unet_3d = model_factory(config, device)

    # Parse room dimensions from data_dir path (e.g. "...room4.0x6.0x3.0_rt200")
    import re as _re
    _m = _re.search(r'room(\d+\.?\d*)x(\d+\.?\d*)x(\d+\.?\d*)', args.data_dir)
    room_dims = (float(_m.group(1)), float(_m.group(2)), float(_m.group(3))) if _m else None
    if args.geo_conditioning:
        if room_dims is None:
            raise ValueError("--geo_conditioning requires room dims in data_dir path (e.g. room4.0x6.0x3.0)")
        print(f"--- Geo conditioning ON: room_dims={room_dims}, coord_dim=9 (rel[3]+d_walls[6]) ---")
    else:
        print("--- Geo conditioning OFF: coord_dim=3 (relative only) ---")

    trainer = ATF3DTrainer(
        path=path,
        model=unet_3d,
        set_encoder=set_encoder,
        eta=training_cfg['eta'],
        M_range=training_cfg['M_range'],
        M_val_fixed=training_cfg['M_val_fixed'],
        sigma=training_cfg['sigma'],
        loss_type=training_cfg.get('loss_type', 'standard'),
        freq_weight_max=training_cfg.get('freq_weight_max', 3.0),
        FM_vs_Diff=model_cfg['FM_vs_Diff'],
        grid_xyz=atf_train_sampler.grid_xyz,
        version=model_cfg.get("architecture_version"),
        setencoderversion=model_cfg.get("setencoder_version"),
        coord_mean=coord_mean,
        coord_std=coord_std,
        idx_mes_pos_path=training_cfg.get("idx_mes_pos_path"),
        geo_conditioning=args.geo_conditioning,
        room_dims=room_dims,
        sweep_M=args.sweep_M,
        time_weight_towards_end=args.time_weight_towards_end,
        time_weight_mean=args.time_weight_mean,
    )

    training_cfg['warmup_iterations'] = args.warmup_iterations
    training_cfg['min_lr'] = args.min_lr

    # --- Training ---
    print(f"\n--- Starting Training for experiment: {experiment_name} ---")
    trainer.train(
        num_iterations=training_cfg['num_iterations'],
        device=device,
        lr=training_cfg['lr'],
        warmup_iterations=training_cfg['warmup_iterations'],
        decay_iterations=training_cfg['decay_iterations'],
        min_lr=training_cfg['min_lr'],
        batch_size=training_cfg['batch_size'],
        valid_sampler=atf_valid_sampler,
        save_path=MODEL_SAVE_PATH,
        checkpoint_path=CHECKPOINT_DIR,
        checkpoint_interval=args.checkpoint_interval,
        validation_interval=training_cfg['validation_interval'],
        lsd_validation_interval=training_cfg['lsd_validation_interval'],
        start_iteration=start_iteration,
        config=config,
        resume_checkpoint_state=resume_checkpoint_state,
        save_start_iter=args.save_start_iter
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="SF-Flow — 3D ATF Trainer")

    # --- Resuming ---
    parser.add_argument('--resume_from_checkpoint', type=str, help='Path to a checkpoint to resume from.')

    # --- WandB ---
    parser.add_argument('--wandb', action=argparse.BooleanOptionalAction, default=False,
                        help='Enable Weights & Biases logging.')
    parser.add_argument('--wandb_key', type=str, default=None,
                        help='W&B API key. If omitted, uses WANDB_API_KEY env var or cached login.')

    # --- Data ---
    parser.add_argument('--data_dir', type=str, default="data/ir_fs2000_s8192_m1331_room4.0x6.0x3.0_rt200/")

    # --- Model ---
    parser.add_argument('--model_name', default="SFFlow", type=str)
    parser.add_argument('--dataset_version', type=str, default=None,
                        help='Override dataset split version (e.g. r1val52). Defaults to inferring from model name.')
    parser.add_argument('--channels', type=lambda s: [int(item) for item in s.split(',')], default=[256,512,1024])
    parser.add_argument('--d_model', type=int, default=512, help='Dimension for tokens and context.')
    parser.add_argument('--nhead', type=int, default=8, help='Number of attention heads.')
    parser.add_argument('--num_encoder_layers', type=int, default=3, help='Layers in the SetEncoder.')
    parser.add_argument('--init_kernel_size', type=int, default=3,
                        help='Initial input Conv3d kernel size for UNet (v1/v2/v3). '
                             'Default 3 keeps current behavior exactly; set e.g. 6 to test larger receptive field.')
    parser.add_argument('--num_ff_coord', type=int, default=0,
                        help='Fourier Feature Mapping for coordinates. 0=disabled (raw coords, legacy). '
                             '>0: coords mapped to 2*num_ff_coord dims via sin/cos before positional MLP. '
                             'E.g. --num_ff_coord 16 with coord_dim=9 gives v:[16,9], output 32-dim.')
    parser.add_argument('--freq_channel_bias', action='store_true', default=False,
                        help='Add a learnable per-frequency channel bias to the UNet input. '
                             'Acts as a positional encoding over the frequency axis (~num_freqs params). '
                             'Tells the network explicitly which channel = which frequency.')
    parser.add_argument('--freq_film', action='store_true', default=False,
                        help='FiLM conditioning: project SetEncoder pooled_context to per-frequency '
                             'scale+shift applied to UNet input before first conv. '
                             'Observation-conditional frequency modulation. ~2*F*d_model extra params.')
    parser.add_argument('--freq_ctx', action='store_true', default=False,
                        help='Per-frequency context FiLM: for each (mic, freq) pair embed '
                             '[coords, freq_idx/F, atf_val] via MLP, mean-pool over M mics → '
                             '[B, F, d_model] contexts applied as per-frequency FiLM in UNet. '
                             'True frequency-specific conditioning. ~2*d_model^2 extra params in encoder MLP.')

    # --- Training ---
    parser.add_argument('--num_iterations', type=int, default=400000)
    parser.add_argument('--batch_size', type=int, default=4)  # NOTE: Must be small for 3D models
    parser.add_argument('--lr', type=float, default=1e-4, help="now it is peak learning rate after warm-up.")
    parser.add_argument('--warmup_iterations', type=int, default=5000,
                        help="Number of iterations for linear LR warm-up.")
    parser.add_argument('--decay_iterations', type=int, default=100000,
                        help="Number of iterations for the cosine decay phase. The rest will be constant min_lr.")
    parser.add_argument('--min_lr', type=float, default=1e-5,
                        help="The minimum learning rate at the end of the cosine decay.")
    parser.add_argument('--M_range', type=lambda s: [int(item) for item in s.split(',')], default=[5,10,20,50])
    parser.add_argument('--M_val_fixed', type=int, default= 5)
    parser.add_argument('--save_start_iter', type=int, default=0,
                        help='Skip saving model checkpoints until this iteration. Avoids frequent saves early in training.')

    parser.add_argument('--freq_up_to', type=int, default=64, help='Upper freq bin (exclusive upper bound of subband)')

    parser.add_argument('--freq_from', type=int, default=0, help='Lower freq bin (inclusive). Default 0 = full range from DC. Set e.g. 20 to train on bins 20..freq_up_to only.')
    parser.add_argument('--eta', type=float, help='Probability for CFG dropout.', default=0.)
    parser.add_argument('--sigma', type=float, help='Sigma for noise in the path.', default=0)
    parser.add_argument('--loss_type', type=str, default='standard',
                        choices=['standard', 'weighted', 'freq_weighted'],
                        help='Loss type: standard MSE, perceptual amplitude-weighted, or freq-bin-weighted.')
    parser.add_argument('--freq_weight_max', type=float, default=3.0,
                        help='For --loss_type freq_weighted: weight applied to the highest freq bin '
                             '(bin 0 always has weight 1.0). Default 3.0.')
    parser.add_argument('--idx_mes_pos_path', type=str, default= "idx_mes_pos_s8192_m1331.npy",
                        help='Path to the deterministic mic permutation matrix for validation.')

    parser.add_argument('--FM_vs_Diff', type=str, default='flow_matching', choices=['flow_matching', 'score_matching'])
    parser.add_argument('--checkpoint_interval', type=int, default=200000,
                        help='Iterations between periodic full-state (model+optimizer, ~1.4 GB) checkpoints '
                             'written to checkpoints/, used for resuming. The best model is saved separately '
                             'whenever validation improves, so this can safely be large.')
    parser.add_argument('--sweep_M', action='store_true', default=False,
                        help='If set, loop over all M values in M_range per step and average losses '
                             '(AE-style training). Default: one random M per step.')
    parser.add_argument('--time_weight_towards_end', action='store_true', default=False,
                        help='If set, sample timesteps from logit-normal distribution biased towards t=1 (data end).')
    parser.add_argument('--time_weight_mean', type=float, default=1.2,
                        help='Mean of the logit-normal distribution for timestep sampling (default=1.2, E[t]≈0.77).')
    parser.add_argument('--geo_conditioning', action='store_true', default=False,
                        help='If set, augment the set encoder coordinate input from 3D (relative only) '
                             'to 9D: [rel_pos(3), src-to-wall distances(6)]. '
                             'Requires room dims to be encoded in --data_dir path.')
    parser.add_argument('--validation_interval', type=int, default=200,
                        help='Iterations between validation passes.')
    parser.add_argument('--lsd_validation_interval', type=int, default=None,
                        help='How often to compute full LSD rollout. Defaults to validation_interval.')
    parser.add_argument('--version', type=str, default="v2_residual_context",
                        help='Model architecture version, options: v1_legacy, v2_residual_context, v3_attention')
    parser.add_argument('--setencoder_version', type=str, default="v3",
                        help='setencoder architecture version, e.g. v12:merged feature, v3:pos embed')

    # --- Paths ---
    parser.add_argument('--experiments_dir', type=str, default="experiments")

    args = parser.parse_args()
    if (args.M_val_fixed is None) != (args.idx_mes_pos_path is None):
        parser.error("--M_val_fixed and --idx_mes_pos_path must both be provided together or both omitted.")
    main(args)
