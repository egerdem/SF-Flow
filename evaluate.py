import torch
import numpy as np
import os
import json
import glob
import hashlib
import time
from tqdm import tqdm
import argparse as _ap
_parser = _ap.ArgumentParser(description="Evaluate an SF-Flow checkpoint on the test set.")
_parser.add_argument('--model_path', type=str, default=None,
                     help="Path to a trained SF-Flow checkpoint (.pt). Required.")
_parser.add_argument('--data_dir', type=str, default="data/ir_fs2000_s8192_m1331_room4.0x6.0x3.0_rt200/")
_parser.add_argument('--M', type=int, nargs='+', default=None)
_parser.add_argument('--timing_warmup_sources', type=int, default=5)
_parser.add_argument('--disable_timing', action='store_true', default=False)
_parser.add_argument('--use_cache', action='store_true', default=False,
                     help="Reuse matching prediction caches instead of running inference. "
                          "Without this flag inference runs from scratch (caches are still saved).")
_cli_args, _ = _parser.parse_known_args()
if _cli_args.model_path is None:
    _parser.error("--model_path is required (path to a trained SF-Flow checkpoint .pt)")

import os
if _cli_args.model_path is not None:
    os.environ['MPLBACKEND_HEADLESS'] = '1'

import matplotlib
if _cli_args.model_path is not None:
    matplotlib.use('Agg')
else:
    matplotlib.use('Qt5Agg', force=True)
from matplotlib import pyplot as plt
from inference import model_factory, load_model_and_config
MULTI_MODEL_PATHS = [_cli_args.model_path]

# Your model imports
from fm_utils import (
    ATF3DSampler, SetEncoder,
    CrossAttentionUNet3D, CrossAttentionUNet3D_RED3d,
    CFGVectorFieldODE_3D, CFGVectorFieldODE_3D_V2, EulerSimulator,
    get_model_info, print_model_info
)

from inference import calculate_lsd_unified

# Set seed for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

def load_gt_from_sflow_sampler(model_config, num_sources_eval=None, data_dir=None):
    """Build atf_mag_gt [1331, F, Src] from the SFlow test sampler — no FSMPAE needed."""
    freq_up_to = model_config['model'].get('freq_up_to')
    freq_from  = model_config['model'].get('freq_from', 0)
    src_splits = model_config['data']['src_splits']
    _data_dir  = data_dir or model_config['data'].get('data_dir', 'ir_fs2000_s8192_m1331_room4.0x6.0x3.0_rt200/')
    model_name = model_config['model'].get('name')

    train_sampler = ATF3DSampler(data_path=_data_dir, mode='train', src_splits=src_splits,
                                 normalize=True, freq_up_to=freq_up_to, freq_from=freq_from, model_name=model_name)
    test_sampler  = ATF3DSampler(data_path=_data_dir, mode='test',  src_splits=src_splits,
                                 normalize=False, freq_up_to=freq_up_to, freq_from=freq_from, model_name=model_name)
    test_sampler.cubes = (test_sampler.cubes - train_sampler.mean) / (train_sampler.std + 1e-8)

    n_src   = len(test_sampler)
    if num_sources_eval is not None:
        n_src = min(num_sources_eval, n_src)
    mean_db = train_sampler.mean.item()
    std_db  = train_sampler.std.item()
    n_freq  = freq_up_to - freq_from  # bins in each cube

    gt_list = []
    for i in range(n_src):
        cube_db = test_sampler.cubes[i] * std_db + mean_db  # [n_freq, D, H, W], denormalized
        flat    = cube_db.view(n_freq, -1).T.contiguous()   # [1331, n_freq]
        gt_list.append(flat)
    atf_mag_gt_sflow = torch.stack(gt_list, dim=2)  # [1331, n_freq, n_src]
    print(f"GT loaded from SFlow sampler: {atf_mag_gt_sflow.shape} (Mic, Freq, Src)")
    return atf_mag_gt_sflow

def evaluate_your_model(set_encoder, ode_3d, config, M_values, device, num_sources_eval=None,
                        guidance_scales=None, random_M_sampling=False, model_name=None,
                        normalize_coords=False, coord_mean=None, coord_std=None,
                        eval_freq_up_to=None, model_path=None, data_dir=None,
                        timing_warmup_sources=5, enable_inference_timing=True):
    """
    Evaluate your 3D model.

    Args:
        guidance_scales: List of guidance scale values to evaluate. If None, defaults to [1.0, 2.0].
    """

    if data_dir is None:
        data_dir = "ir_fs2000_s8192_m1331_room4.0x6.0x3.0_rt200/"
    src_split = config['data']['src_splits']
    model_freq_up_to = config['model'].get('freq_up_to')
    freq_from  = config['model'].get('freq_from', 0)
    if eval_freq_up_to is None:
        eval_freq_up_to = model_freq_up_to
    if eval_freq_up_to > model_freq_up_to:
        raise ValueError(
            f"eval_freq_up_to={eval_freq_up_to} cannot be greater than "
            f"model_freq_up_to={model_freq_up_to} for model '{model_name}'."
        )
    print(f"  SFlow model_freq_up_to={model_freq_up_to}, eval_freq_up_to={eval_freq_up_to}")

    # Detect geo_conditioning from checkpoint config and parse room dims.
    _geo = config.get('training', {}).get('geo_conditioning', False)
    _room_dims = None
    if _geo:
        import re as _re_g
        _dim_source = config.get('data', {}).get('data_dir', data_dir)
        _rm = _re_g.search(r'room(\d+\.?\d*)x(\d+\.?\d*)x(\d+\.?\d*)', _dim_source or '')
        if _rm:
            _room_dims = (float(_rm.group(1)), float(_rm.group(2)), float(_rm.group(3)))
            print(f"  Geo-conditioning active: room_dims={_room_dims}, coord_dim=9")
        else:
            print("  WARNING: geo_conditioning=True but room dims not found in data_dir. Using rel-only coords.")
            _geo = False
    
    # Load data. Training-set statistics (mean/std) normalise the test cubes,
    # since the model was optimised against that normalisation scale.
    train_sampler = ATF3DSampler(
        data_path=data_dir, mode='train', src_splits=src_split,
        normalize=True, freq_up_to=model_freq_up_to, freq_from=freq_from, model_name=model_name
    )
    test_sampler = ATF3DSampler(
        data_path=data_dir, mode='test', src_splits=src_split,
        normalize=False, freq_up_to=model_freq_up_to, freq_from=freq_from, model_name=model_name
    )
    test_sampler.cubes = (test_sampler.cubes - train_sampler.mean) / (train_sampler.std + 1e-8)
    
    # Limit evaluation to first N sources if specified
    total_sources = len(test_sampler)
    if num_sources_eval is not None:
        eval_sources = min(num_sources_eval, total_sources)
        print(f"Evaluating on first {eval_sources} sources (out of {total_sources})")
    else:
        eval_sources = total_sources
    
    grid_xyz = train_sampler.grid_xyz.to(device)
    spec_std = train_sampler.std.item()
    
    simulator = EulerSimulator(ode=ode_3d)
    results = {}

    def _sync_if_cuda():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
    
    # Load the SAME microphone selection strategy as reference model
    idx_mes_pos_path = "idx_mes_pos_s1024_m1331.npy"
    idx_mes_pos_mat = np.load(idx_mes_pos_path)
    print(f"Loaded reference microphone selection matrix: {idx_mes_pos_mat.shape}")
    print("Using source-specific microphone selection (different mic subsets per source)")
    model_dir_for_cache = os.path.dirname(os.path.abspath(model_path)) if model_path else None

    # Precompute denormalized GT in [Mic, Freq, Src] for fast cached-metric evaluation.
    gt_denorm_mic = []
    for i_src in range(eval_sources):
        z_true = test_sampler.cubes[i_src].unsqueeze(0).to(device)  # [1,F,D,H,W] normalized
        z_true_denorm = z_true * spec_std + train_sampler.mean.item()
        z_true_flat = z_true_denorm[:, :eval_freq_up_to].view(eval_freq_up_to, -1).T.contiguous()  # [1331, F]
        gt_denorm_mic.append(z_true_flat.cpu())
    gt_denorm_mic = torch.stack(gt_denorm_mic, dim=2)  # [1331, F, S]

    def _compute_metrics_from_predictions(pred_mic_freq_src):
        # pred_mic_freq_src: [1331, F, S], denormalized dB domain
        lsd_scores_local = []
        per_source_local = {}
        m_fundamental_indices = [0, 272, 665, 937, 1330]

        for src_idx in range(eval_sources):
            pred = pred_mic_freq_src[:, :eval_freq_up_to, src_idx]
            gt = gt_denorm_mic[:, :eval_freq_up_to, src_idx]

            lsd_db = calculate_lsd_unified(pred, gt, freq_dim=1).item()
            mse = torch.mean((pred - gt) ** 2).item()
            gt_var = torch.var(gt).item()
            nmse_linear = mse / gt_var if gt_var > 0 else float('inf')
            nmse = 10 * np.log10(nmse_linear) if nmse_linear > 0 and nmse_linear != float('inf') else float('inf')

            pred_mf = pred[m_fundamental_indices, :]
            gt_mf = gt[m_fundamental_indices, :]
            lsd_m_fund_db = calculate_lsd_unified(pred_mf, gt_mf, freq_dim=1).item()

            source_errors = {
                'lsd': lsd_db, 'nmse': nmse,
                'lsd_m_fund': lsd_m_fund_db
            }
            lsd_scores_local.append(source_errors)
            per_source_local[src_idx] = source_errors
        return lsd_scores_local, per_source_local

    for M in M_values:
        results[M] = {}
        print(f"Evaluating your model with M={M} microphones...")
        
        for w in guidance_scales:
            results[M][w] = {}  # Initialize the dictionary for this guidance scale
            print(f"  Using guidance scale w={w}")
            lsd_scores = []
            timing_ms_per_source = []
            timed_loop_wall_start = None
            cache_meta = None
            cache_path = None
            if model_dir_for_cache is not None:
                cache_meta = _build_atf_cache_meta(
                    model_path=model_path,
                    guidance=w,
                    M=M,
                    num_sources_eval=eval_sources,
                    model_freq_up_to=model_freq_up_to,
                    eval_freq_up_to=eval_freq_up_to,
                    random_M_sampling=random_M_sampling,
                    config=config,
                    data_dir=data_dir,
                )
                cache_path = os.path.join(model_dir_for_cache, _atf_cache_filename(cache_meta))
                loaded_path = None
                if _cli_args.use_cache:
                    loaded_path, cached_predictions = _load_matching_atf_cache(model_dir_for_cache, cache_meta)
                    if cached_predictions is not None and w in cached_predictions:
                        print(f"  ⚡ Using cached predictions for M={M}, w={w} from {loaded_path}")
                        pred_cached = cached_predictions[w].float().cpu()
                        lsd_scores, per_source_cached = _compute_metrics_from_predictions(pred_cached)
                        results[M][w]['per_source_errors'] = per_source_cached
                    else:
                        loaded_path = None

            if cache_meta is not None and 'per_source_errors' in results[M][w]:
                # Cache hit path: metrics already prepared from cached predictions.
                pass
            else:
                pred_cache_tensor = torch.zeros((1331, eval_freq_up_to, eval_sources), dtype=torch.float32)
            # Evaluate each source with this guidance scale
                if enable_inference_timing:
                    _sync_if_cuda()
                    timed_loop_wall_start = time.perf_counter()
                for i in tqdm(range(eval_sources), desc=f"Your Model M={M}, w={w}"):
                    with torch.no_grad():
                        z_true = test_sampler.cubes[i].unsqueeze(0).to(device)
                        src_xyz = test_sampler.source_coords[i].unsqueeze(0).to(device)

                        # or choose randomly
                        if random_M_sampling:
                            source_specific_indices = torch.randperm(grid_xyz.shape[0])[:M]
                        else: # Use source-specific microphones (different subsets for each source)
                            source_specific_indices = idx_mes_pos_mat[:M, i]  # First M mics for this source

                        obs_indices = torch.tensor(source_specific_indices, dtype=torch.long, device=device)

                        obs_xyz_abs = grid_xyz[obs_indices]
                        obs_coords_rel = (obs_xyz_abs - src_xyz)  # [M, 3]
                        if normalize_coords and coord_mean is not None and coord_std is not None:
                            _cm = coord_mean.to(device)
                            _cs = coord_std.to(device)
                            obs_coords_rel = (obs_coords_rel - _cm) / (_cs + 1e-8)
                        obs_coords_rel = obs_coords_rel.unsqueeze(0)  # [1, M, 3]

                        # Geo conditioning: append 6 wall distances if model was trained with --geo_conditioning
                        if _geo and _room_dims is not None:
                            _Lx, _Ly, _Lz = _room_dims
                            _half_min = min(_Lx, _Ly, _Lz) / 2.0
                            _d_walls = torch.stack([
                                src_xyz[:, 0],      _Lx - src_xyz[:, 0],
                                src_xyz[:, 1],      _Ly - src_xyz[:, 1],
                                src_xyz[:, 2],      _Lz - src_xyz[:, 2],
                            ], dim=1) / _half_min  # [1, 6]
                            obs_coords_rel = torch.cat([
                                obs_coords_rel,
                                _d_walls.unsqueeze(1).expand(-1, M, -1)  # [1, M, 6]
                            ], dim=-1)  # [1, M, 9]

                        z_flat = z_true.view(z_true.shape[1], -1)
                        obs_values = z_flat[:, obs_indices].transpose(0, 1).unsqueeze(0)
                        obs_mask = torch.ones(1, M, dtype=torch.bool, device=device)

                        # Inference
                        x0 = torch.randn_like(z_true)
                        if enable_inference_timing:
                            _sync_if_cuda()
                            _tic = time.perf_counter()
                        y_tokens, pooled_context, freq_contexts = set_encoder(obs_coords_rel, obs_values, obs_mask)

                        ts = torch.linspace(0, 1, 11, device=device)
                        ts = ts.view(1, -1, 1, 1, 1, 1).expand(x0.shape[0], -1, -1, -1, -1, -1)

                        simulator.ode.guidance_scale = w

                        z_est = simulator.simulate(x0, ts, x0=x0, z_true=z_true, y_tokens=y_tokens,
                                                 obs_mask=obs_mask, pooled_context=pooled_context,
                                                 freq_contexts=freq_contexts,
                                                 paste_observations=True, obs_indices=obs_indices)
                        if enable_inference_timing:
                            _sync_if_cuda()
                            timing_ms_per_source.append((time.perf_counter() - _tic) * 1e3)

                        # Calculate MSE and LSD in denormalized (dB) domain
                        z_est_denorm = z_est * spec_std + train_sampler.mean.item()
                        z_true_denorm = z_true * spec_std + train_sampler.mean.item()
                        z_est_denorm_eval = z_est_denorm[:, :eval_freq_up_to]
                        z_true_denorm_eval = z_true_denorm[:, :eval_freq_up_to]
                        mse = torch.mean((z_est_denorm_eval - z_true_denorm_eval) ** 2).item()

                        # Calculate NMSE (Normalized MSE) in dB
                        z_true_var = torch.var(z_true_denorm_eval).item()
                        nmse_linear = mse / z_true_var if z_true_var > 0 else float('inf')
                        nmse = 10 * np.log10(nmse_linear) if nmse_linear > 0 and nmse_linear != float('inf') else float('inf')

                        # LSD directly on denormalized (dB domain) data
                        lsd_db = calculate_lsd_unified(
                            z_est_denorm_eval.squeeze(0), z_true_denorm_eval.squeeze(0), freq_dim=0
                        ).item()

                        # M_fundamental evaluation (5 specific positions: [0, 272, 665, 937, 1330])
                        m_fundamental_indices = [0, 272, 665, 937, 1330]

                        # Convert 3D cube to flat format [freq, 1331] for indexing
                        z_est_flat = z_est_denorm_eval.view(z_est_denorm_eval.shape[1], -1)  # [eval_freq, 1331]
                        z_true_flat = z_true_denorm_eval.view(z_true_denorm_eval.shape[1], -1)  # [eval_freq, 1331]

                        lsd_m_fund_db = calculate_lsd_unified(
                            z_est_flat[:, m_fundamental_indices].T,  # [5, freq] denormalized
                            z_true_flat[:, m_fundamental_indices].T,  # [5, freq] denormalized
                            freq_dim=1  # frequency is now dim=1
                        ).item()

                        # Store per-source errors
                        source_errors = {
                            'lsd': lsd_db, 'nmse': nmse,
                            'lsd_m_fund': lsd_m_fund_db
                        }
                        lsd_scores.append(source_errors)

                        # Store in per-source dictionary
                        if 'per_source_errors' not in results[M][w]:
                            results[M][w]['per_source_errors'] = {}
                        results[M][w]['per_source_errors'][i] = source_errors

                        # Save denormalized prediction in [1331, eval_freq, src]
                        pred_cache_tensor[:, :, i] = z_est_denorm_eval.squeeze(0).view(eval_freq_up_to, -1).T.cpu()

                if cache_path is not None and cache_meta is not None:
                    _save_atf_cache(cache_path, cache_meta, {w: pred_cache_tensor})
                    print(f"  ✅ Saved cache for M={M}, w={w} to {cache_path}")
        
            # Extract LSD and NMSE values for this guidance scale
            if not lsd_scores and 'per_source_errors' in results[M][w]:
                # Cache-hit path populated only per_source_errors; materialize list view.
                lsd_scores = [results[M][w]['per_source_errors'][i] for i in range(eval_sources)]

            lsd_values = [score['lsd'] for score in lsd_scores]
            nmse_values = [score['nmse'] for score in lsd_scores]
            lsd_m_fund_values = [score['lsd_m_fund'] for score in lsd_scores]

            results[M][w].update({
                'lsd_mean': np.mean(lsd_values),
                'lsd_std': np.std(lsd_values),
                'nmse_mean': np.mean(nmse_values),
                'nmse_std': np.std(nmse_values),
                'lsd_mean_m_fund': np.mean(lsd_m_fund_values),
                'lsd_std_m_fund': np.std(lsd_m_fund_values),
                'num_sources_eval': eval_sources,
                'model_freq_up_to': model_freq_up_to,
                'eval_freq_up_to': eval_freq_up_to
            })

            if enable_inference_timing and len(timing_ms_per_source) > 0:
                warmup = max(0, min(int(timing_warmup_sources), len(timing_ms_per_source)))
                timed_samples = timing_ms_per_source[warmup:]
                if len(timed_samples) == 0:
                    timed_samples = timing_ms_per_source
                    warmup = 0
                wall_elapsed_s = (time.perf_counter() - timed_loop_wall_start) if timed_loop_wall_start is not None else None
                timing_stats = {
                    'enabled': True,
                    'cache_hit': False,
                    'warmup_sources_ignored': warmup,
                    'num_timed_sources': len(timed_samples),
                    'mean_ms': float(np.mean(timed_samples)),
                    'std_ms': float(np.std(timed_samples)),
                    'median_ms': float(np.median(timed_samples)),
                    'p95_ms': float(np.percentile(timed_samples, 95)),
                    'sum_ms': float(np.sum(timed_samples)),
                    'wall_total_s': float(wall_elapsed_s) if wall_elapsed_s is not None else None,
                }
                results[M][w]['inference_timing'] = timing_stats
                print(
                    f"  Timing (M={M}, w={w}): mean={timing_stats['mean_ms']:.2f} ms/src, "
                    f"p95={timing_stats['p95_ms']:.2f} ms/src, "
                    f"sum={timing_stats['sum_ms'] / 1000.0:.2f} s for {timing_stats['num_timed_sources']} sources "
                    f"(warmup={timing_stats['warmup_sources_ignored']})"
                )
            elif cache_meta is not None and 'per_source_errors' in results[M][w]:
                # Cache-hit path has no fresh forward passes to time.
                results[M][w]['inference_timing'] = {
                    'enabled': bool(enable_inference_timing),
                    'cache_hit': True,
                    'note': 'Timing unavailable on cache hit. Delete cache to benchmark fresh inference.'
                }
    
    return results, idx_mes_pos_mat, grid_xyz


def plot_atf_comparisons(atf_mag_est_ref, atf_mag_est_yours, atf_mag_gt, ref_config, freq_up_to, num_sources_eval, best_guidance=None, output_dir=None, atf_mag_est_eeae=None, grid_xyz=None, per_source_lsd=None, src_offset=0):
    """
    Plot ATF comparisons with 3 methods: True, Reference, Your Model for multiple combinations
    
    Args:
        atf_mag_est_ref: Reference model predictions
        atf_mag_est_yours: Dictionary of your model predictions for each guidance scale
        atf_mag_gt: Ground truth ATF values
        ref_config: Reference model config
        freq_up_to: Number of frequency bins to use
        num_sources_eval: Number of sources to evaluate
        best_guidance: Optional, pre-computed best guidance scale. If None, will compute it.
    """
    # Get the correct number of sources to evaluate
    total_sources = atf_mag_gt.shape[2]  # Total available sources
    eval_sources = min(num_sources_eval, total_sources) if num_sources_eval is not None else total_sources
    print(f"Evaluating ATF plots for first {eval_sources} sources (out of {total_sources})")

    # Use provided best_guidance or compute it
    if best_guidance is None:
        # Select the best guidance scale for visualization (lowest average LSD)
        guidance_scales = list(atf_mag_est_yours.keys())
        best_guidance = guidance_scales[0]  # Default to first scale
        best_lsd = float('inf')
        
        for w in guidance_scales:
            # Make sure to use the same number of sources for comparison
            current_lsd = torch.mean((atf_mag_est_yours[w][:, :, :eval_sources] - 
                                    atf_mag_gt[:, :freq_up_to, :eval_sources]) ** 2).item()
            if current_lsd < best_lsd:
                best_lsd = current_lsd
                best_guidance = w
        print(f"Computed best guidance scale w={best_guidance} (LSD={best_lsd:.4f})")
    else:
        print(f"Using provided best guidance scale w={best_guidance}")
    
    atf_mag_est_yours_best = atf_mag_est_yours[best_guidance]

    # Create frequency axes for both models
    ref_freq_bins = ref_config['num_freq']  # 64 bins

    fs = ref_config['fs']  # 2000 Hz
    
    # Reference frequency axis (0 to 1000 Hz, 64 bins)
    freq_ref = np.arange(1, ref_freq_bins + 1) / ref_freq_bins * fs / 2
    
    # Your model frequency axis (0 to ~312 Hz, 20 bins)  
    freq_yours = np.arange(1, freq_up_to + 1) / freq_up_to * fs / 2
    
    print(f"Reference freq range: 0-{freq_ref[-1]:.0f} Hz ({ref_freq_bins} bins)")
    print(f"Your model freq range: 0-{freq_yours[-1]:.0f} Hz ({freq_up_to} bins)")
    fftlen_algn = 128
    freq_axis = np.arange(1, fftlen_algn // 2 + 1) / fftlen_algn * fs
    freq_axis = freq_axis[:freq_up_to]  # Ensure it matches model's frequency count

    plt.rcParams["font.size"] = 18
    
    # Create output directory (same structure as inference_1d_atf.py)
    # output_dir = "artifacts/eval/atf_comparisons"
    # os.makedirs(output_dir, exist_ok=True)

    if atf_mag_est_yours is not None:
        # Multiple source and microphone combinations (similar to inference_1d_atf.py)
        total_sources_for_plots = min(num_sources_eval, atf_mag_gt.shape[2]) if num_sources_eval is not None else atf_mag_gt.shape[2]
        # --- Source selection: representative spread by LSD percentile ---
        # If per_source_lsd is provided, pick 10 sources spanning best/median/worst
        # thirds of the LSD distribution, so PDFs show a fair cross-section.
        # Fall back to first-10 if not available.
        if per_source_lsd is not None and len(per_source_lsd) >= 10:
            sorted_indices = sorted(range(len(per_source_lsd)), key=lambda i: per_source_lsd[i])
            n = len(sorted_indices)
            # 3 best, 4 median, 3 worst
            picks = (
                sorted_indices[:3] +
                sorted_indices[n//2 - 2 : n//2 + 2] +
                sorted_indices[-3:]
            )
            source_indices = sorted(picks)  # keep sorted for consistent filenames
            print(f"Source selection: 3 best / 4 median / 3 worst by LSD "
                  f"(LSD range {per_source_lsd[sorted_indices[0]]:.2f}–"
                  f"{per_source_lsd[sorted_indices[-1]]:.2f} dB)")
        else:
            source_indices = list(range(min(10, total_sources_for_plots)))
            print("Source selection: first 10 sources (no per_source_lsd provided)")

        # --- Mic selection: 5 spatially spread positions across the 11^3 cube ---
        # Indices [0, 272, 665, 937, 1330] are extreme corners + center.
        # Instead use positions covering different depths and lateral offsets.
        # These map to roughly: front-bottom-left, back-bottom-right,
        # center, front-top-right, back-top-left — evenly spread.
        mic_indices = [0, 272, 665, 937, 1330]  # keep original for PDF reproducibility
        
        plot_count = 0
        total_plots = len(source_indices)  # One PDF per source (each with 5 subplots)
        
        print(f"Generating {total_plots} ATF comparison PDFs (5 microphones per PDF)...")
        
        # Get microphone coordinates for titles
        # grid_xyz is passed in — it's a fixed 11x11x11 meshgrid, identical regardless of
        # source count, so we never need to reconstruct an ATF3DSampler just for this.
        if grid_xyz is None:
            raise ValueError("grid_xyz must be passed to plot_atf_comparisons to avoid cache invalidation")
        # VM runs often keep grid_xyz on CUDA; convert once for NumPy-based plot titles.
        grid_xyz_cpu = grid_xyz.detach().cpu() if isinstance(grid_xyz, torch.Tensor) else grid_xyz
        
        for src_idx in source_indices:
            fig, axes = plt.subplots(5, 1, figsize=(10, 4*5))
            plt.subplots_adjust(hspace=0.5)

            for i, mic_idx in enumerate(mic_indices):
                ax = axes[i]

                ax.plot(freq_axis, atf_mag_gt[mic_idx, :freq_up_to, src_idx], 'k--', label="True", linewidth=1.0)
                if atf_mag_est_ref is not None:
                    ax.plot(freq_axis, atf_mag_est_ref[mic_idx, :freq_up_to, src_idx], 'r-', label="FSMPAE", linewidth=0.9)
                if atf_mag_est_eeae is not None:
                    eeae_bins = atf_mag_est_eeae.shape[1]
                    ax.plot(freq_axis[:eeae_bins], atf_mag_est_eeae[mic_idx, :, src_idx], 'g-', label="EEAE", linewidth=0.9)
                ax.plot(freq_axis, atf_mag_est_yours_best[mic_idx, :, src_idx], 'b-', label="SF-Flow", linewidth=0.9)

                ax.set_xscale('log')
                ax.grid(True)
                ax.tick_params(labelsize=8)
                ax.set_xlabel("Frequency (Hz)", fontsize=8)
                ax.set_ylabel("Magnitude (dB)", fontsize=8)

                mic_coord = grid_xyz_cpu[mic_idx].numpy()
                ax.set_title(f"ATF ({mic_coord[0]:.2f} m, {mic_coord[1]:.2f} m, {mic_coord[2]:.2f} m)", fontsize=9)
                ax.legend(fontsize=7, loc='upper right', ncol=2)
                print(f"Plotting Source {src_idx+src_offset}, Mic {mic_idx} (index {i+1}/5)")

            plt.tight_layout()

            filename = f"ATF_Comparison_src{src_idx+src_offset:04d}_test.pdf"
            filepath = os.path.join(output_dir, filename)
            fig.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            plot_count += 1
            print(f"Saved {plot_count}/{len(source_indices)} plots: {filename}")
        
        print(f"All {len(source_indices)} ATF comparison PDFs saved to {output_dir}/")
    else:
        print("Your model predictions not available - skipping ATF plots")


def get_your_model_atf_predictions(set_encoder, ode_3d, config, device, atf_mag_gt, ref_config,
                                   model_freq_up_to, eval_freq_up_to, num_sources_eval,
                                   guidance_scales=None, single_guidance=None,
                                   random_M_sampling=False, model_name=None):
    """
    Extract ATF predictions from your model in the same format as reference model.
    Based on inference_1d_atf.py approach.
    
    Args:
        guidance_scales: List of guidance scales to evaluate. If single_guidance is provided, this is ignored.
        single_guidance: Optional, single guidance scale to evaluate. If provided, only this scale is used.
    """
    if single_guidance is not None:
        guidance_scales = [single_guidance]
    print("Generating ATF predictions from your 3D model...")

    # Load your data (same as in inference_1d_atf.py)
    data_path = _cli_args.data_dir

    # Detect geo_conditioning from checkpoint config
    _geo_atf = config.get('training', {}).get('geo_conditioning', False)
    _room_dims_atf = None
    if _geo_atf:
        import re as _re_atf
        _cfg_dir_atf = config.get('data', {}).get('data_dir', data_path)
        _rm_atf = _re_atf.search(r'room(\d+\.?\d*)x(\d+\.?\d*)x(\d+\.?\d*)', _cfg_dir_atf)
        if _rm_atf:
            _room_dims_atf = (float(_rm_atf.group(1)), float(_rm_atf.group(2)), float(_rm_atf.group(3)))
        else:
            _geo_atf = False
    src_split = config['data']['src_splits']
    freq_from  = config['model'].get('freq_from', 0)

    # Load normalized data
    train_sampler = ATF3DSampler(
        data_path=data_path, mode='train', src_splits=src_split,
        normalize=True, freq_up_to=model_freq_up_to, freq_from=freq_from, model_name=model_name
    )
    test_sampler = ATF3DSampler(
        data_path=data_path, mode='test', src_splits=src_split,
        normalize=False, freq_up_to=model_freq_up_to, freq_from=freq_from, model_name=model_name
    )
    test_sampler.cubes = (test_sampler.cubes - train_sampler.mean) / (train_sampler.std + 1e-8)
    
    grid_xyz = train_sampler.grid_xyz.to(device)
    mean = train_sampler.mean.item()
    std = train_sampler.std.item()
    
    # Create simulator
    simulator = EulerSimulator(ode=ode_3d)
    
    # Initialize output array matching reference format [Guidance, Mic, Freq, Source]
    total_mics = atf_mag_gt.shape[0] if atf_mag_gt is not None else 1331
    total_sources = min(num_sources_eval, len(test_sampler)) if num_sources_eval is not None else len(test_sampler)
    your_atf_predictions = {w: torch.zeros(total_mics, eval_freq_up_to, total_sources) for w in guidance_scales}
    
    # Fixed M and parameters (from inference_1d_atf.py)
    M = ref_config['num_mes_test']  # Use same M as reference (5)
    num_timesteps = 10
    
    # Load the SAME microphone selection strategy as reference model
    idx_mes_pos_path = "idx_mes_pos_s1024_m1331.npy"
    idx_mes_pos_mat = np.load(idx_mes_pos_path)
    print(f"Loaded reference microphone selection matrix: {idx_mes_pos_mat.shape}")
    print("Using source-specific microphone selection for ATF generation")
    print(f"Generating predictions for {total_sources} sources with M={M} microphones...")
    
    # Generate predictions for each source
    for src_idx in tqdm(range(total_sources), desc="Your Model ATF"):
        with torch.no_grad():
            # Get source data (same as inference_1d_atf.py)
            z_true = test_sampler.cubes[src_idx].unsqueeze(0).to(device)
            src_xyz = test_sampler.source_coords[src_idx].unsqueeze(0).to(device)
            
            # Create sparse observations - use SAME strategy as reference for fair comparison
            if random_M_sampling:
                print("Using random microphone selection for ATF generation")
                obs_indices = torch.randperm(grid_xyz.shape[0])[:M]  # Fallback to random

            else:
                # Use source-specific microphones (different subsets for each source)
                source_specific_indices = idx_mes_pos_mat[:M, src_idx]  # First M mics for this source
                obs_indices = torch.tensor(source_specific_indices, dtype=torch.long, device=device)
            
            obs_xyz_abs = grid_xyz[obs_indices]
            obs_coords_rel = (obs_xyz_abs - src_xyz).unsqueeze(0)  # [1, M, 3]

            # Geo conditioning
            if _geo_atf and _room_dims_atf is not None:
                _Lx, _Ly, _Lz = _room_dims_atf
                _half_min = min(_Lx, _Ly, _Lz) / 2.0
                _d_walls = torch.stack([
                    src_xyz[:, 0],      _Lx - src_xyz[:, 0],
                    src_xyz[:, 1],      _Ly - src_xyz[:, 1],
                    src_xyz[:, 2],      _Lz - src_xyz[:, 2],
                ], dim=1) / _half_min  # [1, 6]
                obs_coords_rel = torch.cat([
                    obs_coords_rel,
                    _d_walls.unsqueeze(1).expand(-1, M, -1)
                ], dim=-1)  # [1, M, 9]
            
            z_flat = z_true.view(z_true.shape[1], -1)
            obs_values = z_flat[:, obs_indices].transpose(0, 1).unsqueeze(0)
            obs_mask = torch.ones(1, M, dtype=torch.bool, device=device)
            
            # Get conditioning tokens
            y_tokens, pooled_context, freq_contexts = set_encoder(obs_coords_rel, obs_values, obs_mask)

            # Generate prediction (same as inference_1d_atf.py)
            x0 = torch.randn_like(z_true)
            ts = torch.linspace(0, 1, num_timesteps + 1, device=device)
            ts = ts.view(1, -1, 1, 1, 1, 1).expand(x0.shape[0], -1, -1, -1, -1, -1)

            # Run inference for each guidance scale
            for w in guidance_scales:
                simulator.ode.guidance_scale = w
                x1_recon = simulator.simulate(x0, ts, x0=x0, z_true=z_true, y_tokens=y_tokens,
                                           obs_mask=obs_mask, pooled_context=pooled_context,
                                           freq_contexts=freq_contexts,
                                           paste_observations=True, obs_indices=obs_indices)
                
                # De-normalize (same as inference_1d_atf.py)
                gen_cube_denorm = (x1_recon * std + mean)
                
                # Convert 3D grid to microphone format
                # Extract ATF values at all microphone positions
                nx, ny, nz = 11, 11, 11  # Grid dimensions
                for mic_idx in range(total_mics):
                    # Convert flat microphone index to 3D coordinates (same as inference_1d_atf.py)
                    iz, iy, ix = np.unravel_index(mic_idx, (nz, ny, nx))
                    
                    # Extract frequency response at this microphone position
                    if iz < gen_cube_denorm.shape[2] and iy < gen_cube_denorm.shape[3] and ix < gen_cube_denorm.shape[4]:
                        your_atf_predictions[w][mic_idx, :, src_idx] = gen_cube_denorm[0, :eval_freq_up_to, iz, iy, ix].cpu()
    
    # print(f"Generated ATF predictions: {your_atf_predictions.shape} (Mic, Freq, Source)")
    return your_atf_predictions

# def get_fallback_reference_results():
#     """Get pre-computed reference results as fallback."""
#     reference_results = {
#         100: {'mean': 3.7072, 'std': 0.8607},
#         50: {'mean': 3.9413, 'std': 0.8662},
#         20: {'mean': 4.1927, 'std': 0.8633},
#         10: {'mean': 4.3775, 'std': 0.8779},
#         5: {'mean': 4.4037, 'std': 0.8952}
#     }
#     return reference_results


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
guidance_scales = [1]
M_values = [5]
num_sources_eval = 102  # Set to None to evaluate all 102 sources, or e.g. 30 for faster testing

random_M_sampling = False

# Set to True to generate distribution and ATF comparison PDFs after the summary table.
# Printing the table is always executed regardless of this flag.
GENERATE_PLOTS = True

# Set to True  → coord normalisation applied (correct, matches training pipeline for new runs).
# Set to False → no coord normalisation (legacy behaviour; needed to reproduce old Tokyo best 2.86 dB).
NORMALIZE_COORDS = True

# Fast debug switch: evaluate only reference methods (FSMPAE / optional EEAE), skip all SFlow inference.
REFERENCE_ONLY = False

# If set (e.g. 20), evaluate only first N bins while still running each model at its own trained freq.
# If None, defaults to the smallest model_freq_up_to across loaded SFlow + AE models.
EVAL_FREQ_UP_TO = None

# Add/remove EEAE runs to compare. Empty list disables EEAE evaluation.
# Example: EEAE_COMPARISONS = [10001, 10002, 10003, 10004]
EEAE_COMPARISONS = []

# CLI overrides (applied after constant defaults above)
if _cli_args.M is not None:
    if any(m <= 0 for m in _cli_args.M):
        raise ValueError(f"--M values must be positive integers, got: {_cli_args.M}")
    M_values = _cli_args.M
RUN_FSMPAE = False  # external AE baseline is not included in the public release
TIMING_WARMUP_SOURCES = max(0, int(_cli_args.timing_warmup_sources))
ENABLE_INFERENCE_TIMING = not _cli_args.disable_timing

def get_dataset_version_from_data_dir(data_dir: str) -> str:
    """Parse dataset version from the data_dir path stored in config.
    Examples:
      'ir_fs2000_s1024_m1331_...' -> 'r1'
      'ir_fs2000_s8192_m1331_...' -> 'r4'
    Falls back to the model-name heuristic if pattern not found.
    """
    import re
    m = re.search(r's(\d+)', data_dir)
    if m:
        num_sources = int(m.group(1))
        mapping = {1024: 'r1', 2048: 'r2', 4096: 'r3', 8192: 'r4'}
        return mapping.get(num_sources, 'r1')
    return 'r1'  # safe fallback


def load_coord_stats(dataset_version='r1'):
    """Load cached coord normalisation statistics written by trainer-atf-3d.py."""
    cache_path = f"coord_stats_{dataset_version}.pt"
    if not os.path.exists(cache_path):
        raise FileNotFoundError(
            f"Coord stats cache not found: {cache_path}. "
            "Run trainer-atf-3d.py once to generate it."
        )
    stats = torch.load(cache_path)
    return stats['mean'], stats['std']  # both are [3] tensors

def get_model_name(model_path):
    """Extract model name from path, including filename if multiple models in same directory.
    Works for both local artifacts/ paths and external/SSD paths."""
    # Use the parent directory name of the .pt file — works regardless of root path
    dir_name = os.path.basename(os.path.dirname(model_path))

    # Get filename without extension
    filename = os.path.basename(model_path).replace('.pt', '')

    # If filename is just "model", return directory name only (backward compatibility)
    if filename == "model":
        return dir_name
    else:
        # Include both directory and filename for unique identification
        return f"{dir_name}_{filename}"


def _canonicalize_for_cache(obj):
    """Recursively convert values into JSON-stable, hashable forms."""
    if isinstance(obj, dict):
        return {str(k): _canonicalize_for_cache(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
    if isinstance(obj, (list, tuple)):
        return [_canonicalize_for_cache(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, torch.Tensor):
        if obj.numel() <= 64:
            return obj.detach().cpu().tolist()
        return {"__tensor_shape__": list(obj.shape), "__dtype__": str(obj.dtype)}
    return obj


def _build_atf_cache_meta(model_path, guidance, M, num_sources_eval, model_freq_up_to, eval_freq_up_to,
                          random_M_sampling, config, data_dir=None):
    """
    Build a cache metadata dictionary that uniquely identifies an ATF inference run.
    Different M / guidance / eval bins / model checkpoint config / data_dir -> different cache key.
    data_dir distinguishes same-model runs on different datasets.
    """
    model_path_abs = os.path.abspath(model_path)
    model_mtime = os.path.getmtime(model_path_abs) if os.path.exists(model_path_abs) else None
    return {
        "cache_version": 2,
        "model_path": model_path_abs,
        "model_mtime": model_mtime,
        "guidance": float(guidance),
        "M": int(M),
        "num_sources_eval": int(num_sources_eval),
        "model_freq_up_to": int(model_freq_up_to),
        "eval_freq_up_to": int(eval_freq_up_to),
        "random_M_sampling": bool(random_M_sampling),
        "num_timesteps": 10,  # fixed in get_your_model_atf_predictions
        "src_splits": config.get("data", {}).get("src_splits"),
        "freq_from": config.get("model", {}).get("freq_from", 0),
        "geo_conditioning": bool(config.get("training", {}).get("geo_conditioning", False)),
        "data_dir": os.path.abspath(data_dir) if data_dir else None,
    }


def _atf_cache_filename(cache_meta):
    canonical = _canonicalize_for_cache(cache_meta)
    cache_hash = hashlib.sha1(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:12]
    return f"atf_pred_cache_{cache_hash}.pt"


def _portable_cache_meta(meta):
    """Reduce cache metadata to machine-independent form so caches generated on
    another machine (e.g. Tokyo server) still match after copying: absolute
    model_path becomes its basename, model_mtime and data_dir are dropped
    (data_dir is checked separately with the same leniency as the legacy path)."""
    portable = {k: v for k, v in meta.items() if k not in ("model_mtime", "data_dir")}
    if portable.get("model_path"):
        portable["model_path"] = os.path.basename(portable["model_path"])
    return portable


def _load_matching_atf_cache(model_dir, cache_meta):
    """Load matching cache if present in model directory."""
    model_dir = os.path.abspath(model_dir)
    exact_path = os.path.join(model_dir, _atf_cache_filename(cache_meta))
    candidates = [exact_path]
    if not os.path.exists(exact_path):
        candidates.extend(sorted(glob.glob(os.path.join(model_dir, "atf_pred_cache*.pt"))))

    for cache_path in candidates:
        if not os.path.exists(cache_path):
            continue
        try:
            payload = torch.load(cache_path, map_location='cpu')
        except Exception as e:
            print(f"  WARNING: Failed reading cache {cache_path}: {e}")
            continue

        # New-format cache with machine-independent metadata match
        if isinstance(payload, dict) and isinstance(payload.get("metadata"), dict) and "atf_predictions" in payload:
            cached_data_dir = payload["metadata"].get("data_dir")
            meta_data_dir = cache_meta.get("data_dir")
            data_dir_ok = (cached_data_dir is None or meta_data_dir is None or
                           cached_data_dir == meta_data_dir)
            if data_dir_ok and _portable_cache_meta(payload["metadata"]) == _portable_cache_meta(cache_meta):
                return cache_path, payload["atf_predictions"]

        # Backward compatibility: older cache payloads (best-effort match).
        # data_dir is checked only when present in both sides to avoid cross-room hits.
        if isinstance(payload, dict) and "atf_predictions" in payload:
            cached_data_dir = payload.get("metadata", {}).get("data_dir") or payload.get("data_dir")
            meta_data_dir = cache_meta.get("data_dir")
            data_dir_ok = (cached_data_dir is None or meta_data_dir is None or
                           cached_data_dir == meta_data_dir)
            old_ok = (
                data_dir_ok and
                payload.get("guidance") == cache_meta["guidance"] and
                payload.get("M") == cache_meta["M"] and
                payload.get("num_sources_eval") == cache_meta["num_sources_eval"] and
                payload.get("model_freq_up_to") == cache_meta["model_freq_up_to"] and
                payload.get("eval_freq_up_to") == cache_meta["eval_freq_up_to"]
            )
            if old_ok:
                return cache_path, payload["atf_predictions"]

    return None, None


def _save_atf_cache(cache_path, cache_meta, atf_predictions):
    payload = {
        "metadata": cache_meta,
        "atf_predictions": {w: t.detach().cpu() for w, t in atf_predictions.items()},
    }
    torch.save(payload, cache_path)

# Get model names
MODEL_NAMES = [get_model_name(path) for path in MULTI_MODEL_PATHS]
MULTI_MODEL_MODE = len(MULTI_MODEL_PATHS) > 1

mode_title = '=== REFERENCE-ONLY EVALUATION ===' if REFERENCE_ONLY else (
    '=== MULTI-MODEL EVALUATION ===' if MULTI_MODEL_MODE else '=== SINGLE MODEL EVALUATION ==='
)
print(mode_title)
print(f"Device: {device}")
print(f"M values: {M_values}")
print(f"Inference timing: {'enabled' if ENABLE_INFERENCE_TIMING else 'disabled'} "
      f"(warmup_sources={TIMING_WARMUP_SOURCES})")
if not REFERENCE_ONLY:
    for i, (path, name) in enumerate(zip(MULTI_MODEL_PATHS, MODEL_NAMES)):
        print(f"  Model {i+1}: {name}")
print()

# Load all your models first (we choose eval_freq_up_to after seeing all methods)
if REFERENCE_ONLY:
    print("\n1. Skipping SFlow model loading (REFERENCE_ONLY=True)")
else:
    print("\n1. Loading your 3D Flow Matching models...")
all_your_results = {}
all_your_predictions = {}  # Store predictions to avoid reloading best model
all_model_info = {}  # Store model information
all_model_configs = {}
all_model_coord_stats = {}
sflow_model_freq_up_to = {}
all_model_paths = {}

if not REFERENCE_ONLY:
    for i, (model_path, model_name) in enumerate(zip(MULTI_MODEL_PATHS, MODEL_NAMES)):
        print(f"Loading model {i+1}/{len(MULTI_MODEL_PATHS)}: {model_name}")

        checkpoint, config, model_states_cfg = load_model_and_config(model_path, device)

        # Print model metadata
        _ckpt_iter = checkpoint.get('iteration', 'Unknown')
        _ckpt_fm = checkpoint.get('best_val_loss', 'Unknown')
        _ckpt_lsd = checkpoint.get('best_val_lsd', 'Unknown')
        print(f"  [Checkpoint Metadata] Iteration: {_ckpt_iter}")
        if _ckpt_fm != 'Unknown': print(f"  [Checkpoint Metadata] Best Val FM-MSE: {_ckpt_fm:.5f}")
        if _ckpt_lsd != 'Unknown': print(f"  [Checkpoint Metadata] Best Val LSD: {_ckpt_lsd}")

        # Create and load models
        set_encoder, unet_3d, ode_3d, is_new_model = model_factory(config, model_states_cfg, device)

        model_freq = config['model'].get('freq_up_to')
        sflow_model_freq_up_to[model_name] = model_freq
        print(f"  {model_name} model_freq_up_to: {model_freq}")
        
        # Get model information
        set_encoder_info = get_model_info(set_encoder, "SetEncoder")
        unet_info = get_model_info(unet_3d, "UNet3D")
        
        # Calculate total model size (SetEncoder + UNet)
        total_params = set_encoder_info['total_params'] + unet_info['total_params']
        total_size_mb = set_encoder_info['model_size_mb'] + unet_info['model_size_mb']
        
        model_info = {
            'set_encoder': set_encoder_info,
            'unet': unet_info,
            'total_params': total_params,
            'total_params_str': f"{total_params:,}",
            'total_size_mb': total_size_mb,
            'total_size_str': f"{total_size_mb:.2f} MB"
        }
        all_model_info[model_name] = model_info
        
        # Print model info
        print(f"\n--- {model_name} Architecture ---")
        print_model_info(set_encoder, "SetEncoder")
        print_model_info(unet_3d, "UNet3D")
        print(f"=== Combined Model ===")
        print(f"Total parameters: {model_info['total_params_str']}")
        print(f"Total size: {model_info['total_size_str']}")
        print("=" * 20)
        
        # Load coord stats from the data_dir stored in this checkpoint's config
        _data_dir = config.get('data', {}).get('data_dir', '')
        dataset_version = get_dataset_version_from_data_dir(_data_dir)
        print(f"  Dataset version inferred from data_dir '{_data_dir}': {dataset_version}")
        try:
            _coord_mean, _coord_std = load_coord_stats(dataset_version)
            print(f"  Coord stats loaded: mean={_coord_mean}, std={_coord_std}")
        except FileNotFoundError as e:
            print(f"  WARNING: {e}. Running without coord normalisation.")
            _coord_mean, _coord_std = None, None

        # Store model components for later plotting (avoid reloading best model)
        all_your_predictions[model_name] = (set_encoder, unet_3d, ode_3d, config)
        all_model_configs[model_name] = config
        all_model_coord_stats[model_name] = (_coord_mean, _coord_std)
        all_model_paths[model_name] = model_path

# Load reference methods and decide eval_freq_up_to
print("\n2. SF-Flow-only evaluation (no external baseline models).")
atf_mag_est = atf_mag_gt = ref_config = ref_data = fsmpae_model_freq_up_to = ref_inv_perm_pt_to_ae = None
eeae_atf_est_by_id = {}
eeae_model_freq_up_to_by_id = {}

all_freq_candidates = list(sflow_model_freq_up_to.values())
if RUN_FSMPAE and fsmpae_model_freq_up_to is not None:
    all_freq_candidates.append(fsmpae_model_freq_up_to)
all_freq_candidates.extend(list(eeae_model_freq_up_to_by_id.values()))
if EVAL_FREQ_UP_TO is None:
    eval_freq_up_to = min(all_freq_candidates)
    print(f"\n[auto eval_freq_up_to] Using smallest available model_freq_up_to: {eval_freq_up_to}")
else:
    eval_freq_up_to = EVAL_FREQ_UP_TO
    min_available = min(all_freq_candidates)
    if eval_freq_up_to > min_available:
        raise ValueError(
            f"EVAL_FREQ_UP_TO={eval_freq_up_to} exceeds smallest available model_freq_up_to={min_available}. "
            "Set EVAL_FREQ_UP_TO <= smallest available value."
        )
    print(f"\n[user eval_freq_up_to] Using EVAL_FREQ_UP_TO={eval_freq_up_to}")

if REFERENCE_ONLY:
    print("\n3. Skipping SFlow evaluation (REFERENCE_ONLY)")
else:
    print("\n3. Evaluating SFlow models...")
    for i, (model_path, model_name) in enumerate(zip(MULTI_MODEL_PATHS, MODEL_NAMES)):
        print(f"Evaluating model {i+1}/{len(MULTI_MODEL_PATHS)}: {model_name}")
        set_encoder, unet_3d, ode_3d, config = all_your_predictions[model_name]
        _coord_mean, _coord_std = all_model_coord_stats[model_name]
        model_results, idx_mes_pos_mat, _grid_xyz = evaluate_your_model(
            set_encoder, ode_3d, config, M_values, device,
            num_sources_eval, guidance_scales,
            random_M_sampling=random_M_sampling,
            model_name=model_name,
            normalize_coords=NORMALIZE_COORDS,
            coord_mean=_coord_mean if NORMALIZE_COORDS else None,
            coord_std=_coord_std if NORMALIZE_COORDS else None,
            eval_freq_up_to=eval_freq_up_to,
            model_path=model_path,
            data_dir=_cli_args.data_dir,
            timing_warmup_sources=TIMING_WARMUP_SOURCES,
            enable_inference_timing=ENABLE_INFERENCE_TIMING,
        )
        all_your_results[model_name] = model_results
        all_model_grid_xyz = _grid_xyz  # fixed 11^3 room grid

ref_results = None
eeae_results_by_id = {}

COL_W = 45  # Method column width

# Print results
print("\n" + "="*80)
print("=== COMPARISON RESULTS ===")
print("="*80)
if ref_config is not None:
    print(f"Evaluation freq range: 0-{eval_freq_up_to*ref_config['fs']//2//ref_config['num_freq']:.0f} Hz ({eval_freq_up_to} bins)")
    print(f"Reference freq range: 0-{ref_config['fs']//2} Hz ({ref_config['num_freq']} bins)")
else:
    print(f"Evaluation: {eval_freq_up_to} bins")
if ref_results is not None:
    print(f"Sources evaluated: {ref_results['num_sources_eval']} (out of 102 total)")
print()
print("M_fundamental = 5 specific evaluation positions [0, 272, 665, 937, 1330] for PDFs")
print("Full cube = All 1331 spatial positions")
if ref_config is not None:
    print(f"FAIR COMPARISON: All methods evaluated on first {eval_freq_up_to} bins (0-{eval_freq_up_to*ref_config['fs']//2//ref_config['num_freq']:.0f} Hz)")
print("-"*140)
print(f"{'Method':<45} | {'w':<4} | {'LSD M_fund (mean±std)':<23} | {'LSD Full (mean±std)':<21} | {'NMSE Full (dB)':<14} | {'Freq Range':<15}")
print("-"*140)

for model_name, model_results in all_your_results.items():
    for M in M_values:
        label = f"{model_name} (M={M})"
        display = label[-COL_W:] if len(label) > COL_W else label

        # Print results for each guidance scale
        for w in guidance_scales:
            your_lsd_m_fund = model_results[M][w]['lsd_mean_m_fund']
            your_lsd_std_m_fund = model_results[M][w]['lsd_std_m_fund']
            your_lsd_full = model_results[M][w]['lsd_mean']
            your_lsd_std_full = model_results[M][w]['lsd_std']
            your_nmse_full = model_results[M][w]['nmse_mean']

            print(f"{display:<{COL_W}} | {w:<4.1f} | {f'{your_lsd_m_fund:.4f}±{your_lsd_std_m_fund:.4f}':<23} | {f'{your_lsd_full:.4f}±{your_lsd_std_full:.4f}':<21} | {your_nmse_full:.4f}       | {f'First {eval_freq_up_to} bins':<15}")
            print("-"*140)

# Find best model and guidance scale combination
if all_your_results:
    best_model = None
    best_guidance = None
    best_lsd = float('inf')
    best_lsd_std = None
    best_results = {}  # Store best results for reuse

    for model_name, model_results in all_your_results.items():
        for M in M_values:
            for w in guidance_scales:
                if model_results[M][w]['lsd_mean'] < best_lsd:
                    best_lsd = model_results[M][w]['lsd_mean']
                    best_lsd_std = model_results[M][w]['lsd_std']
                    best_model = model_name
                    best_guidance = w
                    best_results = {
                        'model': best_model,
                        'guidance': best_guidance,
                        'lsd': best_lsd,
                        'lsd_std': best_lsd_std
                    }

    print("="*80)
    print(f"🏆 BEST MODEL: {best_model}")
    print(f"   Best Guidance Scale: {best_guidance}")
    if best_lsd_std is not None:
        print(f"   Best LSD: {best_lsd:.4f} ± {best_lsd_std:.4f} dB")
    else:
        print(f"   Best LSD: {best_lsd:.4f} dB")
    if best_model in all_model_info:
        best_model_info = all_model_info[best_model]
        print(f"   Model Parameters: {best_model_info['total_params_str']}")
        print(f"   Model Size: {best_model_info['total_size_str']}")
        print(f"   SetEncoder: {best_model_info['set_encoder']['total_params_str']} params")
        print(f"   UNet3D: {best_model_info['unet']['total_params_str']} params")
    # print(f"   Improvement over Reference: {ref_results['mean'] - best_lsd:+.4f} dB")
    print("="*80)
    if RUN_FSMPAE and ref_results is not None:
        print(f"Note: Ref models use M={ref_results['num_mics']} observation microphones")
        print(f"      Reference uses source-specific microphone selection")
        print(f"      Your models use SAME source-specific microphone selection")
        print("      (Different source-specific microphone subsets per source)")
    print("="*80)

# If FSMPAE is disabled but SFlow models ran, load GT from SFlow sampler so ATF plots can be generated.
if not RUN_FSMPAE and atf_mag_gt is None and all_your_results and best_model and best_model in all_model_configs:
    _best_config_for_gt = all_model_configs[best_model]
    print("\n5a. Loading GT from SFlow sampler (no FSMPAE, GT needed for ATF plots)...")
    atf_mag_gt = load_gt_from_sflow_sampler(
        _best_config_for_gt, num_sources_eval=num_sources_eval, data_dir=_cli_args.data_dir
    )
    ref_config = {
        'num_freq': _best_config_for_gt['model'].get('freq_up_to'),
        'fs': 2000,
        'num_mes_test': 5,
    }
    print(f"  Minimal ref_config for plots: {ref_config}")

# Build/load cache for best SFlow model ATF predictions regardless of plotting.
best_model_atf_predictions = None
if all_your_results and best_model and best_model in all_your_predictions:
    print("\n5. Preparing cached ATF predictions for best SFlow model...")
    set_encoder_best, unet_3d_best, ode_3d_best, config_best = all_your_predictions[best_model]
    _best_w    = best_results['guidance']
    _atf_M     = ref_config['num_mes_test'] if ref_config is not None else 5
    _best_model_path = all_model_paths[best_model]
    _best_model_dir = os.path.dirname(_best_model_path)
    # Clamp to the actual number of test sources so the cache key matches the one
    # written by evaluate_your_model (which also clamps); otherwise a smaller test
    # set (e.g. the smoke split) causes a spurious cache miss and duplicate inference.
    _n_src     = num_sources_eval if num_sources_eval is not None else (atf_mag_gt.shape[2] if atf_mag_gt is not None else 102)
    if atf_mag_gt is not None:
        _n_src = min(_n_src, atf_mag_gt.shape[2])
    _best_model_freq_up_to = sflow_model_freq_up_to[best_model]
    _cache_meta = _build_atf_cache_meta(
        model_path=_best_model_path,
        guidance=_best_w,
        M=_atf_M,
        num_sources_eval=_n_src,
        model_freq_up_to=_best_model_freq_up_to,
        eval_freq_up_to=eval_freq_up_to,
        random_M_sampling=random_M_sampling,
        config=config_best,
        data_dir=_cli_args.data_dir,
    )
    _best_cache_path = os.path.join(_best_model_dir, _atf_cache_filename(_cache_meta))
    _loaded_path, _cached_predictions = _load_matching_atf_cache(_best_model_dir, _cache_meta)
    if _cached_predictions is not None:
        print(f"⚡ Loading cached ATF predictions from {_loaded_path}")
        best_model_atf_predictions = _cached_predictions
        if _best_w in best_model_atf_predictions:
            print(f"   Loaded predictions shape: {list(best_model_atf_predictions[_best_w].shape)}")
    else:
        if atf_mag_gt is not None and ref_config is not None:
            print(f"🔄 No cache found — running ATF inference for {_n_src} sources...")
            best_model_atf_predictions = get_your_model_atf_predictions(
                set_encoder_best, ode_3d_best, config_best, device,
                atf_mag_gt, ref_config, _best_model_freq_up_to, eval_freq_up_to, num_sources_eval,
                single_guidance=_best_w, random_M_sampling=random_M_sampling, model_name=best_model
            )
            _save_atf_cache(_best_cache_path, _cache_meta, best_model_atf_predictions)
            print(f"✅ ATF predictions cached to {_best_cache_path}")
        else:
            print("  Skipping ATF prediction cache (FSMPAE disabled, no reference GT).")

if GENERATE_PLOTS and all_your_results:
    # Plot distributions for each model individually and create combined plot
    ref_lsd = source_indices = None

    # Prepare data for combined plot
    all_model_lsd = {}
    colors = plt.cm.tab10(np.linspace(0, 1, len(MODEL_NAMES)))

    # Plot individual model distributions and collect data for combined plot
    for i, model_name in enumerate(MODEL_NAMES):
        if model_name in all_your_results:
            # Get best guidance for this model
            model_best_guidance = None
            model_best_lsd = float('inf')
            for w in guidance_scales:
                if all_your_results[model_name][M_values[0]][w]['lsd_mean'] < model_best_lsd:
                    model_best_lsd = all_your_results[model_name][M_values[0]][w]['lsd_mean']
                    model_best_guidance = w

            model_per_source = all_your_results[model_name][M_values[0]][model_best_guidance]['per_source_errors']
            model_lsd = [model_per_source[j]['lsd'] for j in range(len(model_per_source))]
            _src_indices = source_indices if source_indices is not None else list(range(len(model_lsd)))

            # Store for combined plot
            all_model_lsd[model_name] = {'values': model_lsd, 'guidance': model_best_guidance, 'color': colors[i]}

            # Save individual model plots - create unique subdirectory for each model
            base_model_dir = os.path.dirname(MULTI_MODEL_PATHS[i])
            # Use the filename (without extension) as subdirectory name for uniqueness
            model_filename = os.path.basename(MULTI_MODEL_PATHS[i]).replace('.pt', '')
            if model_filename == 'model':
                model_dir = base_model_dir  # Backward compatibility
            else:
                model_dir = os.path.join(base_model_dir, f"eval_{model_filename}")
            os.makedirs(model_dir, exist_ok=True)

            # Individual LSD plot
            plt.figure(figsize=(12, 6))
            model_mean = np.mean(model_lsd)
            if ref_lsd is not None:
                ref_mean = np.mean(ref_lsd)
                plt.plot(_src_indices, ref_lsd, 'r-', label=f'Reference (mean: {ref_mean:.4f} dB)', alpha=0.7)
            plt.plot(_src_indices, model_lsd, 'b-', label=f'{model_name} w={model_best_guidance} (mean: {model_mean:.4f} dB)', alpha=0.7)
            plt.xlabel('Source Index')
            plt.ylabel('LSD Error (dB)')
            plt.title(f'LSD Distribution - {model_name}')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.savefig(os.path.join(model_dir, 'lsd_distribution.pdf'), dpi=300, bbox_inches='tight')
            plt.close()

            print(f"Individual distribution plots saved to {model_dir}/")

    # Create combined plots in parent directory
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(MULTI_MODEL_PATHS[0])))  # Go up two levels
    os.makedirs(parent_dir, exist_ok=True)

    # Combined LSD plot
    _comb_src_indices = source_indices if source_indices is not None else list(range(len(next(iter(all_model_lsd.values()))['values'])))
    plt.figure(figsize=(14, 8))

    for model_name, data in all_model_lsd.items():
        model_mean = np.mean(data['values'])
        # Wrap long names: insert newline before parenthetical info
        wrapped = model_name.replace('_', '\n', 1) if len(model_name) > 30 else model_name
        plt.plot(_comb_src_indices, data['values'], '-', color=data['color'],
                label=f'{wrapped}\nw={data["guidance"]} mean={model_mean:.4f} dB', alpha=0.7)

    plt.xlabel('Source Index')
    plt.ylabel('LSD Error (dB)')
    plt.title('LSD Distribution Comparison - All Models')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(parent_dir, 'Zcombined_lsd_distribution.pdf'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nCombined distribution plots saved to {parent_dir}/")

    # Plot ATF comparisons using the best model (no reloading needed!)
    print("\n3. Generating ATF comparison plots...")
    print(f"Using best model for plots: {best_model}")

    if best_model and best_model in all_your_predictions and best_model_atf_predictions is not None:
        your_atf_predictions = best_model_atf_predictions
        _best_w = best_results['guidance']
        _best_model_dir = os.path.dirname(all_model_paths[best_model])

        # Build per_source_lsd list for representative source selection in PDF plots
        # _best_w already set above in cache block
        _best_M = min(all_your_results[best_model].keys())  # use smallest M for consistency
        _per_src_errors = all_your_results[best_model][_best_M][_best_w].get('per_source_errors', {})
        _per_source_lsd = [_per_src_errors[i]['lsd'] for i in sorted(_per_src_errors.keys())] if _per_src_errors else None

        # For ATF PDFs, overlay the first configured EEAE (if any) to avoid clutter.
        atf_mag_est_eeae_for_plot = None
        if eeae_atf_est_by_id:
            _first_eeae_id = next(iter(eeae_atf_est_by_id.keys()))
            atf_mag_est_eeae_for_plot = eeae_atf_est_by_id[_first_eeae_id]

        # Source labels/filenames use absolute dataset indices: offset = first test-split index.
        _test_split = all_model_configs[best_model].get('data', {}).get('src_splits', {}).get('test', [0, 0])
        _src_offset = _test_split[0][0] if isinstance(_test_split[0], list) else _test_split[0]

        # Use the already computed best guidance scale
        if atf_mag_gt is not None and ref_config is not None:
            plot_atf_comparisons(atf_mag_est, your_atf_predictions, atf_mag_gt, ref_config,
                                eval_freq_up_to, num_sources_eval, best_guidance=best_results['guidance'],
                                output_dir=_best_model_dir,
                                atf_mag_est_eeae=atf_mag_est_eeae_for_plot,
                                grid_xyz=all_model_grid_xyz,
                                per_source_lsd=_per_source_lsd,
                                src_offset=_src_offset)
        else:
            print("  Skipping ATF comparison plots (FSMPAE disabled, no reference GT).")
    else:
        print("Could not find best model for plotting")


