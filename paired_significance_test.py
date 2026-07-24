"""Paired statistical comparison of SF-Flow vs an external baseline (per-source test LSD).

Loads cached SF-Flow predictions from the given model folder (run evaluate.py
first to create the cache), recomputes per-source LSD for SF-Flow and the
baseline, then runs a paired Wilcoxon signed-rank test. Also reports NCC
(per-bin spatial cosine / Pearson r) as a spatial-detail metric.

The baseline predictions are a torch tensor of shape [1331, F, num_test_sources]
in dB. Microphone ordering:
  --baseline_order grid  (default): flattened 11x11x11 grid order, identical to
                                    SF-Flow's own predictions.
  --baseline_order npz            : posMic order of the dataset's .npz files
                                    (auto-permuted to grid order via coordinates).

Outputs (written next to the model checkpoint):
  paired_test_freq{F}_M{M}.txt   - full text report
  paired_test_freq{F}_M{M}.npz   - per-source LSD/NCC arrays + p-values

Usage:
  python paired_significance_test.py <model folder or checkpoint .pt> \
      --baseline_pred my_baseline_predictions.pt [--M 5]
"""
import argparse
import glob
import os
import re

import numpy as np
import torch
from scipy import stats

parser = argparse.ArgumentParser()
parser.add_argument('model_path', type=str,
                    help="Model checkpoint .pt or its experiment folder (cache + outputs live there)")
parser.add_argument('--baseline_pred', type=str, required=True,
                    help="Baseline prediction tensor .pt of shape [1331, F, num_test_sources] in dB")
parser.add_argument('--baseline_order', type=str, default='grid', choices=['grid', 'npz'],
                    help="Microphone ordering of the baseline predictions (see module docstring)")
parser.add_argument('--M', type=int, default=5, help="Observation count of the cached predictions")
parser.add_argument('--data_dir', type=str,
                    default="data/ir_fs2000_s8192_m1331_room4.0x6.0x3.0_rt200")
args = parser.parse_args()

model_dir = args.model_path if os.path.isdir(args.model_path) else os.path.dirname(os.path.abspath(args.model_path))

_report_lines = []
def say(msg=""):
    print(msg)
    _report_lines.append(str(msg))

# --- SF-Flow cached predictions (denormalized dB, [1331, F, S]) ---
pred_sflow = None
for cache_path in sorted(glob.glob(os.path.join(model_dir, "atf_pred_cache_*.pt"))):
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    meta = payload.get("metadata", {})
    cache_room = os.path.basename(str(meta.get("data_dir") or ""))
    if cache_room and cache_room != os.path.basename(args.data_dir.rstrip("/")):
        continue  # cache built from a different dataset
    if meta.get("M") == args.M:
        w = sorted(payload["atf_predictions"].keys())[0]
        pred_sflow = payload["atf_predictions"][w].float()
        F = meta.get("eval_freq_up_to", pred_sflow.shape[1])
        say(f"SF-Flow cache: {os.path.basename(cache_path)} (M={meta.get('M')}, w={w}, "
            f"{F} bins, {pred_sflow.shape[2]} sources, {meta.get('num_timesteps')} ODE steps)")
        break
if pred_sflow is None:
    raise FileNotFoundError(
        f"No atf_pred_cache_*.pt with M={args.M} in {model_dir}. Run evaluate.py first to create it.")
S = pred_sflow.shape[2]

# --- Ground truth from processed test cubes (raw dB, grid order) ---
m_ver = re.search(r'_(r\d+)_', os.path.basename(model_dir.rstrip('/')))
dataset_version = m_ver.group(1) if m_ver else 'r1'
gt_path = os.path.join(args.data_dir, f"processed_atf3d_test_freqs{F}_{dataset_version}.pt")
gt_data = torch.load(gt_path, map_location="cpu", weights_only=False)
cubes = gt_data["cubes"]            # [S, F, 11, 11, 11]
grid_xyz = gt_data["grid_xyz"]      # [1331, 3] grid order
say(f"GT: {os.path.basename(gt_path)}, cubes {tuple(cubes.shape)}, "
    f"dB range [{cubes.min():.1f}, {cubes.max():.1f}]")
gt = torch.stack([cubes[i, :F].reshape(F, -1).T for i in range(S)], dim=2)  # [1331, F, S]

def per_source_lsd(pred):
    lsd_pos = torch.sqrt(((pred - gt) ** 2).mean(dim=1))   # [1331, S]
    return lsd_pos.mean(dim=0).numpy()

lsd_sflow = per_source_lsd(pred_sflow)
say(f"\nSF-Flow LSD ({F} bins, M={args.M}): {lsd_sflow.mean():.4f} +- {lsd_sflow.std():.4f}")

# --- Baseline predictions ---
pred_base = torch.load(args.baseline_pred, map_location="cpu", weights_only=False)
if pred_base.dim() == 4 and pred_base.shape[0] == 1:
    pred_base = pred_base.squeeze(0)
if pred_base.shape[1] < F or pred_base.shape[2] != S:
    raise ValueError(f"Baseline shape {tuple(pred_base.shape)} incompatible with [1331, >={F}, {S}]")
pred_base = pred_base[:, :F, :].float()

if args.baseline_order == 'npz':
    # Permute from the dataset's npz posMic order to grid order via coordinates.
    npz_files = sorted(glob.glob(os.path.join(args.data_dir, "data_s*.npz")))
    pos_mic_npz = np.load(npz_files[0])["posMic"].astype(np.float64)  # [1331, 3]
    grid_map = {tuple(np.round(c, 6)): i for i, c in enumerate(grid_xyz.numpy().astype(np.float64))}
    perm_npz_to_grid = np.array([grid_map[tuple(np.round(c, 6))] for c in pos_mic_npz])
    inv_perm = np.argsort(perm_npz_to_grid)
    pred_base = pred_base[inv_perm, :, :]
    say(f"Baseline reordered from npz posMic order to grid order.")

lsd_base = per_source_lsd(pred_base)
base_name = os.path.basename(args.baseline_pred)
say(f"Baseline ({base_name}): LSD {lsd_base.mean():.4f} +- {lsd_base.std():.4f}")

# --- Paired Wilcoxon signed-rank test on per-source LSD ---
diff = lsd_base - lsd_sflow                  # >0: SF-Flow better on that source
res_two = stats.wilcoxon(lsd_sflow, lsd_base, alternative="two-sided")
res_one = stats.wilcoxon(lsd_sflow, lsd_base, alternative="less")
ranks = stats.rankdata(np.abs(diff))
rbc = (ranks[diff > 0].sum() - ranks[diff < 0].sum()) / ranks.sum()
t_res = stats.ttest_rel(lsd_sflow, lsd_base)

say(f"\n=== Paired Wilcoxon signed-rank test, per-source LSD (n={S}) ===")
say(f"SF-Flow better on {int((diff > 0).sum())}/{S} sources; "
    f"median paired diff {np.median(diff):.3f} dB (mean {diff.mean():.3f} dB)")
say(f"two-sided: W={res_two.statistic:.1f}, p={res_two.pvalue:.3e}")
say(f"one-sided (SF-Flow < baseline): p={res_one.pvalue:.3e}")
say(f"matched-pairs rank-biserial correlation r={rbc:.3f}")
say(f"(paired t-test reference: t={t_res.statistic:.2f}, p={t_res.pvalue:.3e})")

# --- NCC: per-bin spatial similarity between predicted and GT fields ---
# For each source and bin, the 1331-point spatial field is compared;
# plain = |cos|; zero-mean = Pearson |r|, which removes the dB offset shared
# by both fields.
def per_source_ncc(pred, zero_mean):
    a = pred.numpy().astype(np.float64)      # [1331, F, S]
    b = gt.numpy().astype(np.float64)
    if zero_mean:
        a = a - a.mean(axis=0, keepdims=True)
        b = b - b.mean(axis=0, keepdims=True)
    num = np.abs(np.einsum('mfs,mfs->fs', a, b))
    den = np.linalg.norm(a, axis=0) * np.linalg.norm(b, axis=0)
    return (num / den).mean(axis=0)          # mean over bins -> [S]

say(f"\n=== NCC per source (spatial field per bin, averaged over {F} bins) ===")
ncc = {}
for zm, label in [(False, "plain cosine"), (True, "zero-mean (Pearson |r|)")]:
    ncc_sf = per_source_ncc(pred_sflow, zm)
    ncc_bl = per_source_ncc(pred_base, zm)
    p_ncc = stats.wilcoxon(ncc_sf, ncc_bl, alternative="greater").pvalue
    ncc[label] = (ncc_sf, ncc_bl, p_ncc)
    say(f"{label:26s} SF-Flow {ncc_sf.mean():.4f} +- {ncc_sf.std():.4f} | "
        f"baseline {ncc_bl.mean():.4f} +- {ncc_bl.std():.4f} | one-sided p={p_ncc:.3e}")

# --- Save outputs into the model folder ---
base = os.path.join(model_dir, f"paired_test_freq{F}_M{args.M}")
np.savez(base + ".npz",
         lsd_sflow=lsd_sflow, lsd_baseline=lsd_base,
         ncc_plain_sflow=ncc["plain cosine"][0], ncc_plain_baseline=ncc["plain cosine"][1],
         ncc_pearson_sflow=ncc["zero-mean (Pearson |r|)"][0],
         ncc_pearson_baseline=ncc["zero-mean (Pearson |r|)"][1],
         wilcoxon_p_two_sided=res_two.pvalue, wilcoxon_p_one_sided=res_one.pvalue,
         rank_biserial=rbc, baseline_name=base_name)
with open(base + ".txt", "w") as f:
    f.write("\n".join(_report_lines) + "\n")
print(f"\nSaved: {base}.txt / .npz")
