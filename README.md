# SF-Flow: Sound field magnitude estimation via flow matching guided by sparse measurements

Official implementation of the IWAENC 2026 paper
*"SF-Flow: Sound field magnitude estimation via flow matching guided by sparse measurements"*
(E. Erdem, S. Koyama, T. Nakamura, O. Das, Z. Cvetković).

SF-Flow reconstructs a dense 3D acoustic transfer function (ATF) magnitude field
(64 frequency bins × 11×11×11 grid) from a sparse, variable-size set of microphone
observations, using conditional Flow Matching with a permutation-invariant set encoder
and a 3D U-Net.

Project page: https://egerdem.github.io/sf-flow/

## Repository layout

| File | Purpose |
| --- | --- |
| `generate_dataset.py` | Simulate the RIR/ATF dataset with pyroomacoustics (deterministic, seed 0) |
| `train.py` | Train an SF-Flow model |
| `evaluate.py` | Evaluate a checkpoint on the test set (LSD, plots, prediction cache) |
| `paired_significance_test.py` | Paired Wilcoxon test + NCC against an external baseline |
| `strip_checkpoint.py` | Remove optimizer state from a checkpoint for sharing (~3× smaller) |
| `fm_utils.py` | Models, samplers, Flow Matching training loop |
| `inference.py` | Model factory, checkpoint loading, LSD metric (library) |
| `irdata_utils.py` | Room geometry helpers for the data generator |
| `idx_mes_pos_*.npy` | Fixed per-source microphone-selection matrices (evaluation protocol) |
| `coord_stats_r*.pt` | Cached coordinate-normalization statistics (regenerated automatically if absent) |

## Installation

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## 1. Generate the dataset

Room: 4×6×3 m, T60 = 0.2 s, fs = 2 kHz; ATFs from 128-sample time-aligned RIRs
(64 bins up to 1 kHz) on an 11×11×11 grid (0.1 m spacing) in a central 1 m³ region.
Source splits (R1): train 0–820, validation 820–922, test 922–1024.

```bash
python generate_dataset.py -s 1024    # R1 Dataset: first 1024 sources
python generate_dataset.py            # R2/R3 Datasets: full 8192-source for paper's dataset scaling ablation runs
```

Generation is deterministic (seed 0): the first 1024 sources of the 8192-source run are
identical to the 1024-source run, so `-s 1024` reproduces the paper's R1 dataset exactly.
Output goes to `data/ir_fs2000_s<N>_m1331_room4.0x6.0x3.0_rt200/`.
On the first training run the `.npz` files are packed into `processed_atf3d_*.pt` inside
that directory and reused automatically afterwards, so the data directory
must be writable. Delete those `.pt` files, and `coord_stats_*.pt`, if you regenerate the
dataset with different parameters — they are keyed by the dataset name (r1,r2, etc.), not by dataset contents.

To check the pipeline end to end before committing to a full run, generate a small dataset
(`python generate_dataset.py -s 102`) and train it for a few iterations with a model name
starting with `smoke_`, which selects a matching 80/10/12 split.


## 2. Train

The exact command used for the paper's 0–20 model (R1); the other three differ only in
`--freq_up_to` (30, 40, 64) and `--model_name`:

```bash
python train.py \
    --model_name SFFlow_freq20 --freq_up_to 20 \
    --geo_conditioning --freq_ctx \
    --data_dir data/ir_fs2000_s1024_m1331_room4.0x6.0x3.0_rt200/ \
    --num_iterations 400000 --warmup_iterations 5000 --decay_iterations 100000 \
    --lr 1e-4 --min_lr 1e-5 --batch_size 4 \
    --M_range 5,10,20,50 --M_val_fixed 5 \
    --validation_interval 200 --checkpoint_interval 200000 --save_start_iter 70000
```

Defaults already match this, so the short form
`python train.py --model_name SFFlow_freq20 --freq_up_to 20 --geo_conditioning --freq_ctx`
is equivalent except for `--save_start_iter`, which the paper runs set to 70000 to avoid
writing a best-model file on every early improvement (leave it at 0 for short test runs,
otherwise nothing is saved before iteration 70000).

Which split is used comes from `--dataset_version`. Pass `--dataset_version r3` (or r1,r2) to make sure. 
Check the `Loaded N ATF-3D cubes` and `Using dataset version:` line the loader prints at startup. If `--dataset_version`
Left unset, it is inferred from `--model_name`: a name containing `smoke` selects the small split, `BIGDATA` and `BIG8192`
select the R2 and R3 scaling splits, and **any other name selects R1**.

Training progress is printed to `stdout` and to `<run>/log.txt`: a `** New best LSD ... **`
line each time validation improves, plus a full status line every 1000 iterations. The
live `tqdm` bar carries the running loss (refreshed every 100 iterations).
The best model is written to `<run>/model_<iter>_lsd<value>.pt` whenever validation
improves; `<run>/checkpoints/` additionally receives a periodic full-state snapshot that
`--resume_from_checkpoint <path>` can restart from.
~20 s/epoch on an RTX A5000; the best checkpoint is typically reached in 2.4–5.8 h
depending on the frequency range. Less than an hour for 0-20fbins, on a NVIDIA A6000 Ada GPI.

Add `--wandb` for Weights & Biases logging. Authenticate either by running `wandb login`
once, by exporting `WANDB_API_KEY=<key>`, or by passing `--wandb_key <key>`.

## 3. Evaluate

```bash
python evaluate.py --model_path experiments/<run>/model.pt --data_dir data/ir_fs2000_s1024_m1331_room4.0x6.0x3.0_rt200/
```

Reports LSD (mean ± std over the 102 test sources) for M=5 observations with the fixed
microphone-selection protocol, and writes per-source distribution plots and ATF
comparison PDFs into the model folder. Predictions are cached (`atf_pred_cache_*.pt`);
pass `--use_cache` to reuse them on re-runs. `--M 1 5 10 20 50` reproduces the
observation-count ablation.

Expected test LSD (dB, M=5, R1 models; paper Table 1):

| bins (max freq) | 0–20 (312 Hz) | 0–30 (468 Hz) | 0–40 (625 Hz) | 0–64 (1000 Hz) |
| --- | --- | --- | --- | --- |
| SF-Flow | 1.75 ± 0.58 | 3.17 ± 0.67 | 4.16 ± 0.63 | 5.56 ± 0.52 |

Training is seeded but retrained models may differ in the
second decimal.

## 4. Compare against your own baseline

```bash
python paired_significance_test.py experiments/<run>/ \
    --baseline_pred my_predictions.pt          # [1331, F, 102] dB, grid order
```

Runs a paired Wilcoxon signed-rank test on per-source LSD plus NCC
(per-bin spatial correlation) and writes a report next to the checkpoint.
Run `evaluate.py` first so the SF-Flow prediction cache exists.

## Pretrained checkpoints

The four paper models (dataset R1, optimizer state stripped, ~460 MB each) are available at
**[huggingface.co/egeerdem/sf-flow](https://huggingface.co/egeerdem/sf-flow)**. Download a
checkpoint and pass it to `evaluate.py`:

| file | freq bins (max freq) | val LSD (dB) | paper test LSD (dB, M=5) |
| --- | --- | --- | --- |
| `sfflow_r1_freq20.pt` | 0–20 (312 Hz) | 1.7555 | 1.75 ± 0.58 |
| `sfflow_r1_freq30.pt` | 0–30 (468 Hz) | 3.1703 | 3.17 ± 0.67 |
| `sfflow_r1_freq40.pt` | 0–40 (625 Hz) | 4.1857 | 4.16 ± 0.63 |
| `sfflow_r1_freq64.pt` | 0–64 (1000 Hz) | 5.5800 | 5.56 ± 0.52 |

```bash
hf download egeerdem/sf-flow sfflow_r1_freq30.pt --local-dir .
python evaluate.py --model_path sfflow_r1_freq30.pt --data_dir data/ir_fs2000_s1024_m1331_room4.0x6.0x3.0_rt200/
```

## Citation

```bibtex
@inproceedings{erdem2026sfflow,
  author    = {Erdem, Ege and Koyama, Shoichi and Nakamura, Tomohiko and Das, Orchisama and Cvetkovi\'{c}, Zoran},
  title     = {{SF-Flow}: Sound field magnitude estimation via flow matching guided by sparse measurements},
  booktitle = {Proc. Int. Workshop Acoust. Signal Enhancement (IWAENC)},
  year      = {2026}
}
```

## License

MIT — see [LICENSE](LICENSE).

The autoencoder (AE) and kernel ridge regression (KRR) baselines compared in the paper are
from Koyama and Ishizuka [[arXiv:2506.16729](https://doi.org/10.48550/arXiv.2506.16729)]
and are not part of this release; `paired_significance_test.py` accepts any external
baseline's predictions for comparison.

```bibtex
@article{koyama2025learning,
  title   = {Learning Magnitude Distribution of Sound Fields via Conditioned Autoencoder},
  author  = {Koyama, Shoichi and Ishizuka, Kenji},
  journal = {Proc. 11th Convention of the European Acoustics Association (Forum Acusticum / EuroNoise)},
  year    = {2025},
  doi     = {10.48550/arXiv.2506.16729}
}
```
