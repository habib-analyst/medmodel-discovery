# MedModel-Discovery — New Models & Algorithms for Medical Imaging

Novel architecture + loss + inference algorithm for small-data medical image
classification, extending the GCViT × Perceiver IO research line
(Habib Ur Rehman, J. of Supercomputing 2025).

## The three candidates

| # | Name | Type | One-line claim |
|---|------|------|----------------|
| A | **PLAN-Net** | architecture | Pyramidal latent arrays (16/32/64) with **top-down routing gates**: coarse latents decide which fine latents update — replaces Perceiver IO's flat bottleneck |
| B | **DACF** | loss | Difficulty-adaptive calibrated focal loss: effective-number × focal × an online per-class calibration margin that penalizes overconfident miscalibrated classes |
| C | **CG-TTC** | inference | Consistency-gated test-time consensus: K augmented views vote only when they agree (JS divergence < τ); disagreement triggers human referral instead of silent averaging |

## Results (measured, seed-averaged)

See [`PAPER.md`](PAPER.md) for the full writeup. Headline numbers land here
once the experiment campaign completes.

## Quickstart

```bash
pip install -r requirements.txt
# data: MedMNIST npz files in $MMD_ROOT (or ~/.medmnist)
python run_experiments.py A   # architectures × datasets
python run_experiments.py B   # loss comparison
python run_experiments.py C   # TTA strategies
```

Environment knobs: `MMD_EPOCHS`, `MMD_SEEDS`, `MMD_BATCH`, `MMD_DS`
(comma-separated subset), `MMD_ROOT` (npz dir).

## Repo layout

- `models.py` — PLAN-Net, GCViT-lite+Perceiver-lite, ResNet-18, ViT-Tiny
- `losses.py` — CE, focal, class-balanced focal, DACF
- `tta.py` — single-pass, naive TTA, CG-TTC
- `data.py`, `train.py` — MedMNIST loading, training/eval harness
- `run_experiments.py` — experiment campaigns A/B/C
- `kaggle_full_run.py` — full-scale GPU run script for Kaggle

## Citation

If you use this code, cite the MedMNIST benchmark (Yang et al., 2023) and
link this repo.

## License

MIT
