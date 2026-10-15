[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](requirements.txt)
[![GitHub stars](https://img.shields.io/github/stars/habib-analyst/medmodel-discovery.svg)](https://github.com/habib-analyst/medmodel-discovery)

**Three new ideas for small-data medical image classification, tested honestly on MedMNIST against ResNet-18 and ViT-Tiny — and the two that didn't survive their own success criteria are reported as failures.**

## Quickstart

```bash
git clone https://github.com/habib-analyst/medmodel-discovery.git
cd medmodel-discovery
pip install -r requirements.txt
# Data: MedMNIST .npz files in $MMD_ROOT (or ~/.medmnist)
```

Zero-data smoke test (instantiates PLAN-Net, single forward pass, real output):

```python
import torch
from models import PLANNet

m = PLANNet(in_ch=3, num_classes=7)
logits = m(torch.rand(4, 3, 28, 28))
print(logits.shape)   # -> torch.Size([4, 7])
```

Full experiment campaigns (GPU recommended; Kaggle script included):

```bash
python run_experiments.py A   # architectures x datasets (PLAN-Net vs ResNet-18 / ViT-Tiny / GCViT-lite+Perceiver-lite)
python run_experiments.py B   # loss comparison (CE / focal / class-balanced focal / DACF)
python run_experiments.py C   # test-time adaptation (single-pass / naive TTA / CG-TTC)
```

Environment knobs: `MMD_EPOCHS`, `MMD_SEEDS`, `MMD_BATCH`, `MMD_DS` (comma-separated subset), `MMD_ROOT`.

```bash
python -m pytest test_smoke.py -q
```

## Measured results

Exp-A: architectures, 3 seeds each (validation accuracy / AUC, seed means, seed ranges in brackets). The design criterion — PLAN-Net ≥ best baseline acc *and* AUC on ≥2/3 datasets at ≤1.5M params — was **MISSED**.

| dataset | model | acc | AUC |
|---|---|---|---|
| pneumonia | PLAN-Net | **0.9727** [0.9695, 0.9771] | 0.9932 [0.9899, 0.9953] |
| pneumonia | ResNet-18 | 0.9631 [0.9599, 0.9656] | **0.9953** [0.9946, 0.9963] |
| pneumonia | ViT-Tiny | 0.9586 [0.9523, 0.9618] | 0.9877 [0.9838, 0.9899] |
| derma | PLAN-Net | 0.7468 [0.7418, 0.7567] | 0.9206 [0.9145, 0.9266] |
| derma | ResNet-18 | **0.7694** [0.7637, 0.7747] | **0.9255** [0.9249, 0.9267] |
| blood | PLAN-Net | 0.9338 [0.9258, 0.9381] | 0.9955 [0.9949, 0.9958] |
| blood | ResNet-18 | **0.9628** [0.9614, 0.9655] | **0.9984** [0.9981, 0.9986] |

PLAN-Net wins pneumonia accuracy outright but loses its AUC; on derma and blood ResNet-18 wins both. It does beat ViT-Tiny and the GCViT-lite+Perceiver-lite analog on all three datasets. (28x28 inputs likely understate the pyramidal design's fine-detail advantage.)

Exp-B: DACF loss on derma — **collapsed**. DACF 0.5673 vs focal 0.7494, an 18.2pp collapse (genuine loss behavior: β=0.9999 too aggressive), worse than a majority-class predictor. On blood it behaved (0.9356, within noise of focal).

Exp-C: CG-TTC on pneumonia scored 0.8654 — exactly tied with single-pass, so the gating added nothing; it beat naive TTA on clean derma (0.7198 vs 0.6610) but its referral rates were 0.43/0.69 against a ≤0.15 budget, and naive TTA beat it under noise shift (0.6058 vs 0.5064).

## How it works

- **PLAN-Net (architecture):** pyramidal latent arrays (16/32/64) with top-down routing gates — coarse latents decide which fine latents update, replacing Perceiver IO's flat bottleneck. ~524K params.
- **DACF (loss):** difficulty-adaptive calibrated focal loss — effective-number × focal × an online per-class calibration margin that penalizes overconfident miscalibrated classes.
- **CG-TTC (inference):** consistency-gated test-time consensus — K augmented views vote only when they agree (JS divergence < τ); disagreement triggers human referral instead of silent averaging.
- `papers/PAPER.md` holds the full writeup with negative results and limitations.

## Citation

If you use this code, cite the MedMNIST benchmark:

```bibtex
@Article{medmnistv2,
    author  = {Jiancheng Yang and Rui Shi and Donglai Wei and Zequan Liu and
               Lin Zhao and Bilian Ke and Hanspeter Pfister and Bingbing Ni},
    title   = {MedMNIST v2 - A large-scale lightweight benchmark for 2D and 3D
               biomedical image classification},
    journal = {Scientific Data},
    year    = {2023},
    volume  = {10},
    number  = {41},
}
```

---
Built by [Habib Ur Rehman](https://github.com/habib-analyst) · ML/AI Engineer & healthcare-AI researcher ·
[LinkedIn](https://www.linkedin.com/in/hur-dev) · [habib.top](https://habib.top)
