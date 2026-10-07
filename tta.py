"""Test-time inference strategies for Project #9.

- single_pass: baseline.
- naive_tta: mean of K augmented views (baseline).
- cg_ttc (Candidate C, novel): Consistency-Gated Test-Time Consensus.
  Runs K stochastic views + the clean view. Computes mean pairwise
  Jensen-Shannon divergence across view predictive distributions.
  If JS < tau (calibrated on validation): output ensemble mean.
  Else: output the clean single-pass prediction AND flag=1 (refer to human).

Unlike naive TTA averaging, disagreement is not averaged away — it becomes
a referral signal. Evaluated on risk-coverage curves.
"""
import math
import torch
import torch.nn.functional as F


def _augment(x):
    # x: (C,H,W) tensor in [0,1]; pure-torch augmentation (no torchvision).
    if torch.rand(1).item() < 0.5:
        x = torch.flip(x, dims=[2])
    # random small rotation + translation via affine grid
    angle = (torch.rand(1).item() - 0.5) * 20 * math.pi / 180
    tx, ty = (torch.rand(2) - 0.5) * 0.1
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    theta = torch.tensor([[cos_a, -sin_a, tx], [sin_a, cos_a, ty]],
                         dtype=x.dtype).unsqueeze(0)
    grid = F.affine_grid(theta, (1,) + x.shape, align_corners=False)
    x = F.grid_sample(x.unsqueeze(0), grid, align_corners=False).squeeze(0)
    # brightness / contrast jitter
    b = 1 + (torch.rand(1).item() - 0.5) * 0.3
    c = 1 + (torch.rand(1).item() - 0.5) * 0.3
    x = torch.clamp((x - 0.5) * c + 0.5 + (b - 1), 0, 1)
    # gaussian noise
    x = torch.clamp(x + torch.randn_like(x) * 0.02, 0, 1)
    return x


def _js_div(p, q, eps=1e-8):
    m = 0.5 * (p + q)
    kl = lambda a, b: (a * (torch.log(a + eps) - torch.log(b + eps))).sum(-1)
    return 0.5 * (kl(p, m) + kl(q, m))


def predict_single(model, x):
    model.eval()
    with torch.no_grad():
        return F.softmax(model(x), dim=1)


def predict_naive_tta(model, x, k=8):
    model.eval()
    with torch.no_grad():
        views = []
        for _ in range(k):
            xb = torch.stack([_augment(z) for z in x]) if x.dim() == 4 else _augment(x).unsqueeze(0)
            views.append(F.softmax(model(xb), dim=1))
        return torch.stack(views).mean(0)


def predict_cg_ttc(model, x, k=8, tau=0.05):
    """Returns (probs, referred_flag_tensor)."""
    model.eval()
    with torch.no_grad():
        clean = F.softmax(model(x), dim=1)
        views = []
        for _ in range(k):
            xb = torch.stack([_augment(z) for z in x]) if x.dim() == 4 else _augment(x).unsqueeze(0)
            views.append(F.softmax(model(xb), dim=1))
        views = torch.stack(views)  # (K,B,C)
        # mean pairwise JS divergence per sample
        K, B, C = views.shape
        js_sum = torch.zeros(B)
        cnt = 0
        for i in range(K):
            for j in range(i + 1, K):
                js_sum += _js_div(views[i], views[j])
                cnt += 1
        js = js_sum / max(cnt, 1)
        agree = js < tau
        ens = views.mean(0)
        out = torch.where(agree.unsqueeze(1), ens, clean)
        referred = (~agree).float()
        return out, referred, js
