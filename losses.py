"""Loss functions for Project #9.

- CrossEntropy (baseline)
- FocalLoss (baseline)
- CBFocalLoss: effective-number class-balanced focal (Cui et al. 2019) (baseline)
- DACFLoss (Candidate B, novel): difficulty-adaptive calibrated focal loss.

DACF: L = w_c * (1-p_t)^gamma * CE  -  lambda * m_c * log p_t
  where w_c = (1-beta)/(1-beta^n_c) (effective number),
  m_c = relu(EMA_conf[c] - EMA_acc[c]) is the per-class calibration margin,
  adapted online from batch statistics.

Gradient motivation (full derivation in PAPER.md): the margin term is a
first-order penalty on overconfident predictions for miscalibrated classes:
d/dm_c term pushes p_t down exactly when the class's confidence systematically
exceeds its accuracy. Focal alone sharpens the predictive distribution and
provably worsens ECE on minority classes; the margin counteracts this only
where miscalibration is measured, leaving well-calibrated classes at pure
focal behavior.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction="none")
        p_t = torch.exp(-ce)
        loss = ((1 - p_t) ** self.gamma) * ce
        return loss.mean() if self.reduction == "mean" else loss


class CBFocalLoss(nn.Module):
    """Class-balanced focal: effective-number weights x focal."""
    def __init__(self, class_counts, beta=0.9999, gamma=2.0):
        super().__init__()
        counts = torch.tensor(class_counts, dtype=torch.float)
        eff = (1 - beta) / (1 - torch.pow(beta, counts))
        self.register_buffer("w", eff / eff.sum() * len(counts))
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction="none")
        p_t = torch.exp(-ce)
        loss = self.w[targets] * ((1 - p_t) ** self.gamma) * ce
        return loss.mean()


class DACFLoss(nn.Module):
    """Difficulty-Adaptive Calibrated Focal Loss (Candidate B, novel)."""
    def __init__(self, class_counts, beta=0.9999, gamma=2.0, lam=0.5,
                 ema=0.9):
        super().__init__()
        counts = torch.tensor(class_counts, dtype=torch.float)
        eff = (1 - beta) / (1 - torch.pow(beta, counts))
        self.register_buffer("w", eff / eff.sum() * len(counts))
        self.gamma = gamma
        self.lam = lam
        self.ema = ema
        n = len(counts)
        self.register_buffer("conf_ema", torch.zeros(n))
        self.register_buffer("acc_ema", torch.zeros(n))
        self.register_buffer("seen", torch.zeros(n))

    @torch.no_grad()
    def _update_stats(self, probs, targets):
        pred = probs.argmax(1)
        conf = probs.max(1).values
        for c in range(len(self.conf_ema)):
            mask = targets == c
            if mask.any():
                bc = conf[mask].mean()
                ba = (pred[mask] == c).float().mean()
                if self.seen[c] == 0:
                    self.conf_ema[c] = bc
                    self.acc_ema[c] = ba
                    self.seen[c] = 1
                else:
                    self.conf_ema[c] = self.ema * self.conf_ema[c] + (1 - self.ema) * bc
                    self.acc_ema[c] = self.ema * self.acc_ema[c] + (1 - self.ema) * ba

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        self._update_stats(probs.detach(), targets)
        ce = F.cross_entropy(logits, targets, reduction="none")
        p_t = torch.exp(-ce)
        focal_term = self.w[targets] * ((1 - p_t) ** self.gamma) * ce
        margin = F.relu(self.conf_ema - self.acc_ema).detach()  # (C,)
        calib_term = self.lam * margin[targets] * ce  # -log p_t == ce
        return (focal_term + calib_term).mean()

    def margins(self):
        return F.relu(self.conf_ema - self.acc_ema).cpu().tolist()
