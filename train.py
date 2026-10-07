"""Training + evaluation harness for Project #9.

Metrics: accuracy, macro AUC (OvR), ECE (15 bins), per-class recall.
Losses: ce | focal | cb-focal | dacf (see losses.py).
"""
import time
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from models import build_model, count_params
from losses import FocalLoss, CBFocalLoss, DACFLoss
from data import get_loaders


def build_loss(name, class_counts):
    if name == "ce":
        return torch.nn.CrossEntropyLoss()
    if name == "focal":
        return FocalLoss()
    if name == "cb-focal":
        return CBFocalLoss(class_counts)
    if name == "dacf":
        return DACFLoss(class_counts)
    raise ValueError(name)


@torch.no_grad()
def evaluate(model, loader, n_classes):
    model.eval()
    all_p, all_t = [], []
    for x, y in loader:
        all_p.append(F.softmax(model(x), 1).cpu())
        all_t.append(y)
    p = torch.cat(all_p).numpy()
    t = torch.cat(all_t).numpy()
    pred = p.argmax(1)
    acc = (pred == t).mean()
    try:
        if n_classes == 2:
            auc = roc_auc_score(t, p[:, 1])
        else:
            auc = roc_auc_score(t, p, multi_class="ovr", average="macro")
    except ValueError:
        auc = float("nan")
    # ECE, 15 bins
    ece = 0.0
    conf = p.max(1)
    bins = np.linspace(0, 1, 16)
    for i in range(15):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if m.any():
            ece += m.mean() * abs((pred[m] == t[m]).mean() - conf[m].mean())
    # per-class recall
    recalls = []
    for c in range(n_classes):
        m = t == c
        recalls.append((pred[m] == c).mean() if m.any() else float("nan"))
    return {"acc": float(acc), "auc": float(auc), "ece": float(ece),
            "recalls": [float(r) for r in recalls]}


def train_one(model_name, dataset, loss_name="ce", epochs=15, lr=3e-3,
              batch=128, seed=0, subsample=None, log=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_l, val_l, test_l, n_classes = get_loaders(dataset, batch, subsample, seed)
    in_ch = 3
    model = build_model(model_name, in_ch, n_classes)
    # class counts for balanced losses
    counts = np.zeros(n_classes, dtype=int)
    for _, y in train_l:
        for c in y.numpy():
            counts[c] += 1
    criterion = build_loss(loss_name, counts.tolist())
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        for x, y in train_l:
            opt.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            opt.step()
        sched.step()
    secs = time.time() - t0
    res = {"test": evaluate(model, test_l, n_classes),
           "val": evaluate(model, val_l, n_classes),
           "params": count_params(model), "train_secs": secs,
           "model": model_name, "dataset": dataset, "loss": loss_name,
           "seed": seed, "epochs": epochs}
    if isinstance(criterion, DACFLoss):
        res["dacf_margins"] = criterion.margins()
    return res, model


if __name__ == "__main__":
    import json, sys
    model_name, dataset = sys.argv[1], sys.argv[2]
    loss = sys.argv[3] if len(sys.argv) > 3 else "ce"
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    res, _ = train_one(model_name, dataset, loss, seed=seed)
    res.pop("test", None)  # keep output small; print val+test summary
    print(json.dumps({k: v for k, v in res.items() if k != "model"}, indent=1)[:2000])
