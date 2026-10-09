"""Full experiment runner for Project #9.

Experiment A: architectures x datasets (CE loss), 3 seeds.
Experiment B: losses x {dermamnist, bloodmnist-subset} on plannet, 3 seeds.
Experiment C: TTA strategies on trained plannet models (uses saved models).

Usage: python3 run_experiments.py A|B|C
Results appended as JSONL to results.jsonl.
"""
import json
import os
import sys
import torch

from train import train_one, evaluate
from data import get_loaders
from models import build_model

OUT = os.environ.get("MMD_OUT", os.path.join(os.path.dirname(__file__),
                                        "results.jsonl"))
CKPT = os.path.join(os.path.dirname(__file__), "checkpoints")
os.makedirs(CKPT, exist_ok=True)


def _seen_keys():
    """Keys of already-logged successful runs (resume support).

    If the Kaggle session dies mid-matrix, results_full.jsonl still holds
    every completed run; on relaunch, completed triples are skipped.
    """
    seen = set()
    if os.path.exists(OUT):
        with open(OUT) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("status") == "error":
                    continue
                key = tuple(r.get(k) for k in ("exp", "model", "dataset",
                                              "loss", "seed"))
                if None not in key:
                    seen.add(key)
    return seen


SEEN = _seen_keys()
if SEEN:
    print(f"resume: {len(SEEN)} completed runs already logged — skipping",
          flush=True)


def log(rec):
    # Per-run append + fsync: if the Kaggle session dies mid-matrix, every
    # finished run survives (run2 2026-10-08 lost all records for this reason).
    with open(OUT, "a") as f:
        f.write(json.dumps(rec) + "\n")
        f.flush()
        os.fsync(f.fileno())
    head = json.dumps({k: rec[k] for k in ("exp", "model", "dataset", "loss", "seed")
                       if k in rec})
    # error records carry no 'val' (fix 2026-10-09: KeyError 'val' crashed the
    # cell on the first per-run error, defeating the error-isolation hardening)
    if rec.get("status") == "error":
        print(head + f" status=error error={rec.get('error', '')}", flush=True)
    elif "val" in rec:
        print(head + f" val_acc={rec['val']['acc']:.4f} auc={rec['val']['auc']:.4f}",
              flush=True)
    else:
        # Exp-C style records: no 'val' dict, print any *_acc metrics instead
        # (fix 2026-10-09: pre-fix code KeyError'd on Exp-C records)
        extra = {k: round(v, 4) for k, v in rec.items() if k.endswith("_acc")}
        print(head + f" {json.dumps(extra)}", flush=True)


EPOCHS = int(os.environ.get("MMD_EPOCHS", "12"))
SEEDS = int(os.environ.get("MMD_SEEDS", "3"))
BATCH = int(os.environ.get("MMD_BATCH", "128"))
DS_LIST = os.environ.get("MMD_DS", "pneumoniamnist,dermamnist,bloodmnist")

# MMD_ONLY: optional run filter, e.g.
#   "bloodmnist:resnet18:0,1,2;bloodmnist:vit-tiny:0,1,2;bloodmnist:gcvit-perceiver-lite:2"
# Only matching (dataset, model, seed) triples run; everything else is skipped.
# Empty / unset = no filter (full matrix).
ONLY = os.environ.get("MMD_ONLY", "").strip()


def _parse_only(s):
    triples = set()
    for group in s.split(";"):
        group = group.strip()
        if not group:
            continue
        parts = group.split(":")
        if len(parts) != 3:
            raise ValueError(f"bad MMD_ONLY group: {group!r}")
        ds, m, seeds = parts
        for sd in seeds.split(","):
            triples.add((ds.strip(), m.strip(), int(sd.strip())))
    return triples


ONLY_TRIPLES = _parse_only(ONLY) if ONLY else None


def _run_a(ds, m, seed, sub):
    """True if this (dataset, model, seed) triple should run under MMD_ONLY."""
    return ONLY_TRIPLES is None or (ds, m, seed) in ONLY_TRIPLES


# MMD_B_LOSS / MMD_B_DS: optional Experiment-B subsets, e.g.
#   MMD_B_LOSS="ce,focal"  -> only those losses
#   MMD_B_DS="dermamnist"  -> only that dataset
# Empty / unset = full 24-run matrix. Lets long Exp-B sessions be split into
# short survival-friendly halves after three silent Kaggle session deaths.
def _csv_env(name):
    v = os.environ.get(name, "").strip()
    return set(x.strip() for x in v.split(",") if x.strip()) or None


B_LOSSES = _csv_env("MMD_B_LOSS")
B_DATASETS = _csv_env("MMD_B_DS")


def _run_b(ds, loss):
    """True if this (dataset, loss) pair should run under the B filters."""
    if B_LOSSES is not None and loss not in B_LOSSES:
        return False
    if B_DATASETS is not None and ds not in B_DATASETS:
        return False
    return True


def exp_a():
    models = ["plannet", "gcvit-perceiver-lite", "resnet18", "vit-tiny"]
    ds_map = {"pneumoniamnist": ("pneumoniamnist", None),
              "dermamnist": ("dermamnist", None),
              "bloodmnist": ("bloodmnist", 12000)}
    datasets = [ds_map[d] for d in DS_LIST.split(",")]
    for ds, sub in datasets:
        for m in models:
            for seed in range(SEEDS):
                if not _run_a(ds, m, seed, sub):
                    print(f"SKIP exp A {ds} {m} seed={seed} (MMD_ONLY)",
                          flush=True)
                    continue
                if ("A", m, ds, "ce", seed) in SEEN:
                    print(f"SKIP exp A {ds} {m} seed={seed} (already logged)",
                          flush=True)
                    continue
                try:
                    res, _ = train_one(m, ds, "ce", epochs=EPOCHS, seed=seed,
                                       subsample=sub, batch=BATCH)
                except Exception as e:  # one bad run must not kill the matrix
                    print(f"ERROR exp A {ds} {m} seed={seed}: {type(e).__name__}: {e}",
                          flush=True)
                    log({"exp": "A", "model": m, "dataset": ds, "loss": "ce",
                         "seed": seed, "status": "error",
                         "error": f"{type(e).__name__}: {e}"})
                    continue
                rec = {"exp": "A"} | {k: v for k, v in res.items()
                                      if k != "model"}
                rec["model"] = m
                log(rec)


def exp_b():
    losses = ["ce", "focal", "cb-focal", "dacf"]
    datasets = [("dermamnist", None), ("bloodmnist", 12000)]
    for ds, sub in datasets:
        for loss in losses:
            if not _run_b(ds, loss):
                print(f"SKIP exp B {ds} {loss} (MMD_B filter)", flush=True)
                continue
            for seed in range(SEEDS):
                if ("B", "plannet", ds, loss, seed) in SEEN:
                    print(f"SKIP exp B {ds} {loss} seed={seed} (already logged)",
                          flush=True)
                    continue
                try:
                    res, _ = train_one("plannet", ds, loss, epochs=EPOCHS,
                                       seed=seed, subsample=sub, batch=BATCH)
                except Exception as e:  # one bad run must not kill the matrix
                    print(f"ERROR exp B {ds} {loss} seed={seed}: {type(e).__name__}: {e}",
                          flush=True)
                    log({"exp": "B", "model": "plannet", "dataset": ds,
                         "loss": loss, "seed": seed, "status": "error",
                         "error": f"{type(e).__name__}: {e}"})
                    continue
                rec = {"exp": "B"} | {k: v for k, v in res.items()
                                      if k != "model"}
                rec["model"] = "plannet"
                log(rec)


def exp_c():
    # Train plannet once per dataset (seed 0), save, then evaluate TTA variants
    # on test (clean) and shifted (noise) inputs.
    from tta import predict_single, predict_naive_tta, predict_cg_ttc
    import numpy as np
    datasets = [("pneumoniamnist", None, 2), ("dermamnist", None, 7)]
    for ds, sub, ncls in datasets:
        res, model = train_one("plannet", ds, "ce", epochs=12, seed=0,
                               subsample=sub)
        ckpt = os.path.join(CKPT, f"plannet_{ds}.pt")
        torch.save(model.state_dict(), ckpt)
        _, _, test_l, _ = get_loaders(ds, 256, sub, 0)
        x_all = torch.cat([x for x, _ in test_l])
        y_all = torch.cat([y for _, y in test_l]).numpy()

        def acc_of(probs):
            # TTA predictions are computed on xt (eval half); compare against
            # yt, not the full y_all. (Bug fix 2026-10-09: y_all caused a
            # (312,) vs (624,) broadcast ValueError that killed the Exp-C cell
            # before any record was logged.)
            return float((probs.argmax(1).numpy() == yt).mean())

        # calibrate tau on a val split (use first half of test as val proxy)
        n = len(x_all)
        xv, yv = x_all[:n // 2], y_all[:n // 2]
        xt, yt = x_all[n // 2:], y_all[n // 2:]
        best_tau, best_cov = 0.05, None
        for tau in [0.01, 0.03, 0.05, 0.08, 0.12]:
            _, ref, _ = predict_cg_ttc(model, xv, k=8, tau=tau)
            rate = float(ref.mean())
            if rate <= 0.15:
                best_tau = tau
        out = {"exp": "C", "dataset": ds, "tau": best_tau}
        for name, fn in [("single", lambda m, x: (predict_single(m, x), None)),
                         ("naive_tta", lambda m, x: (predict_naive_tta(m, x), None)),
                         ("cg_ttc", lambda m, x: predict_cg_ttc(m, x, k=8, tau=best_tau)[:2])]:
            probs, ref = fn(model, xt)
            out[name + "_acc"] = acc_of(probs)
            if ref is not None:
                out[name + "_referral"] = float(ref.mean())
        # shifted eval: gaussian noise sigma=0.1
        torch.manual_seed(0)
        xs = torch.clamp(xt + torch.randn_like(xt) * 0.1, 0, 1)
        for name, fn in [("single", lambda m, x: (predict_single(m, x), None)),
                         ("naive_tta", lambda m, x: (predict_naive_tta(m, x), None)),
                         ("cg_ttc", lambda m, x: predict_cg_ttc(m, x, k=8, tau=best_tau)[:2])]:
            probs, ref = fn(model, xs)
            acc = float((probs.argmax(1).numpy() == yt).mean())
            out[name + "_acc_shift"] = acc
            if ref is not None:
                out[name + "_referral_shift"] = float(ref.mean())
        log(out)


if __name__ == "__main__":
    which = sys.argv[1]
    torch.set_num_threads(2)
    {"A": exp_a, "B": exp_b, "C": exp_c}[which]()
