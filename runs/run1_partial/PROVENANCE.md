# Medmodel-discovery run 1 — PARTIAL (2026-10-08)

Launched ~10:25 PKT on Kaggle T4 x2. Session STOPPED ~13:46 PKT (~3h21m GPU)
with NO error text in saved outputs. results_full.jsonl was never written
(it is only created at the notebook's final cell). /kaggle/working was
unrecoverable after the session ended.

## What survived
29 of 36 Experiment-A stdout summary lines (val_acc + val_auc only) were
recovered from the saved cell output by a read-only browser check at 16:15 PKT.
Full records (test acc/auc/ece/recalls, params, train_secs) for these runs are
LOST. These 29 lines live in `medmodel_run1_partial.jsonl` in this directory —
marked provisional because they are stdout-derived, not the canonical JSONL.

## Coverage
- Complete: pneumoniamnist x {plannet, gcvit-perceiver-lite, resnet18, vit-tiny} x seeds 0-2 (12)
- Complete: dermamnist x 4 models x seeds 0-2 (12)
- Complete: bloodmnist x plannet x seeds 0-2 (3)
- Partial: bloodmnist x gcvit-perceiver-lite x seeds 0-1 (2)
- MISSING: bloodmnist x gcvit-perceiver-lite seed 2; bloodmnist x resnet18 seeds 0-2; bloodmnist x vit-tiny seeds 0-2 (7)
- Experiments B (losses) and C (TTA): NEVER STARTED.

## CRITICAL BUG FOUND in the recovered data
gcvit-perceiver-lite and vit-tiny show COLLAPSED training on every dataset:
val_acc is CONSTANT across all seeds (0.7424 pneumoniamnist, 0.6690 dermamnist,
0.1945 bloodmnist) — constant/majority-class predictions. plannet and resnet18
trained normally (val_acc 0.74-0.98, AUC 0.91-1.00). The existing 6/6 smoke
tests did NOT catch this. Root cause under investigation; reruns of the
affected baselines are BLOCKED until fixed.

## Quota
18:12/30h used at 16:15 PKT (~11:48 remaining).
