# Provenance: medmodel-discovery run3 stage-1 salvage (2026-10-09 ~04:45 PKT)

- Session: notebook "medmodel-discovery-run3" (habib734), GPU T4 x2, Internet ON.
- Cell 1 (setup): completed OK, tree OK a9a4e30afa92, torch 2.11.0+cu128 cuda True.
- Cell 2 (Experiment A remainder, MMD_ONLY="bloodmnist:resnet18:2;bloodmnist:vit-tiny:0,1,2"):
  exit=0 in 0.53h (~31.7 min). All 32 non-targeted combos SKIPped; 4 targeted runs completed.
- Source of the numbers: cell-2 stdout lines "val_acc=... val_auc=..." read by the
  browser operator (no error lines observed). Full per-run JSONL records were NOT
  recovered (results_full.jsonl lives in the session's ephemeral /kaggle/working;
  session left running for stage 2).
- These 4 runs complete Experiment A: 29 (run1_partial) + 3 (run2_partial) + 4 = 36/36.
- Quota after stage 1: 20:30/30h used (~9:30 left).
- vit-tiny values are seed-varying (0.9235–0.9282) and competitive with plannet/resnet18:
  the norm_first+BatchNorm1d collapse fix is confirmed on GPU for vit-tiny as well.
- KNOWN GAP: gcvit-perceiver-lite/bloodmnist/seeds 0,1 are still the PRE-FIX collapsed
  values (0.1945) from run1 — they were never rerun post-fix. Stage 2 reruns them.
