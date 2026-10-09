# Provenance: medmodel-discovery run3 Exp-B half2 salvage (2026-10-09 ~16:10 PKT)

- Session: notebook "medmodel-discovery-run3" (habib734), GPU T4 x2, Internet ON.
  Read-only browser check (no cells run); session is now OFF.
- Cell 3 (Experiment B) with MMD_B_LOSS="cb-focal,dacf", MMD_ROOT="/kaggle/working/medmnist":
  exit=0 in 0.58h (2088s). All 12 plannet runs completed, no errors.
  Header: "resume: 12 completed runs already logged — skipping" (half1 cells);
  4 MMD_B filter SKIPs for ce/focal.
- Numbers source: cell-3 stdout lines read by the browser operator. Full per-run
  JSONL records NOT recovered (ephemeral session disk gone with the OFF session).
  stdout carried only val_acc/val_auc — ECE and per-class recalls for Exp-B are
  UNMEASURED (they exist only in the lost results_full.jsonl records).
- HEADLINE: cb-focal and DACF both collapsed on dermamnist (~0.55-0.59 val_acc
  vs ce 0.7428-0.7567 / focal 0.7478-0.7507) while AUC stayed ~0.876-0.882
  (model still ranks; decision boundary distorted). On bloodmnist they stayed
  competitive (cb-focal 0.9211-0.9410, dacf 0.9264-0.9416 vs ce 0.9404-0.9480).
  Working hypothesis (not a claim): the class-balanced effective-number term,
  shared by cb-focal and DACF but not plain focal, over-amplifies rare classes
  on the hard 7-class derma task. Loss-code review queued for the analysis worker.
- ANOMALY noted (read-only): cell 2's persisted output no longer shows the
  "exp A cell exit=0" line (it did at ~13:05 PKT) — kernel died before the final
  print or output tail was lost on reconnect. The 6 A-runs' val lines are
  unchanged and match runs/run3_stage2a3/ salvage; canonical Exp-A unaffected.
- Quota after half2: 25:19/30h used (~4:41 left).
- Cells 4 (Exp-C TTA) and 5 (summary) never executed.
