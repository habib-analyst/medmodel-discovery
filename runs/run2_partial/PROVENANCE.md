# medmodel-discovery run2 partial — PROVENANCE

Notebook: medmodel-discovery-run2 (Kaggle, habib734), launched ~17:00 PKT
2026-10-08, pinned to commit deaabb04 (transformer-collapse fix).

Session died silently mid-Experiment-A (~5h in; "Draft Session off" at
~22:10 PKT check). Cell 2 stopped abruptly after 3 of 7 filtered runs with
no exception/OOM/traceback. Experiments B and C never started. Cell 3
(summary) never executed; `results_full.jsonl` was never created.

Recovery method: read-only browser inspection of cell-2 stdout at
~22:10 PKT. Only per-run `log()` print lines were captured — val_acc and
val_auc only; full result records (ECE, per-class metrics, timing) are lost.
The 3 records in medmodel_run2_partial.jsonl are marked
"stdout-derived-provisional" and must NOT be merged into canonical aggregates
without the full records (pending run3 rerun with the resilient runner).

Still missing after run2 (queued for run3):
- Exp A: bloodmnist resnet18 seed 2; bloodmnist vit-tiny seeds 0,1,2
- Exp B: all 24 plannet-loss runs
- Exp C: full TTA experiment

Weekly GPU quota at the time of the failed run: 19:14/30h used
(10:46 remaining) as of ~22:10 PKT.
