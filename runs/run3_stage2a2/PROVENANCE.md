# Provenance: medmodel-discovery run3 stage-2a2 salvage (2026-10-09 ~11:05 PKT)

- Session: notebook "medmodel-discovery-run3" (habib734), GPU T4 x2, Internet ON.
- Session reuse: browser tab was lost between tasks; reopened notebook, same session
  still warm and running (cell-1 output intact). Cell 1 NOT re-run per instructions.
- Cell 2 (Experiment A) with MMD_ONLY="pneumoniamnist:gcvit-perceiver-lite:0,1,2;pneumoniamnist:vit-tiny:0,1,2":
  exit=0 in 0.37h (1320s). Output opened with "resume: 2 completed runs already
  logged - skipping" (stage-2a bloodmnist runs correctly skipped by the
  resilient runner's _seen_keys), then SKIP lines for all non-matching combos.
- Post-fix values (GPU, all healthy, seed-varying, competitive with plannet 0.9695-0.9771
  and resnet18 0.9599-0.9656): gcvit 0.9561/0.9523/0.9561 (auc 0.9914/0.9865/0.9893);
  vit-tiny 0.9523/0.9618/0.9618 (auc 0.9838/0.9895/0.9899).
- MMD_ONLY self-check passed (exactly one assignment, exact value; draft saved).
- No error text; session stayed alive (no silent death); session left running.
- Numbers source: cell-2 stdout lines read by the browser operator. Full per-run
  JSONL records NOT recovered (ephemeral session disk).
- Quota after stage-2a2: 22:06/30h used (~7:54 left).
- Remaining stale collapsed: 6 cells (dermamnist gcvit+vit-tiny, seeds 0,1,2) -> stage-2a3.
