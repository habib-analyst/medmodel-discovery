# Provenance: medmodel-discovery run3 stage-2a3 salvage (2026-10-09 ~11:40 PKT)

- Session: notebook "medmodel-discovery-run3" (habib734), GPU T4 x2, Internet ON.
- Session reuse: browser tab lost between tasks again; reopened, same session warm
  (58m at reconnect, cell-1 output intact). Cell 1 NOT re-run.
- Cell 2 (Experiment A) with MMD_ONLY="dermamnist:gcvit-perceiver-lite:0,1,2;dermamnist:vit-tiny:0,1,2":
  exit=0 in 0.54h (1937s). Output opened with "resume: 8 completed runs already
  logged - skipping" (all earlier stage runs correctly skipped). SKIP lines for
  non-matching combos. MMD_ONLY self-check passed (single assignment, exact value).
- Post-fix values (GPU, healthy, seed-varying): gcvit 0.7139/0.7218/0.7408
  (auc 0.8983/0.9047/0.9168); vit-tiny 0.7268/0.7178/0.7208 (auc 0.9031/0.9001/0.8972).
  DermaMNIST is the hardest of the 3 datasets; these sit slightly below the
  healthy plannet (0.7418-0.7567) / resnet18 (0.7637-0.7747) ranges on derma -
  plausible post-fix behavior, no collapse signature (values vary by seed).
- No error text; session stayed alive (no silent death); session left running.
- Numbers source: cell-2 stdout read by the browser operator. Full per-run JSONL
  records NOT recovered (ephemeral session disk).
- Quota after stage-2a3: 22:40/30h used (~7:20 left).
- **EXPERIMENT A IS NOW FULLY POST-FIX: 36/36 canonical cells.** No stale
  collapsed cells remain. Next: Exp-B half1 (MMD_B_LOSS="ce,focal") on the warm
  session with the pinned commit updated to 26d33ff (B-filter code).
