# Provenance: medmodel-discovery run3 stage-2a salvage (2026-10-09 ~10:40 PKT)

- Session: notebook "medmodel-discovery-run3" (habib734), GPU T4 x2, Internet ON.
- Cell 1 (setup): ran alone, 84.5s; "tree OK: a9a4e30afa92 | resilience features confirmed", torch 2.11.0+cu128 cuda True.
- Cell 2 (Experiment A) with MMD_ONLY="bloodmnist:gcvit-perceiver-lite:0,1": exit=0 in 0.33h. Only the 2 target runs executed; all others printed "SKIP ... (MMD_ONLY)". No errors; session stayed alive (no silent death). Operator caught and fixed its own cell-edit duplication before running (final source verified clean; draft saved).
- Values: seed-VARYING (0.9182/0.9141), competitive with plannet — the norm_first+BatchNorm1d collapse fix is now confirmed on GPU for BOTH collapsed models (vit-tiny in stage-1, gcvit here).
- Numbers source: cell-2 stdout lines read by the browser operator. Full per-run JSONL records were NOT recovered (ephemeral session disk); session left running.
- Quota after stage-2a: 21:36/30h used (~8:24 left).
- This closes the bloodmnist portion of the stale-collapsed gap. Remaining: 12 cells (gcvit + vit-tiny on pneumoniamnist + dermamnist, seeds 0,1,2) -> stages 2a2/2a3.
