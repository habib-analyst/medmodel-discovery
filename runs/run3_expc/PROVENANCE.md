# Provenance: medmodel-discovery run3 Exp-C salvage (2026-10-09 ~17:05 PKT)

- Session: notebook "medmodel-discovery-run3" (habib734), GPU T4 x2, Internet ON,
  still live at handback. Setup cell re-pinned to d05b1cde2f2a39d79fb251002b4b7383ae284fc3
  (acc_of==yt fix verified in tree: "run_experiments.py acc_of==yt hits: 1",
  "test_smoke.py test_exp_c_acc_of_uses_yt hits: 1"); datasets present;
  torch 2.11.0+cu128 cuda True.
- Experiment C cell: exit=0 in 0.08h (298.66s). Summary cell: "2 records",
  {('C', 'ok'): 2}. No stop conditions; no silent death.
- This file (medmodel_run3_expc.jsonl) is a verbatim copy of the downloaded
  /kaggle/working/results_full.jsonl from that session — COMPLETE JSONL records
  (full precision), not stdout-derived. Source file:
  /home/hatch/workspace/browser_downloads/sess-4257000343/browser-download-20261009T120224.241586458Z-0-results_full.jsonl
- MEASURED:
  - pneumoniamnist (tau=0.05): single_acc 0.8654, naive_tta_acc 0.8462,
    cg_ttc_acc 0.8654, cg_ttc_referral 0.4263; shifted (gaussian noise 0.1):
    single 0.4936, naive_tta 0.6058, cg_ttc 0.5064, referral 0.7853.
  - dermamnist (tau=0.05): single_acc 0.7418, naive_tta_acc 0.6610,
    cg_ttc_acc 0.7198, cg_ttc_referral 0.6889; shifted: single 0.5693,
    naive_tta 0.5563, cg_ttc 0.5613, referral 0.6999.
- Notes for analysis: (a) referral rates 0.43/0.69 far exceed the tau-calibration
  target (<=0.15 on the val half) — calibration does not transfer val->eval half;
  (b) on clean data cg_ttc >= naive_tta but <= single; (c) under noise shift,
  naive_tta BEATS cg_ttc on pneumonia (0.6058 vs 0.5064); (d) risk-coverage AUC
  (the stated C success metric) was NOT recorded — only acc + referral;
  analysis must not claim risk-coverage AUC from these numbers.
- Quota after Exp-C: 25:39/30h used (~4:21 left).
- All three experiments now have salvaged data: Exp-A 36/36 canonical post-fix
  (medmodel_expA_canonical.jsonl), Exp-B 24/24 (half1+half2 jsonl, stdout-derived
  val_acc/val_auc only), Exp-C 2/2 (complete records, this file).
