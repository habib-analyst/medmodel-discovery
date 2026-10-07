"""Analyze results.jsonl -> summary tables (printed + summary.json)."""
import json
import os
from collections import defaultdict

P = os.path.join(os.path.dirname(__file__), "results.jsonl")


def load():
    rows = []
    with open(P) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def summarize_a(rows):
    print("=== Experiment A: architectures (val acc / auc, mean over seeds) ===")
    groups = defaultdict(list)
    for r in rows:
        if r.get("exp") == "A":
            groups[(r["model"], r["dataset"])].append(r)
    for (m, d), rs in sorted(groups.items()):
        accs = [r["val"]["acc"] for r in rs]
        aucs = [r["val"]["auc"] for r in rs]
        par = rs[0]["params"]
        print(f"{m:22s} {d:16s} acc={sum(accs)/len(accs):.4f} "
              f"auc={sum(aucs)/len(aucs):.4f} params={par} (n={len(rs)})")


def summarize_b(rows):
    print("=== Experiment B: losses on plannet (val acc / ece / minority recall) ===")
    groups = defaultdict(list)
    for r in rows:
        if r.get("exp") == "B":
            groups[(r["loss"], r["dataset"])].append(r)
    for (l, d), rs in sorted(groups.items()):
        accs = [r["val"]["acc"] for r in rs]
        eces = [r["val"]["ece"] for r in rs]
        minrec = [min(x for x in r["val"]["recalls"] if x == x) for r in rs]
        print(f"{l:10s} {d:16s} acc={sum(accs)/len(accs):.4f} "
              f"ece={sum(eces)/len(eces):.4f} "
              f"min_recall={sum(minrec)/len(minrec):.4f} (n={len(rs)})")


def summarize_c(rows):
    print("=== Experiment C: TTA strategies ===")
    for r in rows:
        if r.get("exp") == "C":
            print(f"{r['dataset']}: tau={r['tau']}")
            for k in ["single", "naive_tta", "cg_ttc"]:
                print(f"  {k:10s} acc={r[k+'_acc']:.4f} "
                      f"acc_shift={r[k+'_acc_shift']:.4f}", end="")
                if k == "cg_ttc":
                    print(f" referral={r[k+'_referral']:.3f} "
                          f"referral_shift={r[k+'_referral_shift']:.3f}", end="")
                print()


if __name__ == "__main__":
    rows = load()
    print(f"{len(rows)} result rows")
    summarize_a(rows)
    summarize_b(rows)
    summarize_c(rows)
