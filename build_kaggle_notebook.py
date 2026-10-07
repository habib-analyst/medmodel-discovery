"""Build medmnist_28_medmodel_discovery.ipynb for the Kaggle GPU full run.

The notebook clones the pushed repo (parent creates it via browser, worker
pushes code), downloads the 3 npz datasets, and runs experiments A/B/C.
Results -> /kaggle/working/results_full.jsonl (download after the run).
"""
import json

REPO = "https://github.com/habib-analyst/medmodel-discovery"

cell_setup = """# Cell 1 — setup: deps, repo, data
import subprocess, sys, os
# torch is preinstalled on the Kaggle GPU image
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "medmnist==3.0.2", "scikit-learn"], check=True)
os.chdir("/kaggle/working")
if not os.path.exists("/kaggle/working/mmd"):
    subprocess.run(["git", "clone", "REPO_URL", "/kaggle/working/mmd"], check=True)
os.makedirs("/kaggle/working/medmnist", exist_ok=True)
BASE = "https://zenodo.org/records/10519652/files"
for d in ["pneumoniamnist", "dermamnist", "bloodmnist"]:
    p = f"/kaggle/working/medmnist/{d}.npz"
    if not os.path.exists(p):
        print("downloading", d, flush=True)
        subprocess.run(["curl", "-sL", f"{BASE}/{d}.npz?download=1", "-o", p], check=True)
import torch
print("torch", torch.__version__, "| cuda:", torch.cuda.is_available(), flush=True)
print("setup done", flush=True)
""".replace("REPO_URL", REPO)

cell_run = """# Cell 2 — full experiment matrix (A: architectures, B: losses, C: TTA)
import subprocess, sys, os, time
os.chdir("/kaggle/working/mmd")
os.environ["MMD_EPOCHS"] = "12"
os.environ["MMD_SEEDS"] = "3"
os.environ["MMD_BATCH"] = "256"
os.environ["MMD_DS"] = "pneumoniamnist,dermamnist,bloodmnist"
os.environ["MMD_ROOT"] = "/kaggle/working/medmnist"
# results.jsonl lands in /kaggle/working/mmd/; copy to working root for download
for exp in ["A", "B", "C"]:
    print(f"===== Experiment {exp} =====", flush=True)
    t0 = time.time()
    subprocess.run([sys.executable, "run_experiments.py", exp], check=True)
    print(f"exp {exp} done in {(time.time()-t0)/3600:.2f}h", flush=True)
subprocess.run(["cp", "/kaggle/working/mmd/results.jsonl",
                "/kaggle/working/results_full.jsonl"], check=True)
print("ALL DONE — download /kaggle/working/results_full.jsonl", flush=True)
"""

cell_summary = """# Cell 3 — quick summary table
import json, collections
rows = [json.loads(l) for l in open("/kaggle/working/results_full.jsonl")]
print(f"{len(rows)} records")
for r in rows:
    if r.get("exp") == "C":
        print("C", r["dataset"], {k: round(v, 4) for k, v in r.items() if k.endswith("_acc")})
    else:
        v = r["val"]
        print(r["exp"], r["model"], r["dataset"], r["loss"], "seed", r["seed"],
              f"acc={v['acc']:.4f} auc={v['auc']:.4f} ece={v['ece']:.4f}")
"""

def md_cell(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(True)}

def code_cell(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": src.splitlines(True)}

nb = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                 "accelerator": "GPU"},
    "cells": [
        md_cell("# medmodel-discovery — full GPU run (Project #9)\n"
                "Requires: GPU T4 accelerator ON, Internet ON.\n"
                "Runs experiments A (4 archs × 3 datasets × 3 seeds), "
                "B (4 losses × 2 datasets × 3 seeds), C (TTA strategies). "
                "Est. ~3–4 T4-hours. Results → `/kaggle/working/results_full.jsonl`."),
        code_cell(cell_setup),
        code_cell(cell_run),
        code_cell(cell_summary),
    ],
}

out = "/home/hatch/workspace/goals/github-portfolio-linkedin-presence-buildout/hidden_files/medmodel_discovery_kaggle.ipynb"
with open(out, "w") as f:
    json.dump(nb, f)
print("wrote", out)
