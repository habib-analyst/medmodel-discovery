"""Full-scale GPU run for Project #9 medmodel-discovery (Kaggle T4).

Run on Kaggle with GPU enabled. Downloads MedMNIST npz files, installs deps,
runs Experiment A (architectures x datasets, 12 epochs, 3 seeds),
Experiment B (losses), Experiment C (TTA). Results -> results_full.jsonl,
uploaded as a Kaggle dataset output or downloaded.

Usage on Kaggle: paste into notebook cells (one section per cell) or upload
this file and run: !python kaggle_full_run.py
"""
import os, sys, json, subprocess, time

# ---- cell 1: setup ----
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "medmnist==3.0.2", "scikit-learn"], check=True)
os.makedirs("/kaggle/working/medmnist", exist_ok=True)
BASE = "https://zenodo.org/records/10519652/files"
for d in ["pneumoniamnist", "dermamnist", "bloodmnist"]:
    p = f"/kaggle/working/medmnist/{d}.npz"
    if not os.path.exists(p):
        print("downloading", d, flush=True)
        subprocess.run(["curl", "-sL", f"{BASE}/{d}.npz?download=1", "-o", p],
                       check=True)
print("data ready", flush=True)

# ---- cell 2: run (paste the repo's models.py, losses.py, tta.py, data.py,
# train.py, run_experiments.py into /kaggle/working/mmd/ first, or clone:
#   !git clone https://github.com/habib-analyst/medmodel-discovery /kaggle/working/mmd
# then:)
os.chdir("/kaggle/working/mmd")
os.environ["MMD_EPOCHS"] = "12"
os.environ["MMD_SEEDS"] = "3"
os.environ["MMD_BATCH"] = "256"
os.environ["MMD_DS"] = "pneumoniamnist,dermamnist,bloodmnist"
# data.py reads ROOT=/home/hatch/.medmnist -> override via env
os.environ["MMD_ROOT"] = "/kaggle/working/medmnist"
for exp in ["A", "B", "C"]:
    print(f"=== Experiment {exp} ===", flush=True)
    t0 = time.time()
    subprocess.run([sys.executable, "run_experiments.py", exp], check=True)
    print(f"exp {exp} done in {(time.time()-t0)/3600:.2f}h", flush=True)
print("ALL DONE")
