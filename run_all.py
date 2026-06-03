#!/usr/bin/env python3
"""
    python run_all.py                     # full run (all datasets + parameter sweep)
    python run_all.py --fast              # skip sweep, 2 seeds only (faster)
    python run_all.py --tests             # unit tests only
    python run_all.py --dataset skab      # skab only
    python run_all.py --dataset batadal   # batadal only
    python run_all.py --fast --dataset batadal
"""
import os, sys, subprocess, argparse

# dependency check / install
REQUIRED = [
    "tensorflow", "scikit-learn", "pandas", "numpy",
    "matplotlib", "seaborn", "scipy", "networkx", "Levenshtein",
]

def check_deps():
    missing = []
    for pkg in REQUIRED:
        try:
            __import__(pkg.lower().replace("-", "_"))
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"[setup] Installing: {missing}")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install",
             "--break-system-packages", "-q"] + missing
        )

check_deps()

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import CFG
from utils import log, save_json

parser = argparse.ArgumentParser(description="Run all experiments")
parser.add_argument("--fast",    action="store_true",
                    help="Reduced seeds/epochs for quick verification")
parser.add_argument("--tests",   action="store_true",
                    help="Run unit tests only and exit")
parser.add_argument("--dataset", choices=["skab", "batadal", "all"],
                    default="all")
args = parser.parse_args()

# speed override 
if args.fast:
    CFG.dl.epochs          = 15
    CFG.dl.sequence_length = 20
    CFG.experiment.seeds   = [42, 123]
    print("[fast mode] epochs=15  seq_len=20  seeds=[42,123]")

os.makedirs(CFG.results_dir, exist_ok=True)
os.makedirs(CFG.plots_dir,   exist_ok=True)

# unit tests (always run) 
print("\n" + "="*60)
print("  UNIT TESTS")
print("="*60)
import unittest
loader  = unittest.TestLoader()
suite   = loader.discover("tests", pattern="test_*.py")
result  = unittest.TextTestRunner(verbosity=2).run(suite)

if args.tests:
    sys.exit(0 if result.wasSuccessful() else 1)

if not result.wasSuccessful():
    log.warning("Some unit tests failed — experiments will still run.")

# experiments 
all_results = {}
do_sweep    = not args.fast   # skip sweep in fast mode

if args.dataset in ("batadal", "all"):
    print("\n" + "="*60)
    print("  BATADAL EXPERIMENT")
    print("="*60)
    from experiments.batadal_exp import run_batadal
    batadal_res = run_batadal(sweep_params=do_sweep)
    all_results["BATADAL"] = batadal_res

    print("\n--- BATADAL results (original scenario) ---")
    for k, v in sorted(batadal_res.items()):
        if "original" in k:
            f1  = v.get("f1",  {})
            acc = v.get("accuracy", {})
            print(f"  {k:<30} F1={f1.get('mean',0):.3f}±{f1.get('std',0):.3f}"
                  f"  Acc={acc.get('mean',0):.3f}")

if args.dataset in ("skab", "all"):
    print("\n" + "="*60)
    print("  SKAB EXPERIMENT")
    print("="*60)
    from experiments.skab_exp import run_skab
    skab_res = run_skab(sweep_params=do_sweep)
    all_results["SKAB"] = skab_res

    print("\n--- SKAB results (original scenario) ---")
    for k, v in sorted(skab_res.items()):
        if "original" in k:
            f1  = v.get("f1",  {})
            acc = v.get("accuracy", {})
            print(f"  {k:<30} F1={f1.get('mean',0):.3f}±{f1.get('std',0):.3f}"
                  f"  Acc={acc.get('mean',0):.3f}")

# cross-dataset summary
if len(all_results) == 2:
    print("\n" + "="*60)
    print("  CROSS-DATASET COMPARISON (F1)")
    print("="*60)
    print(f"  {'Model':<30} {'SKAB':>8} {'BATADAL':>10}")
    print("  " + "-"*50)
    for mk in ["LSTM_original", "GRU_original", "Automata_original"]:
        sk = all_results["SKAB"].get(mk, {}).get("f1", {}).get("mean", "N/A")
        bt = all_results["BATADAL"].get(mk, {}).get("f1", {}).get("mean", "N/A")
        skf = f"{sk:.3f}" if isinstance(sk, float) else sk
        btf = f"{bt:.3f}" if isinstance(bt, float) else bt
        print(f"  {mk:<30} {skf:>8} {btf:>10}")

save_json(all_results, os.path.join(CFG.results_dir, "full_results.json"))
print(f"\n✓  Results  → {CFG.results_dir}/")
print(f"✓  Plots    → {CFG.plots_dir}/")
print(f"✓  Complete.\n")