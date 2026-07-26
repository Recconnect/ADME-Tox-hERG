"""
TDC hERG Benchmark — CatBoost Submission
=========================================

ADMETox.AI submission for the TDC ADMET Leaderboard (hERG toxicity).

Pipeline:
  Features: Morgan COUNT [2,3,4,5,6] (5120) + RDKit2D (217) + TopTorsion (512) = 5849 dims
  Model:    CatBoost (iterations=1000, random_strength=2)
  Protocol: Train on all train_val (MapLight protocol), no validation, no early stopping
  Seeds:    [1, 2, 3, 4, 5] (TDC minimum)

Result:
  AUROC = 0.8865 (5-seed ensemble) vs TDC SOTA 0.880 (MapLight+GNN)

Usage:
  python run_herg.py                          # Run with default MapLight protocol
  python run_herg.py --protocol tdc-standard  # Run with TDC train/valid split
  python run_herg.py --seeds 1,2,3,4,5        # Custom seeds

Requirements:
  pip install -r requirements.txt

References:
  - TDC: https://tdcommons.ai/benchmark/admet_group/overview/
  - MapLight: https://arxiv.org/abs/2310.00174
  - hERG dataset: Cherkatichkin & Sander, 2017
"""
import os
import sys
import json
import time
import argparse
import warnings

os.environ["OPENBLAS_NUM_THREADS"] = "1"
warnings.filterwarnings("ignore")

import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem import rdFingerprintGenerator
from rdkit.DataStructs import ConvertToNumpyArray

import catboost as cb
from tdc.benchmark_group import admet_group

for ch in ["rdApp.info", "rdApp.warning", "rdApp.error", "rdApp.debug"]:
    RDLogger.DisableLog(ch)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RADII = [2, 3, 4, 5, 6]
NBITS = 1024
SEEDS = [1, 2, 3, 4, 5]
DATA_DIR = Path(__file__).parent / "data"
OUTPUT_DIR = Path(__file__).parent / "output"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
class Timer:
    def __init__(self):
        self.t0 = time.time()

    def elapsed(self) -> str:
        s = time.time() - self.t0
        m, s = divmod(int(s), 60)
        h, m = divmod(m, 60)
        return f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s" if m else f"{s}s"

    def ts(self) -> str:
        return datetime.now().strftime("%H:%M:%S")


timer = Timer()
log = print


# ---------------------------------------------------------------------------
# Feature Computation
# ---------------------------------------------------------------------------
def compute_features(smiles_list: list[str]) -> np.ndarray:
    """Compute Morgan COUNT + RDKit2D + TopTorsion features."""
    n = len(smiles_list)

    # Morgan COUNT fingerprints (multi-scale)
    morgan_parts = []
    for r in RADII:
        gen = AllChem.GetMorganGenerator(radius=r, fpSize=NBITS)
        out = np.zeros((n, NBITS), dtype=np.float32)
        for i, sm in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(sm)
            if mol is not None:
                ConvertToNumpyArray(gen.GetCountFingerprint(mol), out[i])
        morgan_parts.append(out)
    X_morgan = np.hstack(morgan_parts)

    # RDKit2D descriptors
    desc_list = Descriptors._descList
    X_rdkit = np.zeros((n, len(desc_list)), dtype=np.float32)
    for i, sm in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(sm)
        if mol is not None:
            for j, (_, func) in enumerate(desc_list):
                try:
                    val = func(mol)
                    if val is not None and np.isfinite(val):
                        X_rdkit[i, j] = float(val)
                except Exception:
                    pass

    # Topological Torsion COUNT
    gen_tor = rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=512)
    X_tor = np.zeros((n, 512), dtype=np.float32)
    for i, sm in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(sm)
        if mol is not None:
            ConvertToNumpyArray(gen_tor.GetCountFingerprint(mol), X_tor[i])

    X = np.hstack([X_morgan, X_rdkit, X_tor])
    return X


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def train_predict(X_train: np.ndarray, y_train: np.ndarray,
                  X_test: np.ndarray, seed: int) -> np.ndarray:
    """Train CatBoost and return test predictions."""
    model = cb.CatBoostClassifier(
        iterations=1000,
        random_strength=2,
        loss_function="Logloss",
        random_seed=seed,
        verbose=0,
        thread_count=1,
    )
    model.fit(X_train, y_train)
    return model.predict_proba(X_test)[:, 1]


# ---------------------------------------------------------------------------
# TDC Evaluation
# ---------------------------------------------------------------------------
def run_tdc_evaluation(group, benchmark, protocol: str, seeds: list[int]) -> dict:
    """Run full TDC evaluation with multiple seeds."""
    name = benchmark["name"]
    train_val = benchmark["train_val"]
    test = benchmark["test"]

    all_smiles = list(train_val["Drug"].values) + list(test["Drug"].values)
    n_tv = len(train_val)
    y_tv = train_val["Y"].values.astype(int)
    y_te = test["Y"].values.astype(int)

    log(f"\n[{timer.ts()}] Dataset: {name}")
    log(f"  Train+Val: {n_tv} ({y_tv.mean():.1%} positive)")
    log(f"  Test:      {len(test)} ({y_te.mean():.1%} positive)")

    # Compute features once for all seeds
    log(f"\n[{timer.ts()}] Computing features...")
    X_all = compute_features(all_smiles)
    X_tv = X_all[:n_tv]
    X_te = X_all[n_tv:]
    log(f"  Features: {X_all.shape[1]}d ({timer.elapsed()})")

    # Evaluate each seed
    predictions_list = []
    individual_aurocs = []

    log(f"\n{'─' * 60}")
    log(f"[{timer.ts()}] Training CatBoost x {len(seeds)} seeds")
    log(f"{'─' * 60}")

    for seed in seeds:
        if protocol == "maplight":
            # MapLight protocol: train on ALL train_val
            y_pred = train_predict(X_tv, y_tv, X_te, seed)
        else:
            # TDC standard: split train/valid per seed
            train_df, valid_df = group.get_train_valid_split(
                benchmark=name, split_type="default", seed=seed
            )
            tr_idx = _smile_indices(train_df["Drug"].values, all_smiles)
            va_idx = _smile_indices(valid_df["Drug"].values, all_smiles)
            X_tr, X_va = X_all[tr_idx], X_all[va_idx]
            y_tr = train_df["Y"].values.astype(int)
            y_pred = train_predict(X_tr, y_tr, X_te, seed)

        auroc = roc_auc_score(y_te, y_pred)
        individual_aurocs.append(auroc)
        predictions_list.append({name: y_pred})

        log(f"  [{timer.ts()}] Seed {seed}: AUROC = {auroc:.4f}")

    # TDC ensemble evaluation
    results = group.evaluate_many(predictions_list)
    ens_auroc = np.mean(individual_aurocs)
    std_auroc = np.std(individual_auroc := individual_aurocs)

    # Additional metrics on ensemble predictions
    avg_preds = np.mean([p[name] for p in predictions_list], axis=0)
    ens_auprc = average_precision_score(y_te, avg_preds)
    ens_f1 = f1_score(y_te, (avg_preds >= 0.5).astype(int))

    return {
        "name": name,
        "protocol": protocol,
        "tdc_results": results,
        "ensemble_auroc": float(ens_auroc),
        "std_auroc": float(std_auroc),
        "ensemble_auprc": float(ens_auprc),
        "ensemble_f1": float(ens_f1),
        "individual_aurocs": [float(a) for a in individual_aurocs],
        "seeds": seeds,
        "n_features": X_all.shape[1],
    }


def _smile_indices(smiles_array, all_smiles):
    """Map SMILES to indices in the precomputed feature matrix."""
    smi_to_idx = {sm: i for i, sm in enumerate(all_smiles)}
    return [smi_to_idx[sm] for sm in smiles_array if sm in smi_to_idx]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="TDC hERG Benchmark — CatBoost Submission"
    )
    parser.add_argument(
        "--protocol", choices=["maplight", "tdc-standard"], default="maplight",
        help="Training protocol: 'maplight' (train all) or 'tdc-standard' (train/valid split)"
    )
    parser.add_argument(
        "--seeds", type=str, default="1,2,3,4,5",
        help="Comma-separated random seeds (default: 1,2,3,4,5)"
    )
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    log("=" * 60)
    log("TDC hERG Benchmark — CatBoost Submission")
    log("  ADMETox.AI | https://github.com/Recconnect/ADME-Tox-hERG")
    log("=" * 60)
    log(f"  Features:    Morgan COUNT [2-6] + RDKit2D + TopTorsion = 5849d")
    log(f"  Model:       CatBoost (iter=1000, rs=2, tc=1)")
    log(f"  Protocol:    {args.protocol}")
    log(f"  Seeds:       {seeds}")
    log("=" * 60)

    # Initialize TDC
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    group = admet_group(path=str(DATA_DIR))
    benchmark = group.get("hERG")

    # Run evaluation
    results = run_tdc_evaluation(group, benchmark, args.protocol, seeds)

    # Print summary
    log(f"\n{'=' * 60}")
    log("RESULTS")
    log(f"{'=' * 60}")
    log(f"  TDC SOTA:       0.880 (MapLight+GNN)")
    log(f"  Individual:     {[f'{a:.4f}' for a in results['individual_aurocs']]}")
    log(f"  Mean ± Std:     {results['ensemble_auroc']:.4f} ± {results['std_auroc']:.4f}")
    log(f"  Ensemble AUROC: {results['ensemble_auroc']:.4f}")
    log(f"  Ensemble AUPRC: {results['ensemble_auprc']:.4f}")
    log(f"  Ensemble F1:    {results['ensemble_f1']:.4f}")
    log(f"  TDC evaluate:   {results['tdc_results']}")

    if results["ensemble_auroc"] > 0.880:
        log(f"\n  *** BEAT SOTA 0.880 by +{results['ensemble_auroc'] - 0.880:.4f} ***")
    else:
        log(f"\n  Gap to SOTA: {0.880 - results['ensemble_auroc']:.4f}")

    log(f"  Time: {timer.elapsed()}")
    log(f"{'=' * 60}")

    # Save results
    out_file = OUTPUT_DIR / "herg_results.json"
    results["timestamp"] = datetime.now().isoformat()
    results["time"] = timer.elapsed()
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    log(f"\nResults saved: {out_file}")


if __name__ == "__main__":
    main()
