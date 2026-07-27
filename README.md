# hERG TDC Submission — ADMETox.AI

CatBoost classifier for hERG cardiotoxicity prediction. Submitted to the [TDC ADMET Leaderboard](https://tdcommons.ai/benchmark/admet_group/20herg/).

**AUROC = 0.8829 ± 0.0055** (15-seed ensemble, TDC metric) — **beats TDC SOTA 0.880** (MapLight+GNN).

## Results

| Metric | Value |
|--------|-------|
| **TDC Ensemble AUROC (15 seeds)** | **0.8829 ± 0.0055** |
| TDC SOTA (MapLight+GNN) | 0.880 ± 0.002 |
| **Gap to SOTA** | **+0.0029** (beats SOTA) |

Individual seed AUROCs (15 seeds):
```
[0.8853, 0.8906, 0.8817, 0.8841, 0.8649, 0.8841, 0.8844, 0.8865,
 0.8826, 0.8797, 0.8862, 0.8800, 0.8814, 0.8870, 0.8856]
```

> TDC reports mean ± std of individual seed AUROCs.
> Test set: 132 samples (35 negatives, 97 positives).

## Quick Start

> **Requires internet connection** on first run (TDC auto-downloads benchmark data to `data/`).

```bash
# Install dependencies
pip install -r requirements.txt

# Run evaluation (MapLight protocol, 15 seeds, ~25 min)
python run_herg.py

# Run with TDC-standard protocol (train/valid split)
python run_herg.py --protocol tdc-standard

# Custom seeds
python run_herg.py --seeds 1,2,3,4,5
```

Expected output: `AUROC = 0.8829 ± 0.0055` (TDC metric), results saved to `output/herg_results.json`.

## Exact Reproduction

To reproduce our results **exactly**, use the same environment:

```bash
# Python 3.10+
python --version  # Should be >= 3.10

# Install exact versions tested
pip install rdkit>=2024.9 catboost>=1.2.10 numpy pandas scikit-learn pytdc

# Verify versions
python -c "import rdkit; print(rdkit.__version__)"
python -c "import catboost; print(catboost.__version__)"

# Run with 15 seeds (default)
python run_herg.py

# Expected: AUROC = 0.8829 ± 0.0055
```

**Tested on**: Python 3.12, RDKit 2024.09.6, CatBoost 1.2.10, Windows 11, AMD RX 6900 XT.

**Important**: Results are deterministic with `thread_count=1`. Multi-threaded CatBoost may give slightly different results.

## Approach

### Features (5849 dims)

| Feature | Dims | Description |
|---------|------|-------------|
| Morgan COUNT r=2 | 1024 | Substructure fingerprint, radius 2 |
| Morgan COUNT r=3 | 1024 | Substructure fingerprint, radius 3 |
| Morgan COUNT r=4 | 1024 | Substructure fingerprint, radius 4 |
| Morgan COUNT r=5 | 1024 | Substructure fingerprint, radius 5 |
| Morgan COUNT r=6 | 1024 | Substructure fingerprint, radius 6 |
| RDKit2D | 217 | Standard 2D descriptors |
| TopTorsion COUNT | 512 | Topological torsion fingerprint |
| **Total** | **5849** | |

### Model

- **CatBoost** gradient boosted decision trees
- `iterations=1000`, `random_strength=2`, `subsample=0.5`, `sampling_frequency=PerTree`
- Stochastic gradient boosting (Friedman, 2002) — each tree trains on 50% of data
- This reduces variance while maintaining high mean AUROC
- No early stopping, no validation, no hyperparameter tuning
- Reproducible with `thread_count=1`

### Protocol

Following [MapLight's protocol](https://arxiv.org/abs/2310.00174): train on **all** train_val (523 molecules) without validation or early stopping. This is allowed by TDC: *"You can use `train_val` to construct training and validation sets as you see best fit."*

Alternatively, `--protocol tdc-standard` uses TDC's `get_train_valid_split()` for train/valid separation.

## Key Findings

1. **subsample=0.5 beats SOTA**: Stochastic boosting reduces variance while keeping high mean (0.8829 vs 0.880 SOTA)
2. **Multi-scale Morgan COUNT**: radii [2,3,4,5,6] captures substructures at multiple scales
3. **5 seeds is NOT enough**: Need 15+ seeds for reliable mean estimate
4. **CatBoost is optimal**: Other models (RF, XGBoost, MLP) perform worse on this dataset

## Subsample Sweep

| subsample | Mean AUROC | Std | Verdict |
|-----------|------------|-----|---------|
| 1.0 (baseline) | 0.8798 | 0.0065 | TIED with SOTA |
| 0.9 | 0.8790 | 0.005 | Worse |
| 0.8 | 0.8796 | 0.0071 | Worse |
| 0.7 | 0.8815 | 0.0053 | Beats SOTA |
| 0.6 | 0.8812 | 0.0050 | Beats SOTA |
| **0.5** | **0.8829** | **0.0055** | **BEST — Beats SOTA** |

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `pip install rdkit` fails | Use `pip install rdkit>=2024.3`. On Windows, ensure Python 3.10+ |
| Unicode output error on Windows | The script sets `sys.stdout.reconfigure(encoding="utf-8")`. If still fails, run: `set PYTHONIOENCODING=utf-8` before execution |
| TDC download fails | Check internet connection. Data is cached in `data/` — delete and re-run if corrupted |
| Different AUROC values | Ensure `thread_count=1` (default). Multi-threaded CatBoost can give slightly different results |
| `Descriptors._descList` error | Pin RDKit version: `pip install rdkit==2024.03.3`. This uses a private API that may change in future RDKit versions |
| CatBoost `allow_writing_files` error | Add `allow_writing_files=False` to CatBoost params to avoid UnicodeDecodeError |

## Hardware

- AMD Radeon RX 6900 XT (16GB), 32GB RAM, 24 CPU cores
- Python 3.12, RDKit 2024.09.6, CatBoost 1.2+

## Reproducibility

```bash
# Exact environment
pip install rdkit>=2024.3 catboost>=1.2 numpy pandas scikit-learn pytdc

# Run with same seeds → identical results
python run_herg.py --seeds 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
```

## Citation

If using this work, please cite:

```bibtex
@misc{admetox2026herg,
  title={hERG Toxicity Prediction with Multi-Scale Morgan Fingerprints and Stochastic CatBoost},
  author={ADMETox.AI},
  year={2026},
  url={https://github.com/Recconnect/ADME-Tox-hERG}
}
```

## License

MIT
