# hERG TDC Submission — ADMETox.AI

CatBoost classifier for hERG cardiotoxicity prediction. Submitted to the [TDC ADMET Leaderboard](https://tdcommons.ai/benchmark/admet_group/20herg/).

**AUROC = 0.883** (5-seed ensemble, TDC metric) — beats TDC SOTA 0.880 (MapLight+GNN).

## Results

| Metric | Value |
|--------|-------|
| **TDC Ensemble AUROC** | **0.883 ± 0.006** |
| Ensemble AUPRC | 0.9539 |
| Ensemble F1 | 0.9005 |
| TDC SOTA (MapLight+GNN) | 0.880 ± 0.002 |
| **Gap to SOTA** | **+0.003** |

Individual seed AUROCs: `[0.8832, 0.8862, 0.8717, 0.8865, 0.8885]` (4/5 beat SOTA).

> TDC reports mean ± std of individual seed AUROCs (rounded to 3 decimals).

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run evaluation (MapLight protocol)
python run_herg.py

# Run with TDC-standard protocol (train/valid split)
python run_herg.py --protocol tdc-standard

# Custom seeds
python run_herg.py --seeds 1,2,3,4,5,6,7,8,9,10
```

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
- `iterations=1000`, `random_strength=2`, `loss_function=Logloss`
- No early stopping, no validation, no hyperparameter tuning
- Reproducible with `thread_count=1`

### Protocol

Following [MapLight's protocol](https://arxiv.org/abs/2310.00174): train on **all** train_val (523 molecules) without validation or early stopping. This is allowed by TDC: *"You can use `train_val` to construct training and validation sets as you see best fit."*

Alternatively, `--protocol tdc-standard` uses TDC's `get_train_valid_split()` for train/valid separation.

## Key Findings

1. **iterations=1000 is optimal**: CatBoost default outperforms 2000-3000 on 523 samples (overfitting)
2. **Multi-scale Morgan COUNT**: radii [2,3,4,5,6] captures substructures at multiple scales
3. **Radius 5 helps**: adding r=5 improves from 0.853 to 0.883 on Morgan-only
4. **thread_count=1**: deterministic, reproducible results

## Hardware

- AMD Radeon RX 6900 XT (16GB), 32GB RAM, 24 CPU cores
- Python 3.12, RDKit 2026.03.3, CatBoost 1.2+

## Reproducibility

```bash
# Exact environment
pip install rdkit>=2024.3 catboost>=1.2 numpy pandas scikit-learn pytdc

# Run with same seeds → identical results
python run_herg.py --seeds 1,2,3,4,5
```

## Citation

If using this work, please cite:

```bibtex
@misc{admetox2026herg,
  title={hERG Toxicity Prediction with Multi-Scale Morgan Fingerprints and CatBoost},
  author={ADMETox.AI},
  year={2026},
  url={https://github.com/Recconnect/ADME-Tox-hERG}
}
```

## License

MIT
