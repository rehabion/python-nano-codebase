# nanotox — Nanoparticle Cytotoxicity ML Benchmark

A clean, reproducible Python pipeline for **predicting nanoparticle
cytotoxicity** from physicochemical + exposure descriptors, and for
**benchmarking a zoo of machine-learning algorithms** (Logistic Regression,
SVM, k-NN, Random Forest, Extra-Trees, Gradient Boosting, XGBoost, LightGBM)
with publication-grade methodology.

It is built to support nano-QSAR research: honest nested cross-validation, a
single untouched hold-out set, a **leave-one-material-family-out** test that
directly measures the cross-family generalisation gap, permutation-based
feature importance, and ready-to-use ROC / comparison / confusion plots.

> **Data note.** The repo ships a *physicochemically-grounded synthetic data
> generator* so everything runs with **zero network access**. The synthetic
> cohort encodes well-established nano-QSAR relationships (size/surface-area
> driven oxidative stress, ion dissolution, cationic-membrane disruption,
> dose/time escalation, coating and cell-line modulation) — it is realistic
> enough to be useful for methodology development, but it is **not real
> experimental data** and must not be used to draw biological conclusions.
> Point the loader at a real dataset to do science (see
> [Using real data](#using-real-data)).

---

## Why this exists — mapping to the research gaps

The design responds directly to recognised blind spots in AI-driven
nanotoxicology:

| Research gap | What this repo provides |
|---|---|
| **Cross-family generalisation** (models trained on metal oxides fail on lipids/polymers) | `nanotox lofo` — leave-one-material-family-out evaluation quantifying the drop |
| **Weak feature engineering** (raw descriptors only) | `features/engineering.py` — domain-motivated derived descriptors (corona inflation ratio, specific reactivity, redox-window proximity, dose×dissolution) |
| **Black-box / correlation-only explanations** | Model-agnostic permutation importance + per-feature direction, a defensible first layer before causal modelling |
| **Optimistic benchmarking** | Nested CV + untouched hold-out, every stochastic step seeded |
| **Irreproducibility** | Config serialised with results; one global seed; deterministic data generator |

This is the solid, *validated* supervised baseline on top of which the more
ambitious architectures (dynamic-corona transformers, Neural-ODE biokinetics,
structural-causal layers) can later be compared and justified.

---

## Installation

```bash
pip install -r requirements.txt
# or, as a package (adds the `nanotox` command):
pip install -e .
```

Core stack: `numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`.
Optional: `xgboost`, `lightgbm` (auto-detected; the zoo expands when present),
`shap` (optional explanations).

---

## Quick start

```bash
# Inspect the dataset and available models
python -m nanotox.cli info

# Full benchmark across the whole model zoo (+ feature importance)
python -m nanotox.cli benchmark --importance

# Quick iteration preset: fewer folds/iterations, fast models only
# (skips SVM/GradientBoosting). Explicit flags still override it.
python -m nanotox.cli benchmark --fast

# Cross-family generalisation test
python -m nanotox.cli lofo

# Restrict to a few models / smaller search for a fast run
python -m nanotox.cli benchmark --models random_forest,xgboost,lightgbm --n-iter 20
```

Or without installing, from the repo root:

```bash
python scripts/run_benchmark.py --importance
```

Artifacts are written to `results/`:

- `leaderboard.csv` — ranked model comparison (CV + hold-out metrics)
- `summary.json` — best model, dataset provenance, resolved config, per-model detail
- `roc_curves.png`, `model_comparison.png`, `confusion_best.png`
- `feature_importance.csv` (with `--importance`)
- `lofo.csv` (from the `lofo` command)

---

## Methodology

1. **Features.** `DerivedFeatures` appends nano-QSAR-motivated columns, then a
   shared `ColumnTransformer` median-imputes + standardises numerics and
   one-hot encodes categoricals. Every model sees identical preprocessing, so
   comparisons are fair.
2. **Model selection.** For each model, an inner `RandomizedSearchCV`
   (stratified *k*-fold) tunes hyper-parameters.
3. **Unbiased estimate.** The tuned search is wrapped in an outer
   stratified *k*-fold (`cross_validate`) → **nested CV**, avoiding the
   optimism of tuning and evaluating on the same data.
4. **Final hold-out.** The tuned pipeline is refit on all training data and
   scored once on an untouched test split for the headline numbers and plots.
5. **Metrics.** ROC-AUC, PR-AUC, balanced accuracy, F1, MCC — chosen because
   nanotoxicity data is routinely imbalanced, where raw accuracy misleads.
6. **Generalisation.** Leave-one-material-family-out trains on all-but-one
   family and tests on the held-out family.

---

## Using real data

Any CSV with the expected descriptor columns works. Two ways in:

**A. Remote import**

```bash
python -m nanotox.cli benchmark --remote-url https://example.org/your_dataset.csv
```

**B. Drop a file** at `data/nanoparticle_cytotoxicity.csv`.

Expected columns (missing ones are tolerated / imputed):

- Numeric: `core_size_nm`, `hydrodynamic_size_nm`, `zeta_potential_mV`,
  `surface_area_m2_g`, `band_gap_eV`, `cond_band_energy_eV`,
  `dissolution_mg_L`, `concentration_mg_L`, `exposure_time_h`, `purity_pct`
- Categorical: `material_family`, `coating`, `cell_line`, `assay`
- Target: `cytotoxic` (0/1). If absent, it is derived from a `viability_pct`
  (or `cell_viability`/`viability`) column using `--viability` cutoff (default
  50%).

The column naming mirrors conventions used by community resources such as
**eNanoMapper** and published metal-oxide nano-QSAR datasets, so curated
exports can usually be mapped with a thin rename.

---

## Project layout

```
src/nanotox/
  config.py            # dataclass configuration + global seed
  data/loader.py       # remote import + reproducible synthetic fallback
  features/engineering.py  # derived descriptors + shared preprocessor
  models/zoo.py        # estimators + hyper-parameter search spaces
  pipeline.py          # nested CV, hold-out, LOFO, feature importance
  evaluate.py          # metrics, leaderboard, plots
  cli.py               # `nanotox` command
scripts/run_benchmark.py
tests/test_pipeline.py
```

## Tests

```bash
pytest -q
```

## Limitations & responsible use

- The bundled dataset is **synthetic**; it validates the *software*, not any
  biological hypothesis. Swap in experimental data before interpreting
  results scientifically.
- Permutation importance reveals *associations*, not causal mechanisms. It is
  a transparency aid, not regulatory causal evidence.
- Predictions are screening aids only and do not replace in-vitro / in-vivo
  safety testing.

## License

MIT — see [LICENSE](LICENSE).
