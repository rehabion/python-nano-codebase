"""Fast smoke / correctness tests for the nanotox pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nanotox.config import CVConfig, DataConfig, PipelineConfig  # noqa: E402
from nanotox.data import load_dataset, make_synthetic_cohort  # noqa: E402
from nanotox.features import build_preprocessor, engineered_feature_names  # noqa: E402
from nanotox.models import available_models, build_model_specs  # noqa: E402
from nanotox.pipeline import leave_one_family_out, run_benchmark  # noqa: E402


def test_synthetic_is_reproducible():
    a = make_synthetic_cohort(n=300, random_state=7)
    b = make_synthetic_cohort(n=300, random_state=7)
    assert a.equals(b)


def test_synthetic_has_both_classes_and_signal():
    df = make_synthetic_cohort(n=800, random_state=1)
    assert df["cytotoxic"].nunique() == 2
    # Smaller particles should trend more toxic -> negative correlation.
    corr = np.corrcoef(df["core_size_nm"], df["cytotoxic"])[0, 1]
    assert corr < 0


def test_loader_bundle_shapes():
    cfg = DataConfig(n_synthetic=200, cache_path=Path("/tmp/_nanotox_test.csv"))
    bundle = load_dataset(cfg)
    assert len(bundle.frame) == 200
    assert bundle.target == "cytotoxic"
    assert set(bundle.X.columns) == set(bundle.numeric_features + bundle.categorical_features)


def test_preprocessor_produces_finite_matrix():
    bundle = load_dataset(DataConfig(n_synthetic=150,
                                     cache_path=Path("/tmp/_nanotox_pp.csv")))
    pre = build_preprocessor(bundle.numeric_features, bundle.categorical_features)
    Xt = pre.fit_transform(bundle.X, bundle.y)
    assert Xt.shape[0] == len(bundle.frame)
    assert np.isfinite(Xt).all()
    # Engineered columns are present in the numeric feature list.
    names = engineered_feature_names(bundle.numeric_features)
    assert "redox_window_proximity" in names


def test_model_zoo_nonempty():
    specs = build_model_specs()
    assert "grad_boost" in specs  # always available
    assert len(available_models()) >= 5


@pytest.mark.parametrize("model_key", ["logistic", "grad_boost"])
def test_benchmark_runs_small(model_key):
    cfg = PipelineConfig(
        data=DataConfig(n_synthetic=250, cache_path=Path(f"/tmp/_nt_{model_key}.csv")),
        cv=CVConfig(outer_folds=3, inner_folds=2, n_iter_search=4),
        results_dir=Path(f"/tmp/_nt_results_{model_key}"),
        models=[model_key],
    )
    result = run_benchmark(cfg, make_plots=False)
    assert result.best_model_name
    row = result.leaderboard.iloc[0]
    # On data with real signal a reasonable model should beat chance.
    assert row["holdout_roc_auc"] > 0.6


def test_lofo_table():
    cfg = PipelineConfig(
        data=DataConfig(n_synthetic=600, cache_path=Path("/tmp/_nt_lofo.csv")),
    )
    table = leave_one_family_out(cfg, model_key="grad_boost")
    assert not table.empty
    assert {"held_out_family", "roc_auc"}.issubset(table.columns)
