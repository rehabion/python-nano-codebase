"""End-to-end training, model selection and benchmarking.

Core design choices (all publication-relevant):

* **Nested cross-validation** for unbiased model comparison: an inner
  ``RandomizedSearchCV`` tunes hyper-parameters, an outer ``StratifiedKFold``
  estimates generalisation. This avoids the optimistic bias of tuning and
  evaluating on the same folds.
* **A single untouched hold-out set** gives a final, honest point estimate and
  the data for ROC/confusion plots.
* **Leave-one-material-family-out** directly measures the cross-family
  generalisation gap highlighted in the research brief.
* Everything is seeded and the resolved config is serialised with the results.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline

from .config import PipelineConfig
from .data import DatasetBundle, load_dataset
from .evaluate import (
    classification_metrics,
    leaderboard,
    plot_confusion,
    plot_model_comparison,
    plot_roc_curves,
)
from .features import build_preprocessor
from .models import build_model_specs

_SCORING = ["roc_auc", "average_precision", "balanced_accuracy", "f1", "matthews_corrcoef"]


@dataclass
class BenchmarkResult:
    """Everything a run produces, ready to serialise."""

    leaderboard: pd.DataFrame
    per_model: dict[str, dict]
    best_model_name: str
    best_estimator: Pipeline
    dataset_source: str
    config: dict

    def save(self, results_dir: Path) -> None:
        results_dir.mkdir(parents=True, exist_ok=True)
        self.leaderboard.to_csv(results_dir / "leaderboard.csv", index=False)
        summary = {
            "best_model": self.best_model_name,
            "dataset_source": self.dataset_source,
            "config": self.config,
            "per_model": {
                k: {kk: vv for kk, vv in v.items() if kk != "roc_data"}
                for k, v in self.per_model.items()
            },
        }
        (results_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float))


def _make_estimator(preprocessor: Pipeline, clf) -> Pipeline:
    """Glue the shared preprocessor to a classifier as the ``clf`` step."""
    return Pipeline(steps=[("pre", preprocessor), ("clf", clf)])


def run_benchmark(config: PipelineConfig | None = None,
                  bundle: DatasetBundle | None = None,
                  make_plots: bool = True) -> BenchmarkResult:
    """Run the full benchmark and return a :class:`BenchmarkResult`."""
    config = config or PipelineConfig()
    bundle = bundle or load_dataset(config.data)

    X, y = bundle.X, bundle.y
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.data.test_size, stratify=y,
        random_state=config.random_state,
    )

    specs = build_model_specs(config.random_state)
    if config.models:
        specs = {k: v for k, v in specs.items() if k in config.models}
        if not specs:
            raise ValueError(f"No requested models available: {config.models}")

    outer_cv = StratifiedKFold(
        n_splits=config.cv.outer_folds, shuffle=True,
        random_state=config.random_state)
    inner_cv = StratifiedKFold(
        n_splits=config.cv.inner_folds, shuffle=True,
        random_state=config.random_state)

    per_model: dict[str, dict] = {}
    roc_data: dict[str, tuple] = {}

    for key, spec in specs.items():
        t0 = time.time()
        preprocessor = build_preprocessor(
            bundle.numeric_features, bundle.categorical_features)
        estimator = _make_estimator(preprocessor, spec.factory())

        # Inner search wrapped as the estimator for the outer loop => nested CV.
        search = RandomizedSearchCV(
            estimator,
            param_distributions=spec.param_distributions,
            n_iter=config.cv.n_iter_search,
            scoring=config.cv.scoring,
            cv=inner_cv,
            random_state=config.random_state,
            n_jobs=-1,
            refit=True,
            error_score="raise",
        )

        # Unbiased generalisation estimate via nested CV on the training split.
        nested = cross_validate(
            search, X_train, y_train, cv=outer_cv, scoring=_SCORING,
            n_jobs=-1, return_estimator=False,
        )
        cv_mean = {_clean(k): float(np.mean(v)) for k, v in nested.items()
                   if k.startswith("test_")}
        cv_std = {_clean(k): float(np.std(v)) for k, v in nested.items()
                  if k.startswith("test_")}

        # Refit the tuned pipeline on all training data, score the hold-out.
        search.fit(X_train, y_train)
        best = search.best_estimator_
        y_pred = best.predict(X_test)
        y_score = _safe_scores(best, X_test)
        holdout = classification_metrics(y_test, y_pred, y_score)

        per_model[spec.name] = {
            "key": key,
            "cv_mean": cv_mean,
            "cv_std": cv_std,
            "holdout": holdout,
            "best_params": search.best_params_,
            "fit_seconds": round(time.time() - t0, 2),
            "roc_data": (y_test.to_numpy(), y_score),
            "_estimator": best,
        }
        roc_data[spec.name] = (y_test.to_numpy(), y_score)

    board = leaderboard(per_model)
    best_name = board.iloc[0]["model"]
    best_estimator = per_model[best_name]["_estimator"]

    # Strip the heavy estimator object out of the serialisable dict.
    serialisable = {k: {kk: vv for kk, vv in v.items() if kk != "_estimator"}
                    for k, v in per_model.items()}

    if make_plots:
        rdir = config.results_dir
        plot_roc_curves(roc_data, rdir / "roc_curves.png")
        plot_model_comparison(board, rdir / "model_comparison.png")
        y_pred_best = best_estimator.predict(X_test)
        plot_confusion(y_test, y_pred_best, rdir / "confusion_best.png",
                       title=f"Confusion matrix -- {best_name}")

    return BenchmarkResult(
        leaderboard=board,
        per_model=serialisable,
        best_model_name=best_name,
        best_estimator=best_estimator,
        dataset_source=bundle.source,
        config=config.to_dict(),
    )


def leave_one_family_out(config: PipelineConfig | None = None,
                         bundle: DatasetBundle | None = None,
                         model_key: str = "random_forest") -> pd.DataFrame:
    """Measure cross-family generalisation.

    Train on all-but-one material family, test on the held-out family. A large
    drop versus the pooled hold-out AUC quantifies the "generalisation
    inability across diverse material families" gap.
    """
    config = config or PipelineConfig()
    bundle = bundle or load_dataset(config.data)
    frame = bundle.frame
    if "material_family" not in frame.columns:
        raise ValueError("Dataset lacks 'material_family'; cannot run LOFO.")

    specs = build_model_specs(config.random_state)
    if model_key not in specs:
        model_key = "grad_boost"  # always-available fallback
    spec = specs[model_key]

    rows = []
    for fam in sorted(frame["material_family"].unique()):
        train = frame[frame["material_family"] != fam]
        test = frame[frame["material_family"] == fam]
        if test[bundle.target].nunique() < 2 or len(test) < 10:
            continue
        feats = bundle.numeric_features + bundle.categorical_features
        pre = build_preprocessor(bundle.numeric_features, bundle.categorical_features)
        est = _make_estimator(pre, spec.factory())
        est.fit(train[feats], train[bundle.target])
        y_score = _safe_scores(est, test[feats])
        y_pred = est.predict(test[feats])
        m = classification_metrics(test[bundle.target].to_numpy(), y_pred, y_score)
        rows.append({"held_out_family": fam, "n_test": len(test),
                     "roc_auc": m["roc_auc"], "balanced_accuracy": m["balanced_accuracy"],
                     "mcc": m["mcc"]})
    return pd.DataFrame(rows)


def feature_importance(result: BenchmarkResult,
                       bundle: DatasetBundle) -> pd.DataFrame | None:
    """Permutation importance for the best estimator (model-agnostic).

    Permutation importance is preferred over impurity-based importance because
    it is unbiased w.r.t. feature cardinality and works for any estimator.
    """
    from sklearn.inspection import permutation_importance

    est = result.best_estimator
    X, y = bundle.X, bundle.y
    try:
        r = permutation_importance(
            est, X, y, n_repeats=10, random_state=42, n_jobs=-1,
            scoring="roc_auc")
    except Exception:  # noqa: BLE001
        return None
    df = pd.DataFrame({
        "feature": X.columns,
        "importance_mean": r.importances_mean,
        "importance_std": r.importances_std,
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)
    return df


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _clean(metric_key: str) -> str:
    return metric_key.replace("test_", "")


def _safe_scores(estimator, X) -> np.ndarray | None:
    """Positive-class scores via predict_proba or decision_function."""
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X)[:, 1]
    if hasattr(estimator, "decision_function"):
        return estimator.decision_function(X)
    return None
