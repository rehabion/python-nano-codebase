"""The model zoo and their hyper-parameter search spaces.

Each entry is a :class:`ModelSpec` bundling an estimator factory with a
``RandomizedSearchCV`` parameter distribution. Optional gradient-boosting
libraries (XGBoost, LightGBM) are included only if importable, so the pipeline
runs on a minimal install and gets stronger when they are present.

Parameter-grid keys are prefixed with ``clf__`` because every estimator is the
final ``clf`` step of a shared preprocessing ``Pipeline`` (see
:mod:`nanotox.pipeline`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from scipy.stats import loguniform, randint, uniform
from sklearn.base import BaseEstimator
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from ..config import RANDOM_STATE

# Optional boosters -- imported defensively.
try:  # pragma: no cover - availability depends on environment
    from xgboost import XGBClassifier
    _HAS_XGB = True
except Exception:  # noqa: BLE001
    _HAS_XGB = False

try:  # pragma: no cover
    from lightgbm import LGBMClassifier
    _HAS_LGBM = True
except Exception:  # noqa: BLE001
    _HAS_LGBM = False


@dataclass
class ModelSpec:
    """An estimator plus its hyper-parameter search distribution."""

    name: str
    factory: Callable[[], BaseEstimator]
    param_distributions: dict[str, Any] = field(default_factory=dict)
    # Whether this estimator needs scaled/dense input (all do here, since the
    # shared preprocessor always scales + densifies).
    notes: str = ""


def build_model_specs(random_state: int = RANDOM_STATE) -> dict[str, ModelSpec]:
    """Construct every available model spec, keyed by short name."""
    specs: dict[str, ModelSpec] = {}

    specs["logistic"] = ModelSpec(
        name="LogisticRegression",
        factory=lambda: LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=random_state),
        param_distributions={
            "clf__C": loguniform(1e-3, 1e2),
        },
        notes="Linear baseline (L2); interpretable reference point.",
    )

    specs["svm_rbf"] = ModelSpec(
        name="SVM (RBF)",
        factory=lambda: SVC(
            kernel="rbf", probability=True, class_weight="balanced",
            random_state=random_state),
        param_distributions={
            "clf__C": loguniform(1e-1, 1e3),
            "clf__gamma": loguniform(1e-4, 1e0),
        },
        notes="Kernel method; strong on smooth non-linear boundaries.",
    )

    specs["knn"] = ModelSpec(
        name="k-NN",
        factory=lambda: KNeighborsClassifier(),
        param_distributions={
            "clf__n_neighbors": randint(3, 35),
            "clf__weights": ["uniform", "distance"],
            "clf__p": [1, 2],
        },
        notes="Instance-based baseline.",
    )

    specs["random_forest"] = ModelSpec(
        name="RandomForest",
        factory=lambda: RandomForestClassifier(
            class_weight="balanced_subsample", random_state=random_state,
            n_jobs=1),
        param_distributions={
            "clf__n_estimators": randint(200, 800),
            "clf__max_depth": randint(3, 25),
            "clf__min_samples_leaf": randint(1, 12),
            "clf__max_features": uniform(0.3, 0.7),
        },
        notes="Robust bagged-tree workhorse of nano-QSAR.",
    )

    specs["extra_trees"] = ModelSpec(
        name="ExtraTrees",
        factory=lambda: ExtraTreesClassifier(
            class_weight="balanced_subsample", random_state=random_state,
            n_jobs=1),
        param_distributions={
            "clf__n_estimators": randint(200, 800),
            "clf__max_depth": randint(3, 25),
            "clf__min_samples_leaf": randint(1, 12),
            "clf__max_features": uniform(0.3, 0.7),
        },
        notes="Extremely randomised trees; lower variance.",
    )

    specs["grad_boost"] = ModelSpec(
        name="GradientBoosting",
        factory=lambda: GradientBoostingClassifier(random_state=random_state),
        param_distributions={
            "clf__n_estimators": randint(150, 600),
            "clf__learning_rate": loguniform(1e-2, 3e-1),
            "clf__max_depth": randint(2, 6),
            "clf__subsample": uniform(0.6, 0.4),
        },
        notes="sklearn gradient boosting; always available.",
    )

    if _HAS_XGB:
        specs["xgboost"] = ModelSpec(
            name="XGBoost",
            factory=lambda: XGBClassifier(
                objective="binary:logistic", eval_metric="logloss",
                tree_method="hist", random_state=random_state, n_jobs=1),
            param_distributions={
                "clf__n_estimators": randint(200, 800),
                "clf__learning_rate": loguniform(1e-2, 3e-1),
                "clf__max_depth": randint(2, 8),
                "clf__subsample": uniform(0.6, 0.4),
                "clf__colsample_bytree": uniform(0.6, 0.4),
                "clf__reg_lambda": loguniform(1e-2, 1e1),
            },
            notes="Gradient-boosted trees; typically top performer.",
        )

    if _HAS_LGBM:
        specs["lightgbm"] = ModelSpec(
            name="LightGBM",
            factory=lambda: LGBMClassifier(
                objective="binary", random_state=random_state, n_jobs=1,
                verbose=-1),
            param_distributions={
                "clf__n_estimators": randint(200, 800),
                "clf__learning_rate": loguniform(1e-2, 3e-1),
                "clf__num_leaves": randint(15, 120),
                "clf__max_depth": randint(-1, 12),
                "clf__subsample": uniform(0.6, 0.4),
                "clf__colsample_bytree": uniform(0.6, 0.4),
                "clf__reg_lambda": loguniform(1e-2, 1e1),
            },
            notes="Histogram gradient boosting; fast and accurate.",
        )

    return specs


def available_models(random_state: int = RANDOM_STATE) -> list[str]:
    """Names of models usable in the current environment."""
    return list(build_model_specs(random_state).keys())
