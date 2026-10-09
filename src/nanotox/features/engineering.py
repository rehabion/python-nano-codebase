"""Physicochemistry-aware feature engineering.

Two layers:

1. :class:`DerivedFeatures` -- a stateless ``sklearn`` transformer that adds
   domain-motivated derived descriptors (ratios, log-doses, interaction terms)
   that encode nano-QSAR priors the raw columns only imply.
2. :func:`build_preprocessor` -- assembles the full
   ``ColumnTransformer`` (impute + scale numerics, one-hot categoricals) that
   every model in the zoo shares, so comparisons are apples-to-apples.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Names of the engineered columns added by DerivedFeatures (in order).
_DERIVED_COLUMNS = [
    "log_concentration",
    "log_dissolution",
    "corona_inflation_ratio",   # hydrodynamic / core size
    "specific_reactivity",      # surface_area / core_size
    "abs_zeta",                 # |zeta| -- colloidal (in)stability
    "cationic_flag",            # zeta > +10 mV
    "dose_x_dissolution",       # ion-release burden at dose
    "size_x_surface",           # size-normalised surface reactivity
    "redox_window_proximity",   # closeness of cond. band to redox window
]


class DerivedFeatures(BaseEstimator, TransformerMixin):
    """Append domain-motivated derived descriptors to the numeric block.

    The transformer is pure (no fitted state beyond column bookkeeping) so it
    is safe to place upstream of imputation inside a CV pipeline without
    leakage. Missing inputs propagate as NaN and are handled downstream by the
    imputer.
    """

    def __init__(self) -> None:
        self.input_columns_: list[str] = []
        self.output_columns_: list[str] = []

    def fit(self, X: pd.DataFrame, y=None):  # noqa: N803
        self.input_columns_ = list(X.columns)
        self.output_columns_ = list(X.columns) + _DERIVED_COLUMNS
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        X = X.copy()

        def col(name: str) -> pd.Series:
            return X[name] if name in X.columns else pd.Series(np.nan, index=X.index)

        conc = col("concentration_mg_L")
        diss = col("dissolution_mg_L")
        core = col("core_size_nm")
        hydro = col("hydrodynamic_size_nm")
        sarea = col("surface_area_m2_g")
        zeta = col("zeta_potential_mV")
        cbe = col("cond_band_energy_eV")

        eps = 1e-6
        X["log_concentration"] = np.log1p(conc.clip(lower=0))
        X["log_dissolution"] = np.log1p(diss.clip(lower=0))
        X["corona_inflation_ratio"] = hydro / (core + eps)
        X["specific_reactivity"] = sarea / (core + eps)
        X["abs_zeta"] = zeta.abs()
        X["cationic_flag"] = (zeta > 10).astype(float)
        X["dose_x_dissolution"] = np.log1p(conc.clip(lower=0)) * np.log1p(diss.clip(lower=0))
        X["size_x_surface"] = sarea / (core**2 + eps)
        # Cellular redox window centred near -4.5 eV; closeness -> ROS potential.
        X["redox_window_proximity"] = np.exp(-((cbe + 4.5) ** 2) / 0.5)

        return X

    def get_feature_names_out(self, input_features=None):  # noqa: D401
        return np.asarray(self.output_columns_, dtype=object)


def engineered_feature_names(numeric_features: list[str]) -> list[str]:
    """The numeric feature names after :class:`DerivedFeatures` expansion."""
    return list(numeric_features) + _DERIVED_COLUMNS


def build_preprocessor(numeric_features: list[str],
                       categorical_features: list[str]) -> Pipeline:
    """Full preprocessing pipeline shared by every model.

    numeric  : DerivedFeatures -> median impute -> standardise
    category : most-frequent impute -> one-hot (dense, ignore unknown)
    """
    numeric_out = engineered_feature_names(numeric_features)

    numeric_pipe = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_pipe = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    column_transform = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_out),
            ("cat", categorical_pipe, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )

    # DerivedFeatures runs first so the engineered numeric columns exist before
    # the ColumnTransformer selects them by name.
    return Pipeline(steps=[
        ("derive", DerivedFeatures()),
        ("columns", column_transform),
    ])
