"""Central configuration for the nanotox pipeline.

Keeping every tunable knob in one place makes experiments reproducible and
makes it trivial to serialise the exact configuration that produced a result
(important for publication-grade work).
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

# Project root = two levels up from this file (src/nanotox/config.py -> repo).
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

#: Global seed. A single seed threaded through every stochastic component is
#: the cheapest, highest-value reproducibility guarantee a paper can offer.
RANDOM_STATE = 42

#: Name of the binary target column produced by the data layer.
TARGET_COLUMN = "cytotoxic"


@dataclass
class DataConfig:
    """How the dataset is sourced and materialised on disk."""

    # Remote source tried first. If unreachable (offline CI, air-gapped
    # container) the pipeline falls back to a physicochemically-grounded
    # synthetic generator so the whole repo is runnable with zero network.
    remote_url: str | None = None
    cache_path: Path = DATA_DIR / "nanoparticle_cytotoxicity.csv"
    # Size of the synthetic fallback cohort.
    n_synthetic: int = 1500
    # Fraction of rows held out as a final, untouched test set.
    test_size: float = 0.2
    # Viability threshold (%) below which a particle is labelled cytotoxic.
    viability_cutoff: float = 50.0


@dataclass
class CVConfig:
    """Cross-validation / model-selection protocol."""

    outer_folds: int = 5
    inner_folds: int = 3
    n_iter_search: int = 40  # RandomizedSearchCV budget per model
    scoring: str = "roc_auc"


@dataclass
class PipelineConfig:
    """Top-level configuration object."""

    data: DataConfig = field(default_factory=DataConfig)
    cv: CVConfig = field(default_factory=CVConfig)
    random_state: int = RANDOM_STATE
    results_dir: Path = RESULTS_DIR
    # Restrict the model zoo to these keys (None = all available models).
    models: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable snapshot (Paths rendered as strings)."""

        def _coerce(obj: Any) -> Any:
            if isinstance(obj, Path):
                return str(obj)
            if isinstance(obj, dict):
                return {k: _coerce(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_coerce(v) for v in obj]
            return obj

        return _coerce(asdict(self))
