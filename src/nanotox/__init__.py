"""nanotox: a reproducible ML pipeline for nanoparticle cytotoxicity prediction.

The package is organised into small, composable modules:

* :mod:`nanotox.config`    -- central configuration dataclasses.
* :mod:`nanotox.data`      -- dataset acquisition (online import + reproducible
                              physicochemically-grounded fallback generator).
* :mod:`nanotox.features`  -- physicochemistry-aware feature engineering.
* :mod:`nanotox.models`    -- the model zoo + hyper-parameter search spaces.
* :mod:`nanotox.pipeline`  -- end-to-end training / nested cross-validation.
* :mod:`nanotox.evaluate`  -- metrics, comparison tables and plots.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
