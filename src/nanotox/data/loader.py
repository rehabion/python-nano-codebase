"""Dataset acquisition with a graceful offline fallback.

Strategy (in order):

1. If ``remote_url`` is configured and reachable, download the CSV and cache it.
2. Else, if a cached CSV already exists on disk, use it.
3. Else, synthesise a physicochemically-grounded cohort so the pipeline is
   fully runnable with no network access.

The synthetic generator is **not** random noise. It encodes well-established
nano-QSAR relationships from the metal-oxide / lipid / carbon / polymer
literature so that a sane model can actually learn signal, while remaining
clearly labelled as synthetic. Replace the loader's ``remote_url`` (or drop a
curated CSV at ``cache_path``) to train on real experimental data without
touching the rest of the pipeline.

Expected real-world sources this schema is compatible with include
eNanoMapper exports, the NIL/S2NANO cytotoxicity compilations and the metal
oxide nano-QSAR datasets used throughout the literature. Column names below
mirror those conventions.
"""

from __future__ import annotations

import io
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import DataConfig, TARGET_COLUMN

# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #

#: Material families covered. Modelling several families at once is what lets
#: us probe the "cross-family generalisation" gap (train on some, test on a
#: held-out family -- see pipeline.leave_one_family_out).
MATERIAL_FAMILIES = ("metal_oxide", "carbon", "lipid", "polymer")

#: Numeric physicochemical + exposure descriptors.
NUMERIC_FEATURES = [
    "core_size_nm",            # primary particle diameter
    "hydrodynamic_size_nm",    # size in medium (corona-inflated)
    "zeta_potential_mV",       # surface charge
    "surface_area_m2_g",       # specific surface area (BET)
    "band_gap_eV",             # electronic structure (oxidative potential)
    "cond_band_energy_eV",     # conduction band energy (metal-oxide QSAR)
    "dissolution_mg_L",        # ion-release proxy
    "concentration_mg_L",      # administered dose
    "exposure_time_h",         # assay duration
    "purity_pct",              # synthesis purity
]

#: Categorical descriptors.
CATEGORICAL_FEATURES = [
    "material_family",
    "coating",                 # surface functionalisation
    "cell_line",               # biological context
    "assay",                   # viability readout
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


@dataclass
class DatasetBundle:
    """A fully materialised dataset plus provenance metadata."""

    frame: pd.DataFrame
    numeric_features: list[str]
    categorical_features: list[str]
    target: str
    source: str  # "remote", "cache" or "synthetic"

    @property
    def X(self) -> pd.DataFrame:  # noqa: N802 (X is conventional in ML)
        return self.frame[self.numeric_features + self.categorical_features]

    @property
    def y(self) -> pd.Series:
        return self.frame[self.target]


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #

def load_dataset(config: DataConfig | None = None) -> DatasetBundle:
    """Return a :class:`DatasetBundle`, fetching or synthesising as needed."""

    config = config or DataConfig()
    frame: pd.DataFrame | None = None
    source = "synthetic"

    if config.remote_url:
        frame = _try_download(config.remote_url)
        if frame is not None:
            source = "remote"
            _cache(frame, config.cache_path)

    if frame is None and config.cache_path.exists():
        frame = pd.read_csv(config.cache_path)
        source = "cache"

    if frame is None:
        frame = make_synthetic_cohort(
            n=config.n_synthetic,
            viability_cutoff=config.viability_cutoff,
        )
        source = "synthetic"
        _cache(frame, config.cache_path)

    frame = _coerce_schema(frame, config.viability_cutoff)

    return DatasetBundle(
        frame=frame,
        numeric_features=[c for c in NUMERIC_FEATURES if c in frame.columns],
        categorical_features=[c for c in CATEGORICAL_FEATURES if c in frame.columns],
        target=TARGET_COLUMN,
        source=source,
    )


# --------------------------------------------------------------------------- #
# Remote / cache helpers
# --------------------------------------------------------------------------- #

def _try_download(url: str, timeout: int = 30) -> pd.DataFrame | None:
    """Best-effort CSV download. Returns ``None`` on any failure."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310
            raw = resp.read()
        return pd.read_csv(io.BytesIO(raw))
    except (urllib.error.URLError, TimeoutError, ValueError, OSError):
        return None


def _cache(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _coerce_schema(frame: pd.DataFrame, viability_cutoff: float) -> pd.DataFrame:
    """Normalise column names and derive the binary target if necessary."""
    frame = frame.copy()

    # Derive the binary target from a viability column if the dataset ships
    # continuous viability (%) rather than a pre-binarised label.
    if TARGET_COLUMN not in frame.columns:
        for cand in ("viability_pct", "cell_viability", "viability"):
            if cand in frame.columns:
                frame[TARGET_COLUMN] = (frame[cand] < viability_cutoff).astype(int)
                break

    if TARGET_COLUMN not in frame.columns:
        raise ValueError(
            f"Dataset has no '{TARGET_COLUMN}' column and no viability column "
            "to derive it from. Available columns: " + ", ".join(frame.columns)
        )

    frame[TARGET_COLUMN] = frame[TARGET_COLUMN].astype(int)
    return frame


# --------------------------------------------------------------------------- #
# Physicochemically-grounded synthetic generator
# --------------------------------------------------------------------------- #

# Per-family priors for the numeric descriptors: (mean, std, low, high).
# Values are chosen to sit inside ranges commonly reported in the
# nanotoxicology literature for each material class.
_FAMILY_PRIORS: dict[str, dict[str, tuple[float, float, float, float]]] = {
    "metal_oxide": {
        "core_size_nm": (30, 18, 5, 120),
        "zeta_potential_mV": (12, 20, -45, 55),
        "surface_area_m2_g": (90, 50, 10, 300),
        "band_gap_eV": (3.2, 0.6, 1.5, 5.5),
        "cond_band_energy_eV": (-4.3, 0.8, -6.5, -2.5),
        "dissolution_mg_L": (6.0, 5.0, 0.0, 40.0),
        "purity_pct": (98, 1.5, 90, 100),
    },
    "carbon": {
        "core_size_nm": (20, 12, 3, 80),
        "zeta_potential_mV": (-22, 15, -55, 20),
        "surface_area_m2_g": (210, 90, 40, 500),
        "band_gap_eV": (0.4, 0.5, 0.0, 2.0),
        "cond_band_energy_eV": (-4.8, 0.5, -6.0, -3.5),
        "dissolution_mg_L": (0.5, 0.8, 0.0, 5.0),
        "purity_pct": (95, 3.0, 80, 100),
    },
    "lipid": {
        "core_size_nm": (95, 35, 40, 220),
        "zeta_potential_mV": (-8, 18, -40, 40),
        "surface_area_m2_g": (25, 15, 5, 80),
        "band_gap_eV": (4.5, 0.4, 3.5, 5.5),
        "cond_band_energy_eV": (-5.5, 0.4, -6.5, -4.5),
        "dissolution_mg_L": (1.0, 1.2, 0.0, 8.0),
        "purity_pct": (97, 2.0, 88, 100),
    },
    "polymer": {
        "core_size_nm": (110, 45, 30, 300),
        "zeta_potential_mV": (5, 22, -50, 55),
        "surface_area_m2_g": (40, 25, 5, 130),
        "band_gap_eV": (3.8, 0.7, 2.0, 5.5),
        "cond_band_energy_eV": (-5.0, 0.6, -6.5, -3.5),
        "dissolution_mg_L": (0.8, 1.0, 0.0, 6.0),
        "purity_pct": (96, 2.5, 85, 100),
    },
}

_COATINGS = ("none", "PEG", "citrate", "carboxyl", "amine", "silica")
_CELL_LINES = ("A549", "HepG2", "HeLa", "RAW264.7", "BEAS-2B", "HEK293")
_ASSAYS = ("MTT", "MTS", "LDH", "WST-1", "CellTiterGlo")

# Relative biological sensitivity multipliers (toxic response scaling).
_CELL_SENSITIVITY = {
    "A549": 1.0, "HepG2": 1.15, "HeLa": 0.95,
    "RAW264.7": 1.3, "BEAS-2B": 1.1, "HEK293": 0.9,
}
# Protective effect of coatings (lower = more protective / more stealth).
_COATING_PROTECTION = {
    "none": 1.0, "PEG": 0.55, "citrate": 0.85,
    "carboxyl": 0.9, "amine": 1.25, "silica": 0.75,
}


def _truncated_normal(rng, mean, std, low, high, size):
    vals = rng.normal(mean, std, size)
    return np.clip(vals, low, high)


def _scalar_tn(rng, prior) -> float:
    """Draw a single truncated-normal scalar from a (mean, std, low, high) prior."""
    return float(_truncated_normal(rng, *prior, 1)[0])


def make_synthetic_cohort(n: int = 1500, viability_cutoff: float = 50.0,
                          random_state: int = 42) -> pd.DataFrame:
    """Generate a reproducible, mechanism-aware synthetic cohort.

    The latent viability is driven by a smooth, non-linear combination of
    descriptors reflecting three dominant cytotoxicity mechanisms reported in
    the literature:

    * **Ion dissolution** -- soluble-metal release (dose- and family-dependent).
    * **Oxidative stress** -- small size, high surface area and favourable
      conduction-band energy promote ROS generation.
    * **Membrane interaction** -- strongly cationic (positive zeta) particles
      disrupt membranes and are taken up aggressively.

    Coating, cell line, dose and exposure time modulate the response. Gaussian
    noise is added so the task is non-trivial and the problem stays realistic.
    """
    rng = np.random.default_rng(random_state)

    families = rng.choice(MATERIAL_FAMILIES, size=n,
                          p=[0.4, 0.2, 0.2, 0.2])
    rows: list[dict] = []

    for fam in families:
        priors = _FAMILY_PRIORS[fam]
        core = _scalar_tn(rng, priors["core_size_nm"])
        # Hydrodynamic size: corona inflates core size, more so for small/charged
        # particles; add medium-dependent aggregation.
        hydro = core * rng.uniform(1.1, 2.4) + rng.normal(0, 8)
        hydro = float(np.clip(hydro, core, 600))

        zeta = _scalar_tn(rng, priors["zeta_potential_mV"])
        sarea = _scalar_tn(rng, priors["surface_area_m2_g"])
        bgap = _scalar_tn(rng, priors["band_gap_eV"])
        cbe = _scalar_tn(rng, priors["cond_band_energy_eV"])
        diss = _scalar_tn(rng, priors["dissolution_mg_L"])
        purity = _scalar_tn(rng, priors["purity_pct"])

        conc = float(np.clip(rng.lognormal(mean=2.2, sigma=0.9), 0.5, 500))
        exposure = float(rng.choice([6, 12, 24, 48, 72]))
        coating = rng.choice(_COATINGS)
        cell = rng.choice(_CELL_LINES)
        assay = rng.choice(_ASSAYS)

        rows.append(dict(
            material_family=fam, core_size_nm=core, hydrodynamic_size_nm=hydro,
            zeta_potential_mV=zeta, surface_area_m2_g=sarea, band_gap_eV=bgap,
            cond_band_energy_eV=cbe, dissolution_mg_L=diss,
            concentration_mg_L=conc, exposure_time_h=exposure, purity_pct=purity,
            coating=coating, cell_line=cell, assay=assay,
        ))

    df = pd.DataFrame(rows)

    # --- Latent toxicity score (higher = more toxic) ----------------------- #
    # Oxidative-stress term: small size + large surface area + conduction band
    # energy overlapping the cellular redox window (~ -4.8 .. -4.2 eV).
    size_term = np.exp(-df["core_size_nm"] / 40.0)          # small -> ~1
    sa_term = df["surface_area_m2_g"] / 300.0
    redox_overlap = np.exp(-((df["cond_band_energy_eV"] + 4.5) ** 2) / 0.5)
    oxidative = 1.6 * size_term + 0.8 * sa_term + 1.2 * redox_overlap

    # Dissolution term: ion release scaled by dose (log) -- metal oxides matter
    # most here.
    dissolution = 0.9 * np.log1p(df["dissolution_mg_L"]) \
        * np.log1p(df["concentration_mg_L"]) / 4.0

    # Membrane term: cationic particles (positive zeta) are disruptive.
    membrane = 0.04 * np.clip(df["zeta_potential_mV"], 0, None)

    # Dose / time escalation.
    dose_time = 0.5 * np.log1p(df["concentration_mg_L"]) \
        + 0.012 * df["exposure_time_h"]

    score = oxidative + dissolution + membrane + dose_time

    # Biological + formulation modulation.
    score *= df["cell_line"].map(_CELL_SENSITIVITY).to_numpy()
    score *= df["coating"].map(_COATING_PROTECTION).to_numpy()
    # Impurities add a small toxic burden.
    score += 0.05 * (100.0 - df["purity_pct"])

    # Irreducible biological/assay noise.
    score += rng.normal(0, 0.9, size=len(df))

    # Map latent score -> viability (%) via a logistic response, then invert.
    # Centre/scale chosen so the cohort is reasonably balanced.
    p_toxic = 1.0 / (1.0 + np.exp(-(score - score.mean()) / (score.std() + 1e-9)))
    viability = 100.0 * (1.0 - p_toxic) + rng.normal(0, 5, size=len(df))
    df["viability_pct"] = np.clip(viability, 0, 100).round(2)
    df[TARGET_COLUMN] = (df["viability_pct"] < viability_cutoff).astype(int)

    return df
