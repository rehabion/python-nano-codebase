"""Command-line interface: ``nanotox ...``.

Subcommands
-----------
benchmark   Run the full nested-CV benchmark across the model zoo.
lofo        Leave-one-material-family-out generalisation test.
info        Print dataset provenance and available models.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import DataConfig, CVConfig, PipelineConfig
from .data import load_dataset
from .models import available_models
from .pipeline import feature_importance, leave_one_family_out, run_benchmark


# Fast preset: quick folds/search and a subset of inexpensive models, for
# iteration on modest hardware. Heavy models (SVM, GradientBoosting) and the
# full search are reserved for the publication-grade default run.
_FAST_MODELS = ["logistic", "random_forest", "xgboost", "lightgbm"]


def _build_config(args) -> PipelineConfig:
    data = DataConfig()
    if args.remote_url:
        data.remote_url = args.remote_url
    if args.n_synthetic:
        data.n_synthetic = args.n_synthetic
    if args.viability is not None:
        data.viability_cutoff = args.viability

    fast = getattr(args, "fast", False)
    # --fast only fills values the user did not set explicitly, so explicit
    # flags always win.
    outer = args.outer_folds if args.outer_folds is not None else (3 if fast else 5)
    inner = args.inner_folds if args.inner_folds is not None else (2 if fast else 3)
    n_iter = args.n_iter if args.n_iter is not None else (8 if fast else 40)
    cv = CVConfig(outer_folds=outer, inner_folds=inner, n_iter_search=n_iter)

    if args.models:
        models = args.models.split(",")
    elif fast:
        models = list(_FAST_MODELS)
    else:
        models = None

    return PipelineConfig(
        data=data, cv=cv,
        results_dir=Path(args.results_dir),
        models=models,
    )


def _cmd_benchmark(args) -> int:
    config = _build_config(args)
    result = run_benchmark(config, make_plots=not args.no_plots)
    result.save(config.results_dir)

    print(f"\nDataset source: {result.dataset_source}")
    print(f"Best model    : {result.best_model_name}\n")
    cols = [c for c in ["model", "cv_roc_auc", "cv_roc_auc_std",
                        "holdout_roc_auc", "holdout_pr_auc", "holdout_mcc",
                        "holdout_balanced_accuracy"]
            if c in result.leaderboard.columns]
    with_fmt = result.leaderboard[cols].copy()
    for c in cols[1:]:
        with_fmt[c] = with_fmt[c].map(lambda v: f"{v:.3f}")
    print(with_fmt.to_string(index=False))

    if args.importance:
        bundle = load_dataset(config.data)
        imp = feature_importance(result, bundle)
        if imp is not None:
            print("\nTop permutation-importance features:")
            print(imp.head(12).to_string(index=False))
            imp.to_csv(config.results_dir / "feature_importance.csv", index=False)

    print(f"\nArtifacts written to: {config.results_dir}")
    return 0


def _cmd_lofo(args) -> int:
    config = _build_config(args)
    table = leave_one_family_out(config, model_key=args.model_key)
    print("\nLeave-one-material-family-out generalisation:\n")
    fmt = table.copy()
    for c in ("roc_auc", "balanced_accuracy", "mcc"):
        if c in fmt.columns:
            fmt[c] = fmt[c].map(lambda v: f"{v:.3f}")
    print(fmt.to_string(index=False))
    config.results_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(config.results_dir / "lofo.csv", index=False)
    return 0


def _cmd_info(args) -> int:
    config = _build_config(args)
    bundle = load_dataset(config.data)
    print(f"Dataset source      : {bundle.source}")
    print(f"Rows                : {len(bundle.frame)}")
    print(f"Positive rate       : {bundle.y.mean():.3f}")
    print(f"Numeric features    : {', '.join(bundle.numeric_features)}")
    print(f"Categorical features: {', '.join(bundle.categorical_features)}")
    print(f"Available models    : {', '.join(available_models())}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="nanotox",
                                description="Nanoparticle cytotoxicity ML benchmark")
    # Shared options.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--remote-url", default=None,
                        help="CSV URL to import (falls back to synthetic if unreachable)")
    common.add_argument("--n-synthetic", type=int, default=0,
                        help="Synthetic cohort size when generating data")
    common.add_argument("--viability", type=float, default=None,
                        help="Viability%% cutoff for deriving the binary target")
    common.add_argument("--outer-folds", type=int, default=None,
                        help="Outer CV folds (default 5, or 3 with --fast)")
    common.add_argument("--inner-folds", type=int, default=None,
                        help="Inner CV folds (default 3, or 2 with --fast)")
    common.add_argument("--n-iter", type=int, default=None,
                        help="RandomizedSearchCV iterations per model "
                             "(default 40, or 8 with --fast)")
    common.add_argument("--fast", action="store_true",
                        help="Quick preset: fewer folds/iterations and a subset "
                             "of inexpensive models (skips SVM/GradientBoosting)")
    common.add_argument("--models", default=None,
                        help="Comma-separated subset of model keys")
    common.add_argument("--results-dir", default="results")

    sub = p.add_subparsers(dest="command", required=True)

    b = sub.add_parser("benchmark", parents=[common], help="Full benchmark")
    b.add_argument("--no-plots", action="store_true")
    b.add_argument("--importance", action="store_true",
                   help="Compute permutation importance for the best model")
    b.set_defaults(func=_cmd_benchmark)

    lo = sub.add_parser("lofo", parents=[common],
                        help="Leave-one-material-family-out test")
    lo.add_argument("--model-key", default="random_forest")
    lo.set_defaults(func=_cmd_lofo)

    i = sub.add_parser("info", parents=[common], help="Dataset / model info")
    i.set_defaults(func=_cmd_info)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
