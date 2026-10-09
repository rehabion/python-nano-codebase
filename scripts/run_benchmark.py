#!/usr/bin/env python3
"""Convenience entry point mirroring ``nanotox benchmark``.

Run from the repo root::

    python scripts/run_benchmark.py --importance

It adds ``src`` to the path so it works without ``pip install -e .``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nanotox.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(["benchmark", *sys.argv[1:]]))
