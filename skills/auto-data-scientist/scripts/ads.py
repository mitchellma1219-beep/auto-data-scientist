#!/usr/bin/env python3
"""Skill entry point: forwards to the auto_data_scientist CLI."""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from auto_data_scientist.cli import main
except ModuleNotFoundError:  # running from a repository checkout without installation
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
    from auto_data_scientist.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
