# -*- coding: utf-8 -*-
"""CLI entry point for US Morning Scanner performance tracking."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.services.performance_tracker import main


if __name__ == "__main__":
    raise SystemExit(main())
