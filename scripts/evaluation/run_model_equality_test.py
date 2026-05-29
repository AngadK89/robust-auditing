#!/usr/bin/env python3
"""Run Model Equality Testing for OLMo-2 base vs GRPO adapter completions."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_FOR_IMPORTS = Path(__file__).resolve().parents[2]
if str(ROOT_FOR_IMPORTS) not in sys.path:
    sys.path.insert(0, str(ROOT_FOR_IMPORTS))

from robust_auditing.model_equality.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
