#!/usr/bin/env python3
"""Convenience launcher: delegates to scripts/start_dev.py so that

    python start_dev.py

works directly from the project root (same behaviour as
`python scripts\\start_dev.py`).
"""

from __future__ import annotations

import os
import runpy
import sys

if __name__ == "__main__":
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts", "start_dev.py")
    sys.exit(runpy.run_path(script, run_name="__main__"))
