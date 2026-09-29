#!/usr/bin/env python3
"""
=============================================================================
CHANDRASHAKTI (चन्द्रशक्ति)
Universal Sub-Pixel Multi-Modal Lunar Image Co-Registration Engine
ISRO Smart India Hackathon (SIH 2024) — Problem Statement SIH26166

Main CLI Entrypoint:
Delegates directly to the authoritative 5-phase registration engine.
=============================================================================
"""

import sys
from pathlib import Path

# Ensure workspace root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from run_pipeline import main as run_cli_main


if __name__ == "__main__":
    run_cli_main()
