#!/usr/bin/env python3
"""Evaluate one trusted frozen [model, scaler] bundle on a held-out session."""
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from en3_gpr16.cli import main_evaluate

if __name__ == "__main__":
    main_evaluate()
