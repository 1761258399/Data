#!/usr/bin/env python3
"""Train EN3 GPR16, or validate/prepare both sessions without fitting."""
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from en3_gpr16.cli import main_train

if __name__ == "__main__":
    main_train()
