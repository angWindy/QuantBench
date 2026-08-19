"""Pytest configuration for QuantBench.

Adds the ``src/`` directory to sys.path so ``quantbench`` is importable
without needing the package to be installed in editable mode. Lets us run
``pytest`` directly from the project root.
"""
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))