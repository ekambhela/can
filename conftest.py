"""Ensures the repo root is importable so tests can `from model import ...`.

Its mere presence at the repo root also makes pytest add this directory to
sys.path (rootdir insertion), which is what lets the `model` / `app` packages
import during test collection.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "perf: timing-sensitive test (deselect with -m 'not perf')"
    )
