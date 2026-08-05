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

    # The suite deliberately hammers /api/predict* far harder than a real visitor
    # would, so the rate limiter is off unless a test opts back in (see
    # tests/test_rate_limit.py). Set before app import so the Limiter picks it up.
    os.environ.setdefault("KARKIVE_RATE_LIMIT", "0")
