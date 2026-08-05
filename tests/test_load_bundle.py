"""load_bundle must fail loudly, never train.

Training reads the full GDSC matrices and fits ~370 boosters. Reaching that
from a web request means a request that never returns (and on a deployed image,
one that can't work at all — data/ isn't shipped). The contract is: serving
loads an artifact an operator built, or it raises.
"""

import os
import subprocess
import sys

import pytest

from model import predict as P

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def _clean_cache():
    P.load_bundle.cache_clear()
    yield
    P.load_bundle.cache_clear()


def test_missing_model_raises_and_names_the_fix(monkeypatch, tmp_path, _clean_cache):
    monkeypatch.setattr(P, "MODEL_PATH", str(tmp_path / "absent.joblib"))

    with pytest.raises(FileNotFoundError) as exc:
        P.load_bundle()

    assert "python -m model.train" in str(exc.value), "error should tell the operator what to run"


def test_missing_model_does_not_trigger_training(monkeypatch, tmp_path, _clean_cache):
    """The regression guard: a missing artifact must not start a training run."""
    monkeypatch.setattr(P, "MODEL_PATH", str(tmp_path / "absent.joblib"))

    import model.train as T

    called = []
    monkeypatch.setattr(T, "main", lambda *a, **k: called.append(1))
    monkeypatch.setattr(T, "run_training", lambda *a, **k: called.append(1))

    with pytest.raises(FileNotFoundError):
        P.load_bundle()

    assert not called, "load_bundle must never invoke training"


def test_serving_module_does_not_import_train():
    """model.predict used to reach into model.train, which pulls in data/."""
    proc = subprocess.run(
        [sys.executable, "-c",
         "import sys, model.predict; assert 'model.train' not in sys.modules; print('ok')"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
