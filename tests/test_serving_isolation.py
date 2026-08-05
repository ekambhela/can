"""The serving path must not depend on `data/`.

The deployed image ships `artifacts/model.joblib` but no GDSC matrices (see
.dockerignore), so anything the request path imports has to work without them.
`model.gdsc` reads those matrices at module scope, which makes "did we import
it?" a precise, cheap proxy for "did we touch data/?".

Checked in a subprocess so an unrelated test that already imported model.gdsc
can't mask a regression.
"""

import os
import shutil
import subprocess
import sys
import tempfile

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_PROBE = """
import sys
import {module}
assert "model.gdsc" not in sys.modules, (
    "{module} pulled in model.gdsc (the data layer) — the serving path must "
    "read only from the bundle. Offending chain: " + repr(
        [m for m in sys.modules if m.startswith("model")])
)
print("clean")
"""


def _run(code, cwd=ROOT):
    return subprocess.run([sys.executable, "-c", code], cwd=cwd,
                          capture_output=True, text=True)


@pytest.mark.parametrize("module", ["model.predict", "model.mtl", "model.perdrug", "app"])
def test_serving_import_does_not_touch_data_layer(module):
    proc = _run(_PROBE.format(module=module))
    assert proc.returncode == 0, proc.stderr
    assert "clean" in proc.stdout


def test_predict_works_with_data_dir_absent():
    """The real end-to-end check: copy the repo without data/ and score a sample.

    This is what `docker run` does with data/ in .dockerignore.
    """
    with tempfile.TemporaryDirectory() as tmp:
        dest = os.path.join(tmp, "app")
        shutil.copytree(
            ROOT, dest,
            ignore=shutil.ignore_patterns("data", ".git", "__pycache__", "experiments"),
        )
        assert not os.path.exists(os.path.join(dest, "data"))

        proc = _run(
            "from model.predict import predict, sample_from_dict\n"
            "s, _ = sample_from_dict({'tissue': 'skin', 'BRAF_mut': 1})\n"
            "r = predict(s, top_k=3)\n"
            "assert len(r['ranked']) == 3, r\n"
            "print('scored:', r['recommendation'])\n",
            cwd=dest,
        )
        assert proc.returncode == 0, f"scoring without data/ failed:\n{proc.stderr}"
        assert "scored:" in proc.stdout
