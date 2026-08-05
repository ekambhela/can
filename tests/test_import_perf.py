"""Import-time performance guard for model.gdsc.

`_load_drugs` used to re-read the gzipped 988x265 IC50 matrix once per drug id
— 265 full decompressions of the same file — just to count non-NaN entries per
column. Because that runs at module scope, `import model.gdsc` cost ~12 s, and
model.predict imports it transitively, so it landed on the serving path too.

Timed in a *cold* subprocess: an in-process `import` is a no-op once any other
test module has already imported it, so it would measure nothing.
"""

import os
import subprocess
import sys
import time

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Generous vs. the ~0.6 s we actually land at (mostly pandas' own import), but
# far below the ~12 s regression this guards against.
BUDGET_S = 2.0


@pytest.mark.perf
def test_import_gdsc_is_fast():
    t0 = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, "-c", "import model.gdsc"],
        cwd=ROOT, capture_output=True, text=True,
    )
    elapsed = time.perf_counter() - t0

    assert proc.returncode == 0, f"import failed:\n{proc.stderr}"
    assert elapsed < BUDGET_S, (
        f"import model.gdsc took {elapsed:.2f}s (budget {BUDGET_S}s). "
        "Did something start re-reading the IC50 matrix at module scope?"
    )
