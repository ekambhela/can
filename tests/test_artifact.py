"""The committed artifacts/model.joblib must load and be complete.

This is the exact failure that takes the live site down after a dependency
bump: model.joblib is a pickle of scikit-learn estimators, so a scikit-learn or
numpy upgrade without a retrain can make joblib.load raise, or load an object
whose internals no longer match. requirements.txt pins the versions for that
reason; this test is what notices when a bump breaks the pin.

It deliberately loads the real artifact rather than a fixture — a fixture would
test the fixture.
"""

import json
import os
import subprocess
import sys

import joblib
import pandas as pd
import pytest

from model import predict as P
from model.schema import BINARY_FEATURES

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACT = os.path.join(ROOT, "artifacts", "model.joblib")
METRICS_JSON = os.path.join(ROOT, "artifacts", "metrics.json")

# Every key model/predict.py reads out of the bundle.
REQUIRED_KEYS = [
    "model",              # multi-task estimator
    "per_drug_models",    # {drug_name: Pipeline}
    "blend_w_perdrug",
    "cell_cols",
    "drug_feat",          # pathway / target_cols / multihot / id_bucket
    "cat_levels",
    "drug_ids",
    "id_to_name",
    "name_to_id",
    "drug_meta",
    "tissues",
    "resid_std",
    "metrics",
    "version",
]

DRUG_FEAT_KEYS = ["pathway", "target_cols", "multihot", "id_bucket"]


@pytest.fixture(scope="module")
def bundle():
    assert os.path.exists(ARTIFACT), f"missing committed artifact: {ARTIFACT}"
    return joblib.load(ARTIFACT)


def test_artifact_loads_under_pinned_requirements(bundle):
    assert isinstance(bundle, dict)


def test_artifact_loads_in_a_cold_process():
    """A fresh interpreter, as the container does at boot."""
    proc = subprocess.run(
        [sys.executable, "-c",
         "from model.predict import get_bundle; b = get_bundle(); print(b['version'])"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip().isdigit()


@pytest.mark.parametrize("key", REQUIRED_KEYS)
def test_bundle_exposes_every_key_predict_reads(bundle, key):
    assert key in bundle, f"predict.py reads bundle[{key!r}]"


@pytest.mark.parametrize("key", DRUG_FEAT_KEYS)
def test_drug_feat_is_complete(bundle, key):
    assert key in bundle["drug_feat"]


def test_bundle_internals_are_consistent(bundle):
    ids, id_to_name, name_to_id = bundle["drug_ids"], bundle["id_to_name"], bundle["name_to_id"]

    assert ids, "no drugs in the bundle"
    assert all(d in id_to_name for d in ids)
    assert name_to_id == {v: k for k, v in id_to_name.items()}

    # every drug has annotation and a drug-feature entry
    for d in ids:
        assert id_to_name[d] in bundle["drug_meta"]
        assert d in bundle["drug_feat"]["pathway"]
        assert d in bundle["drug_feat"]["multihot"]
        assert d in bundle["drug_feat"]["id_bucket"]


def test_cell_cols_match_the_serving_feature_schema(bundle):
    """The bundle's feature order must match what _normalize produces."""
    assert list(bundle["cell_cols"]) == list(BINARY_FEATURES)


def test_tissues_are_usable(bundle):
    tissues = bundle["tissues"]
    assert len(tissues) > 10
    assert "skin" in tissues and "breast" in tissues
    assert all(isinstance(t, str) and t for t in tissues)


def test_per_drug_models_share_one_preprocessor_shape(bundle):
    """perdrug.score_pairs transforms once and reuses it across boosters."""
    models = list(bundle["per_drug_models"].values())
    assert models

    row = dict.fromkeys(bundle["cell_cols"], 0.0)
    row["tissue"] = bundle["tissues"][0]
    X = pd.DataFrame([row])

    widths = {m.named_steps["pre"].transform(X).shape[1] for m in models[:25]}
    assert len(widths) == 1, f"preprocessors disagree on width: {widths}"


# --- metrics.json <-> bundle --------------------------------------------------
def test_metrics_json_matches_the_bundle(bundle):
    with open(METRICS_JSON) as fh:
        on_disk = json.load(fh)
    in_bundle = bundle["metrics"]

    for key in ("mean_spearman", "mean_r2", "top1_accuracy", "top3_accuracy",
                "top10_accuracy", "best_drug_percentile", "n_test_lines",
                "n_cell_lines", "n_drugs"):
        assert on_disk[key] == in_bundle[key], key

    assert on_disk["per_drug_r2"] == in_bundle["per_drug_r2"]
    assert on_disk["per_drug_spearman"] == in_bundle["per_drug_spearman"]


def test_metrics_cover_the_drug_panel(bundle):
    names = set(bundle["id_to_name"].values())
    scored = set(bundle["metrics"]["per_drug_r2"])
    assert scored <= names, "metrics reference drugs not in the panel"
    assert len(scored) > 0.9 * len(names), "most drugs should have held-out metrics"


def test_n_drugs_metric_matches_the_panel(bundle):
    assert bundle["metrics"]["n_drugs"] == len(bundle["drug_ids"])


def test_api_metrics_omit_the_per_drug_dicts():
    """These are 369-entry dicts; /api/health must not ship them."""
    summary = P.api_metrics()
    assert "per_drug_r2" not in summary
    assert "per_drug_spearman" not in summary
    assert "mean_r2" in summary
