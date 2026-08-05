"""Parsing / coercion tests for model/predict.py.

The uploader accepts several file shapes; they must all normalize to the same
internal sample dict. We monkeypatch the known-tissue list so these stay fast
unit tests that don't need the trained model.
"""

import pytest

from model import predict as P

TISSUES = ["lung_NSCLC", "breast", "large_intestine", "skin"]


@pytest.fixture(autouse=True)
def _fixed_tissues(monkeypatch):
    monkeypatch.setattr(P, "_known_tissues", lambda: TISSUES)


# The same profile expressed five ways.
CSV = b"tissue,ERBB2_amp,TP53_mut\nbreast,1,1\n"
TSV = b"tissue\tERBB2_amp\tTP53_mut\nbreast\t1\t1\n"
JSON_OBJ = b'{"tissue":"breast","ERBB2_amp":1,"TP53_mut":1}'
JSON_ARR = b'[{"tissue":"breast","ERBB2_amp":1,"TP53_mut":1}]'
TWO_COL = b"feature,value\ntissue,breast\nERBB2_amp,1\nTP53_mut,1\n"


def _sample(raw, fn=""):
    sample, _warnings, _specified = P.parse_sample(raw, fn)
    return sample


def test_all_formats_normalize_identically():
    ref = _sample(CSV, "x.csv")
    assert ref["tissue"] == "breast"
    assert ref["ERBB2_amp"] == 1.0
    assert ref["TP53_mut"] == 1.0
    assert ref["KRAS_mut"] == 0.0  # unspecified feature -> 0
    for raw, fn in [(TSV, "x.tsv"), (JSON_OBJ, "x.json"),
                    (JSON_ARR, "x.json"), (TWO_COL, "x.csv")]:
        assert _sample(raw, fn) == ref, fn


@pytest.mark.parametrize("value,expected", [
    ("yes", 1.0), ("no", 0.0), ("true", 1.0), ("false", 0.0),
    ("mut", 1.0), ("wt", 0.0), ("positive", 1.0), ("negative", 0.0),
    ("1", 1.0), ("0", 0.0), (1, 1.0), (0, 0.0), (1.0, 1.0),
    ("", 0.0), ("junk", None), ("maybe", None),
])
def test_coerce_binary(value, expected):
    assert P._coerce_binary(value) == expected


def test_coerce_tissue_exact_and_case_insensitive():
    assert P._coerce_tissue("breast", TISSUES) == "breast"
    assert P._coerce_tissue("BREAST", TISSUES) == "breast"


def test_coerce_tissue_friendly_label():
    # "Colorectal" is the friendly label for the GDSC key "large_intestine".
    assert P._coerce_tissue("Colorectal", TISSUES) == "large_intestine"


def test_coerce_tissue_unknown_returns_none():
    assert P._coerce_tissue("Klingon homeworld", TISSUES) is None


def test_unknown_tissue_is_rejected_not_guessed():
    """Tissue drives the prediction (blocked-CV rho 0.10 vs 0.38 within-tissue),
    so guessing one silently answers a different question."""
    with pytest.raises(P.InvalidSample) as exc:
        P.sample_from_dict({"tissue": "Klingon homeworld", "TP53_mut": 1})
    assert exc.value.field == "tissue"
    assert "Klingon homeworld" in str(exc.value)
    assert exc.value.valid_values == TISSUES, "error must list what IS accepted"


def test_missing_tissue_is_rejected_not_guessed():
    with pytest.raises(P.InvalidSample) as exc:
        P.sample_from_dict({"TP53_mut": 1})
    assert exc.value.field == "tissue"
    assert exc.value.valid_values == TISSUES


def test_no_default_tissue_constant_remains():
    """The fallback is gone, not merely unused — nothing should reintroduce it."""
    assert not hasattr(P, "DEFAULT_TISSUE")


def test_unparseable_binary_becomes_zero_with_warning():
    sample, warnings, _spec = P.sample_from_dict({"tissue": "breast", "TP53_mut": "banana"})
    assert sample["TP53_mut"] == 0.0
    assert any("TP53_mut" in w for w in warnings)


# --- unspecified biomarkers are reported, not silently assumed ----------------
def test_unspecified_features_are_reported():
    _s, warnings, specified = P.sample_from_dict({"tissue": "breast", "TP53_mut": 1})

    assert specified == ["TP53_mut"], "only the stated marker counts as specified"
    warned = " ".join(warnings)
    assert "not specified" in warned
    assert "wild-type" in warned
    # names the markers it assumed, by friendly label
    assert "KRAS mutation" in warned


def test_explicit_zero_counts_as_specified():
    """'KRAS_mut: 0' is a measurement; omitting it is not. They must differ."""
    _s, _w, specified = P.sample_from_dict({"tissue": "breast", "KRAS_mut": 0})
    assert "KRAS_mut" in specified


def test_fully_specified_sample_warns_about_nothing():
    payload = {"tissue": "breast", **dict.fromkeys(P.BINARY_FEATURES, 0)}
    _s, warnings, specified = P.sample_from_dict(payload)
    assert set(specified) == set(P.BINARY_FEATURES)
    assert not any("not specified" in w for w in warnings)


def test_unparseable_value_is_not_counted_as_specified():
    _s, _w, specified = P.sample_from_dict({"tissue": "breast", "TP53_mut": "banana"})
    assert "TP53_mut" not in specified
