"""Does the model still reproduce the pharmacology it was trained on?

These are the checks that catch a regression no schema assertion would: a
mis-wired feature vector, a scrambled drug index, or a retrain that quietly
lost the biomarker signal all still return a well-formed 200 with a plausible
ranking. What they cannot do is keep putting BRAF inhibitors at the top for a
BRAF-mutant melanoma.

Thresholds are deliberately loose — top-10 membership and the SIGN of an
attribution, not exact ranks or magnitudes. Held-out top-1 accuracy is only
11%, so anything tighter would be flaky by construction.
"""

import pytest

from model import predict as P

# Marker -> (tissue, drugs whose class is the textbook match). Named compounds
# are all in the GDSC panel.
CASES = [
    pytest.param(
        "BRAF_mut", "skin",
        {"Dabrafenib", "PLX-4720", "SB590885"},
        id="BRAF-mutant melanoma -> BRAF inhibitors",
    ),
    pytest.param(
        "ERBB2_amp", "breast",
        {"Lapatinib", "Afatinib", "Sapitinib", "CP724714", "Osimertinib", "AST-1306"},
        id="HER2-amplified breast -> HER2/EGFR-family inhibitors",
    ),
    pytest.param(
        "EGFR_mut", "lung_NSCLC",
        {"Gefitinib", "Erlotinib", "Afatinib", "Osimertinib", "Sapitinib", "AZD3759"},
        id="EGFR-mutant NSCLC -> EGFR inhibitors",
    ),
]


def _sample(tissue, **markers):
    s, _w, _spec = P.sample_from_dict({"tissue": tissue, **markers})
    return s


@pytest.fixture(scope="module")
def bundle():
    return P.get_bundle()


@pytest.mark.parametrize("marker,tissue,expected", CASES)
def test_expected_drug_class_reaches_the_top_10(marker, tissue, expected, bundle):
    known = expected & set(bundle["name_to_id"])
    assert known, f"none of {expected} are in the panel — fixture is stale"

    result = P.predict(_sample(tissue, **{marker: 1}), top_k=10)
    top10 = {r["therapy"] for r in result["ranked"]}

    assert top10 & known, (
        f"{marker}+ {tissue}: expected one of {sorted(known)} in the top 10, "
        f"got {[r['therapy'] for r in result['ranked']]}"
    )


@pytest.mark.parametrize("marker,tissue,expected", CASES)
def test_marker_has_a_positive_effect_on_its_matched_drug(marker, tissue, expected, bundle):
    """_explain must attribute the pick to the biomarker, with the right sign."""
    result = P.predict(_sample(tissue, **{marker: 1}), top_k=10)
    known = expected & set(bundle["name_to_id"])
    hits = [r["therapy"] for r in result["ranked"] if r["therapy"] in known]
    assert hits, "covered by the previous test"

    drug = hits[0]
    exp = P._explain(_sample(tissue, **{marker: 1}), drug, bundle)
    effects = {i["feature"]: i["effect_pct"] for i in exp["supporting"]}

    assert marker in effects, (
        f"{marker} is not listed as supporting {drug}; "
        f"supporting={list(effects)}, cautions={[c['feature'] for c in exp['cautions']]}"
    )
    assert effects[marker] > 0, f"{marker} should raise {drug}, got {effects[marker]}"


@pytest.mark.parametrize("marker,tissue,expected", CASES)
def test_marker_actually_moves_the_matched_drug(marker, tissue, expected, bundle):
    """Turning the marker on must raise the matched drug's predicted sensitivity."""
    known = expected & set(bundle["name_to_id"])
    result = P.predict(_sample(tissue, **{marker: 1}), top_k=10)
    hits = [r["therapy"] for r in result["ranked"] if r["therapy"] in known]
    drug = hits[0]
    did = bundle["name_to_id"][drug]

    off = P._score(bundle, _sample(tissue, **{marker: 0}), drug_ids=[did])[drug]
    on = P._score(bundle, _sample(tissue, **{marker: 1}), drug_ids=[did])[drug]

    assert on > off, f"{marker} did not increase predicted sensitivity to {drug}"


def test_braf_melanoma_is_dominated_by_braf_inhibitors():
    """The strongest signal in GDSC — worth a tighter assertion than the rest."""
    result = P.predict(_sample("skin", BRAF_mut=1), top_k=5)
    top5 = {r["therapy"] for r in result["ranked"]}
    assert len({"Dabrafenib", "PLX-4720", "SB590885"} & top5) >= 2, sorted(top5)


def test_tissue_changes_the_answer():
    """Guards against tissue being dropped from the feature vector entirely."""
    skin = P.predict(_sample("skin", TP53_mut=1), top_k=5)["recommendation"]
    leukemia = P.predict(_sample("leukemia", TP53_mut=1), top_k=5)["recommendation"]
    assert skin != leukemia


def test_a_marker_changes_the_answer():
    """Guards against the binary features being ignored."""
    plain = P.predict(_sample("skin"), top_k=8)
    braf = P.predict(_sample("skin", BRAF_mut=1), top_k=8)
    assert [r["sensitivity"] for r in plain["ranked"]] != \
           [r["sensitivity"] for r in braf["ranked"]]
