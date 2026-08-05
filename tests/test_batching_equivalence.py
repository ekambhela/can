"""Batching the scoring passes must not change a single number.

_explain used to rescore every (drug, toggled-feature) variant on its own: up to
8 drugs x 15 features x 2 models = ~240 predict calls per request. Cohort
scoring called each of ~369 boosters on one row at a time, so a 500-row upload
meant ~184,500 one-row calls.

Both now assemble one design matrix per model. Boosted-tree prediction is
row-independent, so this is supposed to be exactly equal — these tests pin that
rather than trusting it. Tolerance is 1e-9; the values are float32-derived and
in practice come back bit-identical.
"""

import pytest

from model import mtl, perdrug
from model import predict as P

TOL = 1e-9

SAMPLES = [
    {"tissue": "skin", "BRAF_mut": 1.0, "TP53_mut": 1.0, "CDKN2A_mut": 1.0, "PTEN_mut": 1.0},
    {"tissue": "breast", "ERBB2_amp": 1.0, "PIK3CA_mut": 1.0, "TP53_mut": 1.0},
    {"tissue": "large_intestine", "KRAS_mut": 1.0, "APC_mut": 1.0, "MSI": 1.0},
    {"tissue": "lung_NSCLC", "EGFR_mut": 1.0},
]


@pytest.fixture(scope="module")
def bundle():
    return P.get_bundle()


def _full(sample):
    s, _w, _spec = P.sample_from_dict(sample)
    return s


# --- the reference implementations, as they were before batching --------------
def _reference_score(bundle, sample, drug_ids):
    """One (sample, drug) at a time, through the un-batched public helpers."""
    out = {}
    for d in drug_ids:
        name = bundle["id_to_name"][d]
        mt = mtl.score_sample(bundle, sample, drug_ids=[d])[name]
        pdv = perdrug.score_sample(bundle["per_drug_models"], sample,
                                   bundle["cell_cols"], [d], bundle["id_to_name"])
        w = bundle.get("blend_w_perdrug", 0.35)
        a = pdv.get(name)
        out[name] = mt if a is None else w * a + (1 - w) * mt
    return out


def _reference_explain(sample, therapy, bundle):
    """The pre-batching _explain: rescore each toggled feature on its own."""
    did = bundle["name_to_id"][therapy]
    base = _reference_score(bundle, sample, [did])[therapy]
    supporting, cautions = [], []
    for f in P.MUTATION_FEATURES + [P.ERBB2_AMP, P.MSI]:
        if float(sample.get(f, 0)) < 0.5:
            continue
        off = dict(sample)
        off[f] = 0.0
        eff = P._pct(base) - P._pct(_reference_score(bundle, off, [did])[therapy])
        if abs(eff) < 1.5:
            continue
        (supporting if eff > 0 else cautions).append((f, round(eff, 1)))
    supporting.sort(key=lambda t: -t[1])
    cautions.sort(key=lambda t: t[1])
    return supporting, cautions


# --- equivalence --------------------------------------------------------------
@pytest.mark.parametrize("raw", SAMPLES)
def test_batched_scoring_matches_per_call(bundle, raw):
    sample = _full(raw)
    ids = bundle["drug_ids"][:40]

    batched = P._score(bundle, sample, drug_ids=ids)
    reference = _reference_score(bundle, sample, ids)

    assert set(batched) == set(reference)
    for name in reference:
        assert abs(batched[name] - reference[name]) < TOL, name


@pytest.mark.parametrize("raw", SAMPLES)
def test_batched_explanation_matches_per_call(bundle, raw):
    sample = _full(raw)
    therapies = [r["therapy"] for r in P._predict_impl(sample, top_k=5)["ranked"]]

    batched = P._explain_all(sample, therapies, bundle)

    for t in therapies:
        ref_sup, ref_cau = _reference_explain(sample, t, bundle)
        got_sup = [(i["feature"], i["effect_pct"]) for i in batched[t]["supporting"]
                   if i["feature"] is not None]
        got_cau = [(i["feature"], i["effect_pct"]) for i in batched[t]["cautions"]]

        assert got_sup == ref_sup, t
        assert got_cau == ref_cau, t


def test_explain_all_matches_single_explain(bundle):
    """The multi-drug pass and the single-drug wrapper must agree."""
    sample = _full(SAMPLES[0])
    therapies = [r["therapy"] for r in P._predict_impl(sample, top_k=4)["ranked"]]
    together = P._explain_all(sample, therapies, bundle)
    for t in therapies:
        assert P._explain(sample, t, bundle) == together[t], t


def test_cohort_scoring_matches_single_scoring(bundle):
    """_score_cohort must equal scoring each row on its own."""
    samples = [_full(s) for s in SAMPLES]
    batched = list(P._score_cohort(bundle, samples, chunk=3))   # forces >1 chunk

    assert len(batched) == len(samples)
    for s, got in zip(samples, batched):
        want = P._score(bundle, s)
        assert set(got) == set(want)
        for n in want:
            assert abs(got[n] - want[n]) < TOL, n


def test_cohort_chunking_is_invariant(bundle):
    """Results must not depend on how the cohort is split into chunks."""
    samples = [_full(s) for s in SAMPLES]
    a = list(P._score_cohort(bundle, samples, chunk=1))
    b = list(P._score_cohort(bundle, samples, chunk=len(samples)))
    for x, y in zip(a, b):
        assert max(abs(x[n] - y[n]) for n in x) < TOL


def test_pair_scoring_handles_repeated_and_reordered_pairs(bundle):
    """Pairs are arbitrary: the same drug twice, out of order, must be consistent."""
    sample = _full(SAMPLES[0])
    ids = bundle["drug_ids"][:5]
    pairs = [(0, ids[2]), (0, ids[0]), (0, ids[2]), (0, ids[4])]
    vals = P._score_pairs(bundle, [sample], pairs)

    assert vals[0] == pytest.approx(vals[2], abs=TOL), "same pair -> same value"
    ref = P._score(bundle, sample, drug_ids=ids)
    assert abs(vals[1] - ref[bundle["id_to_name"][ids[0]]]) < TOL


def test_empty_pairs_is_safe(bundle):
    assert len(P._score_pairs(bundle, [], [])) == 0


def test_full_prediction_is_unchanged_end_to_end(bundle):
    """Belt and braces: the whole response, scored the slow way, matches."""
    sample = _full(SAMPLES[1])
    result = P._predict_impl(sample, top_k=6)

    reference = _reference_score(bundle, sample, bundle["drug_ids"])
    order = sorted(reference, key=lambda n: reference[n], reverse=True)[:6]

    assert [r["therapy"] for r in result["ranked"]] == order
    for r in result["ranked"]:
        assert abs(r["sensitivity"] - round(reference[r["therapy"]], 4)) < 1e-4
