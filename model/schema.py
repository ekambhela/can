"""Definitions shared by the training and serving paths — with no dependency on
`data/`.

The serving path (model.predict, app) must be able to answer requests from
`artifacts/model.joblib` alone: the deployed image ships no GDSC matrices, and
reading them at import cost ~12 s anyway. But serving still needs the *shape* of
the model's input — which features exist, what they're called, how a tissue key
maps to a human label — and those are code, not learned parameters.

So they live here: pure constants and pure functions, importable without
touching a data file. `model.gdsc` (the training-time data layer) re-exports
them, so training code and experiments can keep importing from there.

Deliberately NOT serialized into the bundle: fixing a label typo or adding a
tooltip shouldn't require retraining the model. Anything derived from the data
itself (the drug panel, the tissue list) does live in the bundle — see
model/train.py.
"""

from __future__ import annotations

# --- curated genomic features the model + UI use -------------------------------
# Driver-gene mutation flags present in the GDSC feature matrix.
MUTATION_GENES = [
    "TP53", "KRAS", "EGFR", "BRAF", "ALK", "ERBB2", "BRCA1", "BRCA2",
    "PIK3CA", "PTEN", "NRAS", "APC", "CDKN2A",
]
MUTATION_FEATURES = [f"{g}_mut" for g in MUTATION_GENES]

ERBB2_AMP = "ERBB2_amp"          # friendly key used everywhere in the app
MSI = "MSI"                       # microsatellite instability (0/1)

# The full ordered binary feature list (tissue is handled separately, one-hot).
BINARY_FEATURES = MUTATION_FEATURES + [ERBB2_AMP, MSI]

# --- friendly labels -----------------------------------------------------------
TISSUE_LABELS = {
    "lung_NSCLC": "Lung (NSCLC)", "lung_SCLC": "Lung (SCLC)", "lung": "Lung (other)",
    "leukemia": "Leukemia", "lymphoma": "Lymphoma", "myeloma": "Myeloma",
    "aero_dig_tract": "Head & neck / aerodigestive", "skin": "Skin / melanoma",
    "nervous_system": "CNS / nervous system", "neuroblastoma": "Neuroblastoma",
    "breast": "Breast", "large_intestine": "Colorectal", "ovary": "Ovarian",
    "bone": "Bone / sarcoma", "kidney": "Kidney", "pancreas": "Pancreas",
    "stomach": "Stomach / gastric", "soft_tissue": "Soft tissue", "Bladder": "Bladder",
    "liver": "Liver", "thyroid": "Thyroid", "cervix": "Cervix",
    "endometrium": "Endometrium", "prostate": "Prostate", "biliary_tract": "Biliary tract",
    "urogenital_system_other": "Urogenital (other)", "testis": "Testis",
}

FEATURE_LABEL = {
    "TP53_mut": "TP53 mutation", "KRAS_mut": "KRAS mutation", "EGFR_mut": "EGFR mutation",
    "BRAF_mut": "BRAF mutation", "ALK_mut": "ALK mutation", "ERBB2_mut": "ERBB2/HER2 mutation",
    "BRCA1_mut": "BRCA1 mutation", "BRCA2_mut": "BRCA2 mutation", "PIK3CA_mut": "PIK3CA mutation",
    "PTEN_mut": "PTEN mutation", "NRAS_mut": "NRAS mutation", "APC_mut": "APC mutation",
    "CDKN2A_mut": "CDKN2A mutation", "ERBB2_amp": "ERBB2 / HER2 amplification",
    "MSI": "MSI-high (microsatellite instability)",
}


def feature_schema(tissues: list[str]) -> dict:
    """Input-field schema for the manual-entry form / API.

    `tissues` comes from the bundle at serving time (it is data-derived), the
    labels from this module (they are not).
    """
    return {
        "tissues": [{"key": t, "label": TISSUE_LABELS.get(t, t.replace("_", " ").title())}
                    for t in tissues],
        "mutations": [{"key": f, "label": FEATURE_LABEL.get(f, f)} for f in MUTATION_FEATURES],
        "extras": [
            {"key": ERBB2_AMP, "label": FEATURE_LABEL[ERBB2_AMP]},
            {"key": MSI, "label": FEATURE_LABEL[MSI]},
        ],
    }


# --- per-drug reliability ------------------------------------------------------
# The panel is not uniformly predictable. On the held-out split, per-drug R^2
# ranges from -0.274 to 0.495 (median 0.188) and 29 of 369 drugs land BELOW
# ZERO — the model predicts them worse than always guessing that drug's mean.
# Those drugs can still surface at rank 1, previously with exactly the same
# visual treatment as a drug the model predicts well.
#
# Thresholds are read off the shipped metrics distribution: R^2 < 0 is the
# meaningful cliff, and rho < 0.15 catches drugs with no usable ranking signal
# even where R^2 scrapes above zero. "moderate" is roughly the lowest quartile
# (p25: R^2 0.086, rho 0.308).
RELIABILITY_TIERS = ("high", "moderate", "low", "unknown")


def reliability_tier(spearman: float | None, r2: float | None) -> str:
    """Bucket one drug's held-out performance. See RELIABILITY_TIERS."""
    if spearman is None or r2 is None:
        return "unknown"
    if r2 < 0.0 or spearman < 0.15:
        return "low"
    if r2 < 0.15 or spearman < 0.30:
        return "moderate"
    return "high"


def build_reliability(metrics: dict) -> dict:
    """{drug_name: {tier, spearman, r2}} from a metrics dict.

    Derived rather than stored separately so it can never drift from the
    per-drug metrics it summarizes, and so bundles trained before this existed
    still get tiers (model/predict.py falls back to computing it on load).
    """
    rho = metrics.get("per_drug_spearman", {}) or {}
    r2 = metrics.get("per_drug_r2", {}) or {}
    return {
        name: {"tier": reliability_tier(rho.get(name), r2.get(name)),
               "spearman": rho.get(name), "r2": r2.get(name)}
        for name in set(rho) | set(r2)
    }


# --- metrics shaping -----------------------------------------------------------
PER_DRUG_METRIC_KEYS = ("per_drug_spearman", "per_drug_r2", "per_drug_reliability")


def summary_metrics(metrics: dict) -> dict:
    """Metrics minus the big per-drug dicts — safe for API responses."""
    return {k: v for k, v in metrics.items() if k not in PER_DRUG_METRIC_KEYS}
