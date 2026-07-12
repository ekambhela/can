"""
Real GDSC (Genomics of Drug Sensitivity in Cancer) data layer.

Loads the bundled GDSC release-17 matrices (988 human cancer cell lines) and
builds the training frame the model learns from: a curated set of genomic
features per cell line, and a per-drug sensitivity target derived from real
IC50 measurements.

Data provenance: Sanger `gdsctools` package (see data/gdsc/README.md).
Cite: Iorio et al., Cell 2016; Yang et al., Nucleic Acids Research 2013.

IMPORTANT: this trains on cancer *cell lines*, not patients. It is a research
demonstration of pharmacogenomic drug-response modeling, not a clinical tool.
"""

from __future__ import annotations

import os
import re as _re

import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "gdsc")

# --- curated genomic features the model + UI use -------------------------------
# Driver-gene mutation flags present in the GDSC feature matrix.
MUTATION_GENES = [
    "TP53", "KRAS", "EGFR", "BRAF", "ALK", "ERBB2", "BRCA1", "BRCA2",
    "PIK3CA", "PTEN", "NRAS", "APC", "CDKN2A",
]
MUTATION_FEATURES = [f"{g}_mut" for g in MUTATION_GENES]

# ERBB2 (HER2) amplification is a copy-number feature in GDSC.
ERBB2_AMP_COL = "gain_cnaPANCAN301_(CDK12,ERBB2,MED24)"
ERBB2_AMP = "ERBB2_amp"          # friendly key used everywhere in the app

MSI = "MSI"                       # microsatellite instability (0/1)

# The full ordered binary feature list (tissue is handled separately, one-hot).
BINARY_FEATURES = MUTATION_FEATURES + [ERBB2_AMP, MSI]

# --- therapy panel: the full GDSC compound set, named from the GDSC drug list --
# The drug annotation (drug_id -> name / targets / target pathway) is the public
# GDSC "screened compounds" table, validated against the authoritative GDSC
# database export bundled in gdsctools (9/9 anchor IDs match exactly).
# See data/gdsc/README.md.
def _load_drugs() -> tuple[dict, dict]:
    """Collapse the v17 drug columns to one clean entry per *compound*.

    Several compounds were screened twice (two drug ids, e.g. Afatinib /
    "Afatinib (1377)"). We group by base name, keep the screen with the most
    measurements as the canonical id, and record every source screen so the
    replicates can be averaged into one denoised target.

    Returns (DRUGS, DRUG_SOURCE):
      DRUGS       {canonical_id: (display_name, target_pathway, targets)}
      DRUG_SOURCE {canonical_id: [source_drug_id, ...]}   (>=1 per compound)
    """
    import pandas as pd
    ic = pd.read_csv(os.path.join(DATA_DIR, "IC50_v17.csv.gz"), nrows=1)
    v17_ids = [int(c.split("_")[1]) for c in ic.columns if c.startswith("Drug_")]
    counts = {i: int(pd.read_csv(os.path.join(DATA_DIR, "IC50_v17.csv.gz"),
                                 usecols=[f"Drug_{i}_IC50"])[f"Drug_{i}_IC50"].notna().sum())
              for i in v17_ids}
    dl = pd.read_csv(os.path.join(DATA_DIR, "drug_list_gdsc.csv"))
    meta = {int(r.drug_id): (str(r.Name).strip(), str(r["Target pathway"]).strip(),
                             str(r.Targets).strip())
            for _, r in dl.iterrows()}

    groups: dict[str, list[int]] = {}
    for did in v17_ids:
        if did not in meta:
            continue
        base = _re.sub(r"\s*\(\d+\)$", "", meta[did][0]).strip()
        groups.setdefault(base, []).append(did)

    drugs, source = {}, {}
    for base, ids in groups.items():
        ids = sorted(ids, key=lambda d: counts[d], reverse=True)   # largest screen first
        cid = ids[0]
        _, pathway, targets = meta[cid]
        drugs[cid] = (base, pathway, targets)                      # clean, unsuffixed name
        source[cid] = ids

    # --- add GDSC2 compounds we don't already model (ids namespaced 200000+) ----
    # GDSC2 is a separate, newer Sanger screen. We add only compounds absent from
    # the GDSC1/v17 panel so no drug name appears twice. See data/gdsc/README.md.
    g2_path = os.path.join(DATA_DIR, "drug_list_gdsc2.csv")
    if os.path.exists(g2_path):
        seen = {_re.sub(r"[^a-z0-9]", "", n.lower()) for (n, _p, _t) in drugs.values()}
        for _, r in pd.read_csv(g2_path).iterrows():
            name = str(r.Name).strip()
            key = _re.sub(r"[^a-z0-9]", "", name.lower())
            if key in seen:
                continue
            seen.add(key)
            nid = int(r.drug_id)
            drugs[nid] = (name, str(r["Target pathway"]).strip(), str(r.Targets).strip())
            source[nid] = [nid]
    return drugs, source


DRUGS, DRUG_SOURCE = _load_drugs()
DRUG_COL = {i: f"Drug_{i}_IC50" for i in DRUGS}
THERAPY_NAMES = [DRUGS[i][0] for i in DRUGS]

def _class_label(pathway: str, targets: str) -> str:
    """A short, human class for a drug: the target pathway, or its targets when
    the pathway is uninformative ('Other')."""
    p = (pathway or "").strip()
    if p and p.lower() not in ("other", "other, kinases", "unknown", "nan", ""):
        return p
    t = (targets or "").strip()
    return t if t and t.lower() != "nan" else "Other"

THERAPY_CLASS = {DRUGS[i][0]: _class_label(DRUGS[i][1], DRUGS[i][2]) for i in DRUGS}

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


def load_frame() -> tuple[pd.DataFrame, dict, list[str]]:
    """Return (features_df, {drug_col: raw_logIC50_series}, tissue_categories).

    Features: one row per cell line with BINARY_FEATURES + a `tissue` column.
    Target per drug: the **raw** replicate-averaged natural-log IC50 (lower =
    more sensitive), NaN where untested.

    NOTE: no normalization happens here. Converting IC50 to the model's
    sensitivity target (`-z(logIC50)`) requires a mean/std, and fitting those on
    every cell line — including the ones train.py later holds out for testing —
    would leak test-set statistics into training. The z-scoring is therefore
    done in train.py using *train-split* statistics only. See train.py.
    """
    ic = pd.read_csv(os.path.join(DATA_DIR, "IC50_v17.csv.gz"))
    gf = pd.read_csv(os.path.join(DATA_DIR, "genomic_features_v17.csv.gz"))
    df = gf.merge(ic, on="COSMIC_ID")

    # merge the added GDSC2 screen (same COSMIC ids; NaN for lines it didn't test)
    g2_path = os.path.join(DATA_DIR, "IC50_gdsc2.csv.gz")
    if os.path.exists(g2_path):
        df = df.merge(pd.read_csv(g2_path), on="COSMIC_ID", how="left")

    feats = pd.DataFrame()
    for f in MUTATION_FEATURES:
        feats[f] = df[f].fillna(0).astype(float)
    feats[ERBB2_AMP] = df[ERBB2_AMP_COL].fillna(0).astype(float)
    feats[MSI] = df["MSI_FACTOR"].fillna(0).astype(float)
    feats["tissue"] = df["TISSUE_FACTOR"].astype(str)

    targets = {}
    for cid, col in DRUG_COL.items():
        # Average the compound's replicate screens on the raw log-IC50 scale
        # (all GDSC screens report IC50 in the same natural-log µM units, so this
        # is a like-for-like average), ignoring NaNs. Normalization is deferred
        # to train.py so it can be fit on train lines only.
        raw = pd.concat([df[f"Drug_{sid}_IC50"].astype(float)
                         for sid in DRUG_SOURCE[cid]], axis=1)
        targets[col] = raw.mean(axis=1)   # NaN only where all screens missing

    tissues = sorted(df["TISSUE_FACTOR"].astype(str).unique().tolist())
    return feats, targets, tissues


def feature_schema(tissues: list[str]) -> dict:
    """Input-field schema for the manual-entry form / API."""
    return {
        "tissues": [{"key": t, "label": TISSUE_LABELS.get(t, t.replace("_", " ").title())}
                    for t in tissues],
        "mutations": [{"key": f, "label": FEATURE_LABEL.get(f, f)} for f in MUTATION_FEATURES],
        "extras": [
            {"key": ERBB2_AMP, "label": FEATURE_LABEL[ERBB2_AMP]},
            {"key": MSI, "label": FEATURE_LABEL[MSI]},
        ],
    }
