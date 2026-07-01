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

import numpy as np
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

# --- therapy panel: drugs we can authoritatively name (GDSC db export) ---------
# id -> (display name, putative target / class)
DRUGS = {
    1:  ("Erlotinib",   "EGFR inhibitor"),
    3:  ("Rapamycin",   "mTOR inhibitor"),
    5:  ("Sunitinib",   "Multi-RTK inhibitor (VEGFR/PDGFR/KIT)"),
    6:  ("PHA-665752",  "MET inhibitor"),
    9:  ("MG-132",      "Proteasome inhibitor"),
    11: ("Paclitaxel",  "Taxane (microtubule stabilizer)"),
    17: ("Cyclopamine", "Hedgehog / SMO inhibitor"),
    29: ("AZ628",       "BRAF inhibitor"),
    30: ("Sorafenib",   "Multi-kinase inhibitor (RAF/VEGFR)"),
}
DRUG_COL = {i: f"Drug_{i}_IC50" for i in DRUGS}
THERAPY_NAMES = [DRUGS[i][0] for i in DRUGS]
THERAPY_CLASS = {DRUGS[i][0]: DRUGS[i][1] for i in DRUGS}

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
    """Return (features_df, {drug_col: sensitivity_series}, tissue_categories).

    Features: one row per cell line with BINARY_FEATURES + a `tissue` column.
    Target per drug: -z(logIC50) so higher = more sensitive; NaN where untested.
    """
    ic = pd.read_csv(os.path.join(DATA_DIR, "IC50_v17.csv.gz"))
    gf = pd.read_csv(os.path.join(DATA_DIR, "genomic_features_v17.csv.gz"))
    df = gf.merge(ic, on="COSMIC_ID")

    feats = pd.DataFrame()
    for f in MUTATION_FEATURES:
        feats[f] = df[f].fillna(0).astype(float)
    feats[ERBB2_AMP] = df[ERBB2_AMP_COL].fillna(0).astype(float)
    feats[MSI] = df["MSI_FACTOR"].fillna(0).astype(float)
    feats["tissue"] = df["TISSUE_FACTOR"].astype(str)

    targets = {}
    for i, col in DRUG_COL.items():
        v = df[col].astype(float)
        targets[col] = -(v - v.mean()) / v.std()   # z-scored sensitivity

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
