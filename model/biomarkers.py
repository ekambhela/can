"""
Domain knowledge layer for the Tumor → Chemotherapy matcher.

This module is the single source of truth for:
  * the tumor-sample feature schema the model expects,
  * the therapy panel the model ranks over,
  * the pharmacogenomic rules that link tumor biology to drug sensitivity.

The rules below are deliberately grounded in well-established, textbook
oncology biomarker–drug relationships (e.g. BRCA loss → platinum/PARP
sensitivity, HER2 amplification → trastuzumab, EGFR-activating mutation →
EGFR-TKI, high TMB/MSI-high → checkpoint inhibitor). They are used to
generate a biologically plausible training set so the model learns
clinically meaningful structure rather than arbitrary noise.

IMPORTANT: this is a research / educational demonstration, NOT a validated
clinical decision tool. See README for the disclaimer.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Feature schema
# ---------------------------------------------------------------------------
# Cancer types the model is aware of (one-hot encoded at training time).
CANCER_TYPES = [
    "breast",
    "lung_nsclc",
    "ovarian",
    "colorectal",
    "melanoma",
    "pancreatic",
    "gastric",
    "glioma",
]

# Binary genomic-alteration features (1 = alteration present in the tumor).
MUTATION_FEATURES = [
    "TP53_mut",
    "KRAS_mut",
    "EGFR_mut",        # activating (exon 19 del / L858R)
    "BRAF_V600",
    "ALK_fusion",
    "HER2_amp",        # ERBB2 amplification
    "BRCA1_mut",
    "BRCA2_mut",
    "PIK3CA_mut",
    "PTEN_loss",
    "MSI_high",        # microsatellite instability-high / dMMR
]

# Continuous / ordinal sample-level features.
#   TMB           : tumor mutational burden (mutations / Mb), ~0–40
#   proliferation : Ki-67-like proliferation index, 0–1
#   ER_expr       : estrogen-receptor expression, 0–1
#   ABCB1_expr    : MDR1 efflux pump expression (drug resistance), 0–1
#   ERCC1_expr    : nucleotide-excision-repair capacity (platinum resistance), 0–1
#   TUBB3_expr    : class-III beta-tubulin (taxane resistance), 0–1
CONTINUOUS_FEATURES = [
    "TMB",
    "proliferation",
    "ER_expr",
    "ABCB1_expr",
    "ERCC1_expr",
    "TUBB3_expr",
]

# Full ordered feature list used by the model (cancer type expands to one-hot).
FEATURE_COLUMNS = MUTATION_FEATURES + CONTINUOUS_FEATURES + ["cancer_type"]

# Reasonable population priors for the continuous features (mean, sd, lo, hi).
CONTINUOUS_PRIORS = {
    "TMB": (6.0, 6.0, 0.0, 40.0),
    "proliferation": (0.35, 0.20, 0.0, 1.0),
    "ER_expr": (0.45, 0.30, 0.0, 1.0),
    "ABCB1_expr": (0.30, 0.20, 0.0, 1.0),
    "ERCC1_expr": (0.40, 0.22, 0.0, 1.0),
    "TUBB3_expr": (0.35, 0.20, 0.0, 1.0),
}

# ---------------------------------------------------------------------------
# Therapy panel
# ---------------------------------------------------------------------------
# Each therapy carries a human-readable class for display.
THERAPIES = {
    "Cisplatin":     "Platinum alkylating agent",
    "Carboplatin":   "Platinum alkylating agent",
    "Paclitaxel":    "Taxane (microtubule stabilizer)",
    "Docetaxel":     "Taxane (microtubule stabilizer)",
    "Doxorubicin":   "Anthracycline (topo-II inhibitor)",
    "Gemcitabine":   "Antimetabolite (nucleoside analog)",
    "Fluorouracil":  "Antimetabolite (5-FU)",
    "Irinotecan":    "Topoisomerase-I inhibitor",
    "Olaparib":      "PARP inhibitor",
    "Erlotinib":     "EGFR tyrosine-kinase inhibitor",
    "Trastuzumab":   "Anti-HER2 monoclonal antibody",
    "Vemurafenib":   "BRAF V600 inhibitor",
    "Crizotinib":    "ALK inhibitor",
    "Pembrolizumab": "PD-1 checkpoint inhibitor",
    "Temozolomide":  "Alkylating agent (CNS-penetrant)",
}

THERAPY_NAMES = list(THERAPIES.keys())


# ---------------------------------------------------------------------------
# Pharmacogenomic response model (used only to synthesize training labels)
# ---------------------------------------------------------------------------
# A drug's *sensitivity score* for a sample starts at a baseline and is then
# pushed up or down by the sample's biology. Higher score = more effective.
# Scores are later normalized + noised in generate_data.py.

# Baseline effectiveness per drug (broad-spectrum cytotoxics start higher).
BASELINE = {
    "Cisplatin": 0.45, "Carboplatin": 0.45, "Paclitaxel": 0.45,
    "Docetaxel": 0.45, "Doxorubicin": 0.42, "Gemcitabine": 0.40,
    "Fluorouracil": 0.40, "Irinotecan": 0.38, "Olaparib": 0.20,
    "Erlotinib": 0.18, "Trastuzumab": 0.15, "Vemurafenib": 0.15,
    "Crizotinib": 0.15, "Pembrolizumab": 0.22, "Temozolomide": 0.30,
}


def sensitivity_scores(sample: dict) -> dict:
    """Return a dict {therapy: latent_sensitivity} for one tumor sample.

    `sample` is a dict with keys from MUTATION_FEATURES (0/1),
    CONTINUOUS_FEATURES (floats), and "cancer_type" (str).
    The returned latent score is unbounded-ish; the caller normalizes it.
    """
    s = {k: BASELINE[k] for k in THERAPY_NAMES}

    g = lambda k: float(sample.get(k, 0))          # genomic flag accessor
    c = lambda k: float(sample.get(k, 0.0))        # continuous accessor
    ctype = sample.get("cancer_type", "")

    # --- Targeted-agent biomarker matches (strong, specific signals) --------
    # EGFR-activating mutation → EGFR-TKI responder.
    if g("EGFR_mut"):
        s["Erlotinib"] += 0.75
    # HER2 amplification → anti-HER2 antibody.
    if g("HER2_amp"):
        s["Trastuzumab"] += 0.78
    # BRAF V600 → BRAF inhibitor (esp. melanoma).
    if g("BRAF_V600"):
        s["Vemurafenib"] += 0.72 + (0.1 if ctype == "melanoma" else 0.0)
    # ALK fusion → ALK inhibitor.
    if g("ALK_fusion"):
        s["Crizotinib"] += 0.80
    # BRCA1/2 loss → synthetic lethality with PARP inhibition + platinum.
    brca = g("BRCA1_mut") or g("BRCA2_mut")
    if brca:
        s["Olaparib"] += 0.70
        s["Cisplatin"] += 0.30
        s["Carboplatin"] += 0.30
    # MSI-high / high TMB → checkpoint-inhibitor benefit.
    if g("MSI_high"):
        s["Pembrolizumab"] += 0.55
    s["Pembrolizumab"] += 0.020 * c("TMB")          # graded by TMB

    # --- Resistance / sensitivity modifiers (continuous expression) ---------
    # ERCC1 high → platinum resistance.
    s["Cisplatin"]   -= 0.45 * c("ERCC1_expr")
    s["Carboplatin"] -= 0.45 * c("ERCC1_expr")
    # TUBB3 high → taxane resistance.
    s["Paclitaxel"] -= 0.45 * c("TUBB3_expr")
    s["Docetaxel"]  -= 0.45 * c("TUBB3_expr")
    # ABCB1 (MDR1) high → broad efflux-mediated resistance to many cytotoxics.
    for d in ["Paclitaxel", "Docetaxel", "Doxorubicin", "Irinotecan", "Vemurafenib"]:
        s[d] -= 0.35 * c("ABCB1_expr")
    # High proliferation → more sensitive to antimetabolites / cycle-active drugs.
    for d in ["Gemcitabine", "Fluorouracil", "Doxorubicin", "Paclitaxel"]:
        s[d] += 0.25 * c("proliferation")
    # KRAS mutation → reduced benefit from EGFR-TKI.
    if g("KRAS_mut"):
        s["Erlotinib"] -= 0.40
    # TP53 mutation → modest anthracycline sensitivity shift.
    if g("TP53_mut"):
        s["Doxorubicin"] += 0.12
    # PTEN loss / PIK3CA → modest taxane sensitivity changes.
    if g("PTEN_loss"):
        s["Paclitaxel"] -= 0.10

    # --- Cancer-type backbone regimens (standard-of-care priors) ------------
    backbone = {
        "ovarian":     {"Carboplatin": 0.30, "Paclitaxel": 0.25},
        "lung_nsclc":  {"Cisplatin": 0.22, "Pemetrexed": 0.0, "Docetaxel": 0.18},
        "breast":      {"Doxorubicin": 0.20, "Paclitaxel": 0.20},
        "colorectal":  {"Fluorouracil": 0.30, "Irinotecan": 0.25},
        "pancreatic":  {"Gemcitabine": 0.35, "Fluorouracil": 0.18},
        "melanoma":    {"Pembrolizumab": 0.25},
        "gastric":     {"Fluorouracil": 0.25, "Cisplatin": 0.20},
        "glioma":      {"Temozolomide": 0.55},
    }.get(ctype, {})
    for d, bonus in backbone.items():
        if d in s:
            s[d] += bonus

    # ER-positive breast tumors are typically less chemo-driven overall.
    if ctype == "breast" and c("ER_expr") > 0.5:
        for d in ["Doxorubicin", "Paclitaxel", "Gemcitabine"]:
            s[d] -= 0.12 * c("ER_expr")

    return s
