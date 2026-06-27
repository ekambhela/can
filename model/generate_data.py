"""
Generate a biologically plausible pharmacogenomic training set.

Each row is a simulated tumor sample (genomic alterations + expression /
clinical features). For every sample we compute a latent sensitivity score
for every therapy via the rules in biomarkers.py, add realistic measurement
noise, and emit per-drug response labels.

Outputs (written to artifacts/):
  * dataset_features.csv  : one row per sample, the model inputs
  * dataset_response.csv  : one row per sample, sensitivity (0-1) per drug

The "ground-truth best therapy" for a sample is simply the argmax of its
noise-free sensitivity vector; the trained model is evaluated on how often
it recovers that choice.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from .biomarkers import (
    CANCER_TYPES,
    CONTINUOUS_FEATURES,
    CONTINUOUS_PRIORS,
    MUTATION_FEATURES,
    THERAPY_NAMES,
    sensitivity_scores,
)

# Per-cancer-type prevalence of each genomic alteration. Values are rough,
# literature-inspired prevalences so that, e.g., EGFR mutations cluster in
# lung adenocarcinoma and BRCA loss in ovarian/breast tumors.
MUT_PREVALENCE = {
    "breast":     {"TP53_mut": .35, "HER2_amp": .20, "BRCA1_mut": .06, "BRCA2_mut": .06,
                   "PIK3CA_mut": .35, "PTEN_loss": .12, "KRAS_mut": .02},
    "lung_nsclc": {"TP53_mut": .50, "KRAS_mut": .30, "EGFR_mut": .20, "ALK_fusion": .05,
                   "BRAF_V600": .03, "PTEN_loss": .08},
    "ovarian":    {"TP53_mut": .70, "BRCA1_mut": .15, "BRCA2_mut": .10, "PIK3CA_mut": .12,
                   "PTEN_loss": .07},
    "colorectal": {"TP53_mut": .55, "KRAS_mut": .42, "BRAF_V600": .10, "PIK3CA_mut": .18,
                   "MSI_high": .15},
    "melanoma":   {"BRAF_V600": .50, "TP53_mut": .18, "PTEN_loss": .20, "KRAS_mut": .02,
                   "MSI_high": .05},
    "pancreatic": {"KRAS_mut": .90, "TP53_mut": .65, "BRCA2_mut": .07, "PTEN_loss": .05},
    "gastric":    {"TP53_mut": .45, "HER2_amp": .15, "MSI_high": .20, "PIK3CA_mut": .12},
    "glioma":     {"TP53_mut": .40, "PTEN_loss": .30, "EGFR_mut": .25, "BRAF_V600": .05},
}

DEFAULT_PREV = 0.05  # fallback prevalence for any (type, gene) not listed


def _sample_one(rng: np.random.Generator, ctype: str) -> dict:
    """Draw a single random-but-plausible tumor sample for a cancer type."""
    row = {"cancer_type": ctype}
    prev = MUT_PREVALENCE.get(ctype, {})
    for g in MUTATION_FEATURES:
        p = prev.get(g, DEFAULT_PREV)
        row[g] = int(rng.random() < p)

    # MSI-high tumors carry many mutations → elevate TMB.
    for f in CONTINUOUS_FEATURES:
        mean, sd, lo, hi = CONTINUOUS_PRIORS[f]
        val = rng.normal(mean, sd)
        row[f] = float(np.clip(val, lo, hi))
    if row["MSI_high"]:
        row["TMB"] = float(np.clip(row["TMB"] + rng.normal(20, 5), 0, 40))

    # ER expression is really only meaningful in breast; zero it elsewhere-ish.
    if ctype != "breast":
        row["ER_expr"] = float(np.clip(row["ER_expr"] * 0.3, 0, 1))
    return row


def generate(n_samples: int = 9000, seed: int = 7) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build feature + response frames for `n_samples` simulated tumors."""
    rng = np.random.default_rng(seed)
    feats, resp = [], []

    for _ in range(n_samples):
        ctype = CANCER_TYPES[rng.integers(len(CANCER_TYPES))]
        sample = _sample_one(rng, ctype)

        scores = sensitivity_scores(sample)
        # Squash latent scores to (0,1) and add heteroscedastic assay noise.
        vec = {}
        for drug in THERAPY_NAMES:
            base = 1.0 / (1.0 + np.exp(-(scores[drug] - 0.5) * 3.0))  # logistic squash
            noisy = base + rng.normal(0, 0.05)
            vec[drug] = float(np.clip(noisy, 0.0, 1.0))

        feats.append(sample)
        resp.append(vec)

    fx = pd.DataFrame(feats)
    ry = pd.DataFrame(resp)
    return fx, ry


def main() -> None:
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = os.path.join(here, "artifacts")
    os.makedirs(out, exist_ok=True)
    fx, ry = generate()
    fx.to_csv(os.path.join(out, "dataset_features.csv"), index=False)
    ry.to_csv(os.path.join(out, "dataset_response.csv"), index=False)
    print(f"Wrote {len(fx)} samples -> {out}")
    print("Feature columns:", list(fx.columns))
    print("Therapies:", list(ry.columns))


if __name__ == "__main__":
    main()
