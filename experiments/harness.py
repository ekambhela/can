"""
Evaluation harness for Karkive accuracy experiments.

Methodology (enforced here so every experiment obeys it):
  * Splits are BY CELL LINE — a line is never in both train and eval.
  * All target scaling / feature-prevalence stats are fit on TRAIN ONLY.
  * Model / feature choices are made on the VALIDATION split; TEST is looked at
    only at milestones as the honest final estimate.
  * Metrics: mean per-drug Spearman, mean per-drug R^2, top-1/3/10 accuracy,
    best_drug_percentile — all computed in the same train-z-scored sensitivity
    space so per-drug and multi-task models are directly comparable.

Target: sensitivity = -z(logIC50), where the z uses the drug's TRAIN mean/std.
Higher = more sensitive (lower IC50).
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from model.gdsc import (
    DATA_DIR,
    DRUG_SOURCE,
    DRUGS,
    ERBB2_AMP_COL,
    MUTATION_FEATURES,
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_universe() -> dict:
    """Load the 988-cell-line modeling universe once.

    Returns a dict with:
      feats      : DataFrame (n x [tissue + all 677 binary genomic features])
      curated    : list[str] curated binary feature columns (production set)
      full_bin   : list[str] all binary genomic feature columns
      tissue_vals: sorted unique tissues (one-hot vocabulary)
      targets_raw: {drug_id: np.array(n) raw replicate-averaged logIC50, nan=unobs}
      tissue_arr : np.array(n) tissue string per line (for tissue-blocked CV)
      drug_ids   : list[int] canonical drug ids
    """
    ic = pd.read_csv(os.path.join(DATA_DIR, "IC50_v17.csv.gz"))
    gf = pd.read_csv(os.path.join(DATA_DIR, "genomic_features_v17.csv.gz"))
    df = gf.merge(ic, on="COSMIC_ID")
    g2 = os.path.join(DATA_DIR, "IC50_gdsc2.csv.gz")
    if os.path.exists(g2):
        df = df.merge(pd.read_csv(g2), on="COSMIC_ID", how="left")
    df = df.reset_index(drop=True)
    n = len(df)

    full_bin = [c for c in gf.columns
                if c.endswith("_mut") or c.startswith("gain_") or c.startswith("loss_")]
    # build all columns at once (avoids DataFrame fragmentation)
    parts = {c: df[c].fillna(0).astype(np.float32) for c in full_bin}
    parts["MSI"] = df["MSI_FACTOR"].fillna(0).astype(np.float32)
    parts["ERBB2_amp"] = df[ERBB2_AMP_COL].fillna(0).astype(np.float32)
    parts["tissue"] = df["TISSUE_FACTOR"].astype(str)
    feats = pd.DataFrame(parts)
    curated = list(MUTATION_FEATURES) + ["ERBB2_amp", "MSI"]

    targets_raw = {}
    for cid in DRUGS:
        cols = [f"Drug_{sid}_IC50" for sid in DRUG_SOURCE[cid]]
        raw = df[cols].astype(float).mean(axis=1)  # replicate-averaged raw logIC50
        targets_raw[cid] = raw.to_numpy()

    return {
        "feats": feats, "curated": curated, "full_bin": full_bin,
        "tissue_vals": sorted(df["TISSUE_FACTOR"].astype(str).unique().tolist()),
        "targets_raw": targets_raw, "tissue_arr": df["TISSUE_FACTOR"].astype(str).to_numpy(),
        "cosmic_arr": df["COSMIC_ID"].astype(int).to_numpy(),
        "drug_ids": list(DRUGS.keys()), "n": n,
    }


# ---------------------------------------------------------------------------
# Gene-expression features (optional add-on; see run_expression.py)
# ---------------------------------------------------------------------------
EXPR_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "expr_cosmic.csv.gz")


def load_expression(cosmic_arr):
    """Return an (n x 706) expression DataFrame aligned to the universe row order,
    NaN where a line has no expression, plus a boolean mask of covered rows."""
    expr = pd.read_csv(EXPR_PATH, index_col=0)
    expr.index = expr.index.astype(int)
    aligned = expr.reindex(cosmic_arr).reset_index(drop=True)
    aligned.columns = list(expr.columns)
    mask = aligned.notna().any(axis=1).to_numpy()
    return aligned.astype(np.float32), mask


def expression_pca(expr_df, train_idx, k=50):
    """Fit standardization + PCA on TRAIN rows only, transform all rows.

    706 raw genes overfit the small per-drug models (like the 677 mutation flags
    did), so we compress to k principal components. Fitting on train only keeps
    the reduction leakage-free. Returns a DataFrame of pc columns (NaN rows stay
    NaN so tree models treat them as missing)."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    X = expr_df.to_numpy(dtype=np.float32)
    row_ok = ~np.isnan(X).any(axis=1)
    tr_ok = np.array([i for i in train_idx if row_ok[i]])
    scaler = StandardScaler().fit(X[tr_ok])
    pca = PCA(n_components=k, random_state=0).fit(scaler.transform(X[tr_ok]))
    out = np.full((len(X), k), np.nan, dtype=np.float32)
    out[row_ok] = pca.transform(scaler.transform(X[row_ok])).astype(np.float32)
    cols = [f"pc{i}" for i in range(k)]
    return pd.DataFrame(out, columns=cols)


# ---------------------------------------------------------------------------
# Splits (by cell line)
# ---------------------------------------------------------------------------
def random_splits(n: int, seed: int = 0, fracs=(0.64, 0.16, 0.20)):
    """train / val / test indices, split by cell line."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    a = int(fracs[0] * n)
    b = int((fracs[0] + fracs[1]) * n)
    return idx[:a], idx[a:b], idx[b:]


def tissue_blocked_split(tissue_arr, seed: int = 0, test_frac: float = 0.25):
    """Hold out WHOLE tissues. Returns (trainval_idx, test_idx) where no tissue
    appears in both — a harder test of learned biology vs tissue memorization."""
    rng = np.random.default_rng(seed)
    tissues = sorted(set(tissue_arr.tolist()))
    rng.shuffle(tissues)
    n = len(tissue_arr)
    test_t, cum = set(), 0
    for t in tissues:
        c = int((tissue_arr == t).sum())
        if cum < test_frac * n:
            test_t.add(t)
            cum += c
    test_idx = np.array([i for i in range(n) if tissue_arr[i] in test_t])
    train_idx = np.array([i for i in range(n) if tissue_arr[i] not in test_t])
    return train_idx, test_idx


# ---------------------------------------------------------------------------
# Target scaling (train-only) + truth
# ---------------------------------------------------------------------------
def ztarget(y_raw: np.ndarray, train_idx: np.ndarray):
    """Sensitivity target: -(logIC50 - mu)/sd, mu/sd from TRAIN observations."""
    v = y_raw[train_idx]
    v = v[~np.isnan(v)]
    mu = float(v.mean()) if v.size else 0.0
    sd = float(v.std())
    sd = sd if sd > 1e-9 else 1.0
    return -(y_raw - mu) / sd, mu, sd


def build_truth(targets_raw: dict, drug_ids, train_idx):
    """Per-drug full-length sensitivity arrays (test lines scaled by TRAIN stats)."""
    return {d: ztarget(targets_raw[d], train_idx)[0] for d in drug_ids}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(pred: dict, truth: dict, drug_ids, eval_idx) -> dict:
    """pred[d][line] = predicted sensitivity; truth[d] = full sensitivity array."""
    eval_set = list(eval_idx)
    per_rho, per_r2 = [], []
    for d in drug_ids:
        pr = pred.get(d)
        if not pr:
            continue
        t = truth[d]
        xs, ys = [], []
        for i in eval_set:
            p = pr.get(i)
            if p is not None and not np.isnan(t[i]):
                xs.append(p)
            ys.append(t[i])
        if len(xs) >= 5 and np.std(ys) > 1e-9 and np.std(xs) > 1e-9:
            rho = spearmanr(ys, xs).correlation
            per_rho.append(0.0 if np.isnan(rho) else float(rho))
            per_r2.append(float(r2_score(ys, xs)))

    hits = hits3 = hits10 = tot = 0
    pcts = []
    for i in eval_set:
        avail = [d for d in drug_ids
                 if d in pred and pred[d].get(i) is not None and not np.isnan(truth[d][i])]
        if len(avail) < 5:
            continue
        best = max(avail, key=lambda d: truth[d][i])
        order = sorted(avail, key=lambda d: pred[d][i], reverse=True)
        tot += 1
        hits += order[0] == best
        hits3 += best in order[:3]
        hits10 += best in order[:10]
        pcts.append(1 - order.index(best) / (len(order) - 1))

    return {
        "mean_spearman": round(float(np.mean(per_rho)), 4) if per_rho else 0.0,
        "mean_r2": round(float(np.mean(per_r2)), 4) if per_r2 else 0.0,
        "top1": round(hits / tot, 4) if tot else 0.0,
        "top3": round(hits3 / tot, 4) if tot else 0.0,
        "top10": round(hits10 / tot, 4) if tot else 0.0,
        "best_drug_percentile": round(float(np.mean(pcts)), 4) if pcts else 0.0,
        "n_drugs_scored": len(per_rho),
        "n_eval_lines": tot,
    }


def fmt(tag: str, m: dict) -> str:
    return (f"{tag:<34} rho={m['mean_spearman']:.3f}  R2={m['mean_r2']:.3f}  "
            f"top1={m['top1']:.3f} top3={m['top3']:.3f} top10={m['top10']:.3f}  "
            f"pct={m['best_drug_percentile']:.3f}  (drugs={m['n_drugs_scored']}, lines={m['n_eval_lines']})")


# ---------------------------------------------------------------------------
# Per-drug model runner (the baseline architecture)
# ---------------------------------------------------------------------------
def make_pipeline(tissue_vals, binary_cols, **gbm):
    params = {"max_iter": 170, "learning_rate": 0.07, "l2_regularization": 1.5,
                  "max_leaf_nodes": 13, "early_stopping": True, "validation_fraction": 0.12,
                  "random_state": 0}
    params.update(gbm)
    pre = ColumnTransformer([
        ("tissue", OneHotEncoder(categories=[tissue_vals], handle_unknown="ignore"), ["tissue"]),
        ("bin", "passthrough", binary_cols),
    ])
    return Pipeline([("pre", pre), ("gbm", HistGradientBoostingRegressor(**params))])


def run_perdrug(U, train_idx, eval_idx, binary_cols, truth=None, **gbm) -> dict:
    """Train one model per drug on `train_idx`, predict `eval_idx`. Returns pred dict."""
    feats, drug_ids, tissue_vals = U["feats"], U["drug_ids"], U["tissue_vals"]
    if truth is None:
        truth = build_truth(U["targets_raw"], drug_ids, train_idx)
    cols = binary_cols + ["tissue"]
    Xall = feats[cols]
    pred = {}
    for d in drug_ids:
        yraw = U["targets_raw"][d]
        obs = ~np.isnan(yraw)
        tr = np.array([i for i in train_idx if obs[i]])
        ev = np.array([i for i in eval_idx if obs[i]])
        if tr.size < 10 or ev.size < 1:
            continue
        y = truth[d]
        pipe = make_pipeline(tissue_vals, binary_cols, **gbm)
        pipe.fit(Xall.iloc[tr], y[tr])
        p = pipe.predict(Xall.iloc[ev])
        pred[d] = {int(i): float(pp) for i, pp in zip(ev, p, strict=True)}
    return pred


# Pathway groupings (gene symbols); we OR the matching *_mut flags into one
# "pathway active" feature so sparse single-gene signal becomes denser.
PATHWAYS = {
    "pw_RTK_RAS_RAF": ["EGFR", "ERBB2", "ERBB3", "ERBB4", "ALK", "MET", "KRAS",
                       "NRAS", "HRAS", "BRAF", "RAF1", "FLT3", "KIT", "PDGFRA",
                       "FGFR1", "FGFR2", "FGFR3", "RET", "ROS1", "MAP2K1",
                       "MAPK1", "NF1", "PTPN11", "SOS1", "CBL"],
    "pw_PI3K_AKT": ["PIK3CA", "PIK3R1", "PIK3CB", "PTEN", "AKT1", "AKT2", "AKT3",
                    "MTOR", "TSC1", "TSC2", "RICTOR", "RPTOR", "STK11", "INPP4B"],
    "pw_CELL_CYCLE": ["CDKN2A", "CDKN2B", "RB1", "CCND1", "CCNE1", "CDK4", "CDK6",
                      "MYC", "MYCN", "E2F3", "CDKN1A", "CDKN1B"],
    "pw_P53_DNA_REPAIR": ["TP53", "MDM2", "MDM4", "ATM", "ATR", "BRCA1", "BRCA2",
                          "PALB2", "RAD51", "CHEK1", "CHEK2", "MLH1", "MSH2",
                          "MSH6", "PMS2", "FANCA"],
    "pw_WNT": ["APC", "CTNNB1", "AXIN1", "AXIN2", "RNF43", "TCF7L2"],
    "pw_CHROMATIN": ["ARID1A", "ARID1B", "SMARCA4", "SMARCB1", "KMT2D", "KMT2C",
                     "EP300", "CREBBP", "PBRM1", "SETD2", "KDM6A", "BAP1"],
    "pw_NOTCH": ["NOTCH1", "NOTCH2", "NOTCH3", "FBXW7"],
    "pw_TGFB": ["SMAD4", "SMAD2", "TGFBR2", "ACVR2A"],
}


def pathway_features(feats: pd.DataFrame):
    """Add pathway-active OR flags. Returns (feats_with_pw, pathway_col_names)."""
    new = {}
    for pw, genes in PATHWAYS.items():
        cols = [f"{g}_mut" for g in genes if f"{g}_mut" in feats.columns]
        if cols:
            new[pw] = (feats[cols].to_numpy(dtype=np.float32).max(axis=1))
    out = pd.concat([feats, pd.DataFrame(new, index=feats.index)], axis=1)
    return out, list(new.keys())


def prevalence_filter(feats, binary_cols, train_idx, min_frac=0.01):
    """Keep binary features present in >= min_frac of TRAIN lines (train-only stat)."""
    sub = feats.iloc[train_idx][binary_cols].to_numpy(dtype=np.float32)
    keep_mask = sub.mean(axis=0) >= min_frac
    return [c for c, k in zip(binary_cols, keep_mask, strict=True) if k]
