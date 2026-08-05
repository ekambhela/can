import warnings

from scipy.stats import ConstantInputWarning

from experiments.harness import build_truth, compute_metrics, fmt, load_universe, random_splits
from experiments.mf import run_mf

warnings.simplefilter("ignore", ConstantInputWarning)
U = load_universe()
tr, va, te = random_splits(U["n"], seed=0)
truth = build_truth(U["targets_raw"], U["drug_ids"], tr)
ids = U["drug_ids"]
print(f"train={len(tr)} val={len(va)} test={len(te)}\n")
for K in (8, 16, 32):
    pred = run_mf(U, tr, va, U["curated"], truth=truth, K=K, epochs=25, lr=0.02, reg=0.02)
    print(fmt(f"MF K={K} [VAL]", compute_metrics(pred, truth, ids, va)))
