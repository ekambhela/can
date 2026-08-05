import warnings

from scipy.stats import ConstantInputWarning

from experiments.harness import (
    build_truth,
    compute_metrics,
    fmt,
    load_universe,
    random_splits,
    run_perdrug,
)
from experiments.multitask import build_drug_features, run_multitask

warnings.simplefilter("ignore", ConstantInputWarning)

def blend(a, b, w=0.5):
    out={}
    for d in set(a)|set(b):
        da=a.get(d,{})
        db=b.get(d,{})
        keys=set(da)&set(db)
        if keys:
            out[d]={i: w*da[i]+(1-w)*db[i] for i in keys}
    return out

U=load_universe()
tr,va,te=random_splits(U["n"],seed=0)
truth=build_truth(U["targets_raw"],U["drug_ids"],tr)
ids=U["drug_ids"]
dfid=build_drug_features(ids,truth=truth,train_idx=tr,use_id=True)
best={"max_leaf_nodes": 127,"learning_rate": 0.05,"max_iter": 600,"l2_regularization": 1.0}
for split,name in [(va,"VAL"),(te,"TEST")]:
    pd_=run_perdrug(U,tr,split,U["curated"],truth=truth)
    mt_,_=run_multitask(U,tr,split,U["curated"],truth=truth,drug_feat=dfid,**best)
    print(fmt(f"per-drug [{name}]",compute_metrics(pd_,truth,ids,split)))
    print(fmt(f"multitask [{name}]",compute_metrics(mt_,truth,ids,split)))
    for w in (0.5,0.35):
        print(fmt(f"ENSEMBLE w={w} [{name}]",compute_metrics(blend(pd_,mt_,w),truth,ids,split)))
    print()
