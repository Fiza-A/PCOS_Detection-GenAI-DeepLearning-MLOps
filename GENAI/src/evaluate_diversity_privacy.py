"""Diversity and preliminary record-overlap/nearest-neighbor risk indicators."""
import json

import numpy as np
import pandas as pd

from phase2_data import ROOT, read_partition, CATEGORICAL, load_config


def _distances(a: pd.DataFrame,b: pd.DataFrame, categorical: list[str], reference: pd.DataFrame) -> np.ndarray:
    # Gower-like normalized absolute difference / categorical mismatch, fitted to training support.
    num=[c for c in a if c not in categorical]; spans={c:float(reference[c].max()-reference[c].min()) or 1.0 for c in num}
    d=np.zeros((len(a),len(b)),dtype=float)
    for c in a.columns:
        if c in categorical:
            x=a[c].astype(str).to_numpy(); y=b[c].astype(str).to_numpy(); d += (x[:,None]!=y[None,:])
        else:
            x=pd.to_numeric(a[c],errors="coerce").to_numpy(dtype=float); y=pd.to_numeric(b[c],errors="coerce").to_numpy(dtype=float); d += np.abs(x[:,None]-y[None,:])/spans[c]
    return d/max(len(a.columns),1)


def evaluate(generator: str) -> dict:
    g=generator.lower(); target=load_config()["target_column"]; categorical=CATEGORICAL+[target]
    train=read_partition("train"); syn=pd.read_csv(ROOT/f"GENAI/data/synthetic/{g}/synthetic_{g}.csv",dtype={c:str for c in categorical})
    val=read_partition("validation"); test=read_partition("test")
    key=lambda d:d.astype(str).agg("\x1f".join,axis=1)
    syn_keys=key(syn); duplicate_rows=int(syn.duplicated().sum()); train_set=set(key(train)); validation_set=set(key(val)); test_set=set(key(test))
    combo_cols=[c for c in categorical if c != load_config()["target_column"]]
    combo_counts=syn[combo_cols].astype(str).value_counts()
    dtrain=_distances(syn,train,categorical,train); dval=_distances(syn,val,categorical,train); dtest=_distances(syn,test,categorical,train)
    # Descriptive reference: leave-one-out nearest distances among training records.
    dloo=_distances(train,train,categorical,train); np.fill_diagonal(dloo,np.inf); ref_nearest=dloo.min(axis=1); threshold=float(np.quantile(ref_nearest,.01))
    nearest=dtrain.min(axis=1)
    result={"generator":g.upper(),"synthetic_rows":len(syn),"unique_synthetic_rows":int(syn_keys.nunique()),"duplicate_rows":duplicate_rows,"duplicate_rate":float(duplicate_rows/max(len(syn),1)),"categorical_non_target_combination_count":int(len(combo_counts)),"categorical_combination_duplicate_rows":int(len(syn)-len(combo_counts)),"most_repeated_categorical_combination_count":int(combo_counts.max()) if len(combo_counts) else 0,"exact_record_overlap_training":int(syn_keys.isin(train_set).sum()),"exact_record_overlap_validation":int(syn_keys.isin(validation_set).sum()),"exact_record_overlap_test":int(syn_keys.isin(test_set).sum()),"nearest_training_distance_quantiles":{str(q):float(np.quantile(nearest,q)) for q in [0,.01,.05,.25,.5,.75,1]},"training_leave_one_out_nearest_distance_quantiles":{str(q):float(np.quantile(ref_nearest,q)) for q in [0,.01,.05,.5,.95,1]},"synthetic_below_training_loo_1pct_reference_count":int((nearest<threshold).sum()),"synthetic_nearest_training_record_indices_top10":np.argsort(nearest)[:10].tolist(),"validation_nearest_distance_min":float(dval.min(axis=1).min()),"test_nearest_distance_min":float(dtest.min(axis=1).min()),"method":"Gower-like distance: numeric absolute difference scaled by training range, categorical mismatch; descriptive indicator only","privacy_claim":"Preliminary privacy-risk assessment only; no anonymity or formal differential-privacy guarantee."}
    out=ROOT/f"GENAI/reports/{g}"; (out/"diversity_privacy.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    return result
