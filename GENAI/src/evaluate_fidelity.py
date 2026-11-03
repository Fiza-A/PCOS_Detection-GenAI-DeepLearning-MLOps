"""Univariate fidelity against training partition only."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance
from scipy.spatial.distance import jensenshannon

from phase2_data import ROOT, read_partition, CATEGORICAL, load_config

BASE = ROOT / "GENAI/data/processed"; OUT = ROOT / "GENAI/reports"


def evaluate(generator: str) -> dict:
    g=generator.lower(); real=read_partition("train"); categorical=CATEGORICAL+[load_config()["target_column"]]
    syn=pd.read_csv(ROOT/f"GENAI/data/synthetic/{g}/synthetic_{g}.csv",dtype={c:str for c in categorical})
    numeric=[c for c in real if c not in categorical]
    num_scores={}
    for c in numeric:
        a=pd.to_numeric(real[c],errors="coerce").dropna(); b=pd.to_numeric(syn[c],errors="coerce").dropna()
        ks=float(ks_2samp(a,b).statistic); num_scores[c]={"ks_statistic":ks,"one_minus_ks_similarity":1-ks,"wasserstein_distance":float(wasserstein_distance(a,b)),"real_mean":float(a.mean()),"synthetic_mean":float(b.mean()),"real_std":float(a.std()),"synthetic_std":float(b.std())}
    cat_scores={}
    for c in categorical:
        levels=sorted(set(real[c].astype(str))|set(syn[c].astype(str)))
        p=real[c].astype(str).value_counts(normalize=True).reindex(levels,fill_value=0).to_numpy(); q=syn[c].astype(str).value_counts(normalize=True).reindex(levels,fill_value=0).to_numpy()
        js=float(jensenshannon(p,q,base=2)); cat_scores[c]={"jensen_shannon_distance":js,"js_similarity":1-js,"real_levels":{str(k):int(v) for k,v in real[c].value_counts().items()},"synthetic_levels":{str(k):int(v) for k,v in syn[c].value_counts().items()}}
    num_mean=float(np.mean([x["one_minus_ks_similarity"] for x in num_scores.values()])) if num_scores else None
    cat_mean=float(np.mean([x["js_similarity"] for x in cat_scores.values()])) if cat_scores else None
    result={"generator":g.upper(),"reference":"real training partition only","numerical_feature_scores":num_scores,"categorical_feature_scores":cat_scores,"aggregate_mean_one_minus_ks":num_mean,"aggregate_mean_js_similarity":cat_mean,"weakest_numeric_features":sorted(num_scores,key=lambda x:num_scores[x]["one_minus_ks_similarity"])[:5],"strongest_numeric_features":sorted(num_scores,key=lambda x:num_scores[x]["one_minus_ks_similarity"],reverse=True)[:5],"weakest_categorical_features":sorted(cat_scores,key=lambda x:cat_scores[x]["js_similarity"])[:5],"strongest_categorical_features":sorted(cat_scores,key=lambda x:cat_scores[x]["js_similarity"],reverse=True)[:5]}
    out=OUT/g; out.mkdir(parents=True,exist_ok=True); (out/"fidelity.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    return result
