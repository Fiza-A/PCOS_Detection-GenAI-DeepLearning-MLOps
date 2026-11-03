"""Pairwise statistical association preservation; not clinical causality."""
import json

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.metrics import normalized_mutual_info_score

from phase2_data import ROOT, read_partition, CATEGORICAL, load_config


def association_matrix(df: pd.DataFrame, categorical: list[str]) -> pd.DataFrame:
    cols=list(df.columns); out=pd.DataFrame(np.eye(len(cols)),index=cols,columns=cols)
    for i,a in enumerate(cols):
        for b in cols[i+1:]:
            if a not in categorical and b not in categorical:
                x=pd.to_numeric(df[a],errors="coerce"); y=pd.to_numeric(df[b],errors="coerce"); v=x.corr(y,method="pearson")
            elif a in categorical and b in categorical:
                v=normalized_mutual_info_score(df[a].astype(str),df[b].astype(str))
            else:
                cat, num=(a,b) if a in categorical else (b,a)
                groups=pd.to_numeric(df[num],errors="coerce"); labels=df[cat].astype(str)
                valid=groups.notna(); overall=groups[valid].mean(); between=sum((groups[valid & (labels==k)].count())*(groups[valid & (labels==k)].mean()-overall)**2 for k in labels.unique())
                total=((groups[valid]-overall)**2).sum(); v=between/total if total else 0
            out.loc[a,b]=out.loc[b,a]=float(v) if pd.notna(v) else 0.0
    return out


def evaluate(generator: str) -> dict:
    g=generator.lower(); real=read_partition("train"); categorical=CATEGORICAL+[load_config()["target_column"]]
    syn=pd.read_csv(ROOT/f"GENAI/data/synthetic/{g}/synthetic_{g}.csv",dtype={c:str for c in categorical})
    a=association_matrix(real,categorical); b=association_matrix(syn,categorical); diff=(a-b).abs()
    pairs=[(float(diff.iloc[i,j]),a.index[i],a.columns[j],float(a.iloc[i,j]),float(b.iloc[i,j])) for i in range(len(a)) for j in range(i+1,len(a))]
    pairs.sort(reverse=True)
    result={"generator":g.upper(),"method":"Pearson for numerical-numerical; normalized mutual information for categorical-categorical; correlation ratio eta-squared for mixed pairs","aggregate_mean_absolute_association_error":float(np.mean([p[0] for p in pairs])),"aggregate_association_similarity_1_minus_error":float(1-np.mean([p[0] for p in pairs])),"largest_mismatches":[{"feature_a":x,"feature_b":y,"real_training_association":u,"synthetic_association":v,"absolute_difference":d} for d,x,y,u,v in pairs[:20]],"causality_note":"Associations are statistical summaries and do not imply clinical causality."}
    out=ROOT/f"GENAI/reports/{g}"; out.mkdir(parents=True,exist_ok=True); a.to_csv(out/"real_association_matrix.csv"); b.to_csv(out/"synthetic_association_matrix.csv"); (out/"correlation.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    return result
