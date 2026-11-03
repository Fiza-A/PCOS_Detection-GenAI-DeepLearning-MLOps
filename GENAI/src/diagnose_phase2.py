"""Read-only diagnostics over saved Phase 2 data/model/report artifacts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from phase2_data import ROOT, CATEGORICAL, load_config, read_partition
from evaluate_diversity_privacy import _distances

OUT = ROOT / "GENAI/reports/diagnostics"
GENERATORS = ("ctgan", "tvae")
QUANTILES = (.01, .05, .25, .5, .75, .95, .99)


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify_lineage() -> dict:
    split = _json(ROOT / "GENAI/data/processed/split_metadata.json")
    idx = pd.read_csv(ROOT / "GENAI/data/processed/split_indices.csv")
    groups = {k: set(v["source_row_index_zero_based"]) for k, v in idx.groupby("partition")}
    parts = {"train": read_partition("train"), "validation": read_partition("validation"), "test": read_partition("test")}
    raw = ROOT / "GENAI/data/raw/PCOS_data.csv"
    raw_hash = hashlib.sha256(raw.read_bytes()).hexdigest()
    model_meta_path = ROOT / "GENAI/models/shared/pcos_metadata.json"
    schema = _json(model_meta_path)
    columns = schema["tables"]["pcos"]["columns"]
    reports = {}
    for g in GENERATORS:
        meta = _json(ROOT / f"GENAI/data/synthetic/{g}/run_metadata.json")
        synth_path = ROOT / f"GENAI/data/synthetic/{g}/synthetic_{g}.csv"
        model_path = ROOT / f"GENAI/models/{g}/{g}_synthesizer.pkl"
        train_hash = hashlib.sha256((ROOT / "GENAI/data/processed/train.csv").read_bytes()).hexdigest()
        expected_class = "CTGANSynthesizer" if g == "ctgan" else "TVAESynthesizer"
        import_ok = False
        if model_path.is_file():
            from sdv.single_table import CTGANSynthesizer, TVAESynthesizer
            cls = CTGANSynthesizer if g == "ctgan" else TVAESynthesizer
            loaded = cls.load(filepath=str(model_path))
            import_ok = type(loaded).__name__ == expected_class
        reports[g] = {
            "generator_type_from_run_metadata": meta.get("generator"),
            "generator_class_verified_by_loading_saved_model": expected_class if import_ok else "NOT VERIFIABLE FROM SAVED ARTIFACTS",
            "seed": meta.get("seed"), "training_rows_recorded": meta.get("training_rows"),
            "synthetic_rows_recorded": meta.get("synthetic_rows"),
            "synthetic_rows_actual": len(pd.read_csv(synth_path)),
            "model_configuration": meta.get("config"), "model_path_exists": model_path.is_file(),
            "synthetic_output_path": str(synth_path.relative_to(ROOT)),
            "metadata_schema_path": meta.get("metadata_path"),
            "metadata_schema_version": meta.get("metadata_schema_version"),
            "schema_columns": len(columns), "identifier_columns_in_schema": sorted(set(load_config()["identifier_columns"]) & set(columns)),
            "train_partition_hash_matches_metadata": meta.get("training_partition_sha256") == train_hash,
            "raw_dataset_hash_matches_metadata": meta.get("raw_dataset_sha256") == raw_hash,
            "metadata_claims_validation_test_not_used": meta.get("validation_or_test_passed_to_fit") is False,
            "fit_partition_claim": meta.get("fit_data_source"),
            "fit_dataset_independently_provable_from_saved_model": False,
            "fit_lineage_verification": "NOT VERIFIABLE FROM SAVED ARTIFACTS" if not (meta.get("fit_data_source") == "GENAI/data/processed/train.csv only" and meta.get("training_partition_sha256") == train_hash and meta.get("validation_or_test_passed_to_fit") is False) else "metadata and train hash consistent; serialized model does not independently encode fit-row lineage",
        }
        # Recalculate exact overlaps from saved data, using CSV parsing consistently.
        synthetic = pd.read_csv(synth_path)
        def row_keys(frame):
            return frame.astype("string").fillna("<NA>").agg("\x1f".join, axis=1)
        synthetic_keys = row_keys(synthetic)
        reports[g]["exact_record_overlap_recomputed"] = {
            part: int(synthetic_keys.isin(set(row_keys(pd.read_csv(ROOT / f"GENAI/data/processed/{part}.csv")))).sum())
            for part in ("train", "validation", "test")
        }
    return {
        "raw_rows": sum(split["row_counts"].values()), "split_row_counts_metadata": split["row_counts"],
        "split_row_counts_indices": {k: len(v) for k,v in groups.items()},
        "pairwise_split_intersections": {"train_validation": len(groups["train"] & groups["validation"]), "train_test": len(groups["train"] & groups["test"]), "validation_test": len(groups["validation"] & groups["test"])},
        "split_indices_cover_unique_rows": len(set.union(*groups.values())) == sum(len(v) for v in groups.values()),
        "split_indices_cover_total": len(set.union(*groups.values())) == sum(split["row_counts"].values()),
        "raw_sha256_matches_split_metadata": raw_hash == split["source_sha256"],
        "partition_actual_row_counts": {k: len(v) for k,v in parts.items()},
        "generator_lineage": reports,
    }


def numerical_diagnostics() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = read_partition("train")
    cat = set(CATEGORICAL + [load_config()["target_column"]])
    numeric = [c for c in train if c not in cat]
    fidelity = {g: _json(ROOT / f"GENAI/reports/{g}/fidelity.json") for g in GENERATORS}
    rows=[]
    for c in numeric:
        rec={"feature":c}
        for g in GENERATORS:
            syn=pd.read_csv(ROOT/f"GENAI/data/synthetic/{g}/synthetic_{g}.csv")
            a=pd.to_numeric(train[c],errors="coerce").dropna(); b=pd.to_numeric(syn[c],errors="coerce").dropna()
            ks=float(ks_2samp(a,b).statistic)
            # Check calculation against stored Phase 2 score; retain actual recomputation.
            rec[f"{g}_ks"]=ks; rec[f"{g}_similarity"]=1-ks
            rec[f"{g}_stored_similarity"]=fidelity[g]["numerical_feature_scores"][c]["one_minus_ks_similarity"]
        rec["absolute_similarity_difference"]=abs(rec["ctgan_similarity"]-rec["tvae_similarity"])
        rec["winner"]="CTGAN" if rec["ctgan_similarity"]>rec["tvae_similarity"] else "TVAE" if rec["tvae_similarity"]>rec["ctgan_similarity"] else "TIE"
        rows.append(rec)
    table=pd.DataFrame(rows).sort_values(["ctgan_similarity","tvae_similarity"],ascending=True)
    focus=set(table.nsmallest(10,"ctgan_similarity")["feature"])|set(table.nsmallest(10,"tvae_similarity")["feature"])
    summaries=[]
    for c in sorted(focus):
        for label, frame in [("REAL_TRAIN",train)]+[(g.upper(),pd.read_csv(ROOT/f"GENAI/data/synthetic/{g}/synthetic_{g}.csv")) for g in GENERATORS]:
            s=pd.to_numeric(frame[c],errors="coerce").dropna()
            q={str(q):float(s.quantile(q)) for q in QUANTILES}
            summaries.append({"feature":c,"source":label,"count":len(s),"minimum":float(s.min()),"maximum":float(s.max()),"mean":float(s.mean()),"median":float(s.median()),"std":float(s.std()),"unique_count":int(s.nunique()),"quantiles_json":json.dumps(q)})
    return table, pd.DataFrame(summaries)


def categorical_diagnostics() -> pd.DataFrame:
    cats=CATEGORICAL+[load_config()["target_column"]]
    train=read_partition("train"); rows=[]
    for c in cats:
        real=train[c].astype(str).value_counts()
        series={"REAL_TRAIN":real}
        for g in GENERATORS:
            syn=pd.read_csv(ROOT/f"GENAI/data/synthetic/{g}/synthetic_{g}.csv",dtype={c:str})
            series[g.upper()]=syn[c].astype(str).value_counts()
        levels=sorted(set().union(*(set(s.index) for s in series.values())))
        for level in levels:
            counts={name:int(s.get(level,0)) for name,s in series.items()}
            props={name:counts[name]/len(train if name=="REAL_TRAIN" else pd.read_csv(ROOT/f"GENAI/data/synthetic/{name.lower()}/synthetic_{name.lower()}.csv")) for name in series}
            rows.append({"feature":c,"category":level,"real_train_count":counts["REAL_TRAIN"],"ctgan_count":counts["CTGAN"],"tvae_count":counts["TVAE"],"real_train_percent":100*props["REAL_TRAIN"],"ctgan_percent":100*props["CTGAN"],"tvae_percent":100*props["TVAE"],"ctgan_abs_frequency_difference_pp":100*abs(props["CTGAN"]-props["REAL_TRAIN"]),"tvae_abs_frequency_difference_pp":100*abs(props["TVAE"]-props["REAL_TRAIN"]),"ctgan_missing_category":counts["CTGAN"]==0,"tvae_missing_category":counts["TVAE"]==0,"ctgan_new_category":counts["REAL_TRAIN"]==0 and counts["CTGAN"]>0,"tvae_new_category":counts["REAL_TRAIN"]==0 and counts["TVAE"]>0})
    return pd.DataFrame(rows)


def target_diagnostic() -> pd.DataFrame:
    c=load_config()["target_column"]; frames={"REAL_TRAIN":read_partition("train")}
    for g in GENERATORS: frames[g.upper()]=pd.read_csv(ROOT/f"GENAI/data/synthetic/{g}/synthetic_{g}.csv",dtype={c:str})
    real=frames["REAL_TRAIN"][c].astype(str).value_counts(); rows=[]
    for label,frame in frames.items():
        counts=frame[c].astype(str).value_counts()
        for value in sorted(set(real.index)|set(counts.index)):
            rc=int(real.get(value,0)); cc=int(counts.get(value,0)); rp=rc/len(frames["REAL_TRAIN"]); cp=cc/len(frame)
            rows.append({"partition":label,"target_value":value,"count":cc,"total":len(frame),"percent":100*cp,"real_train_count":rc,"real_train_percent":100*rp,"absolute_percentage_point_difference":100*abs(cp-rp),"relative_count_difference_percent":(cc-rc)/rc*100 if rc else None})
    return pd.DataFrame(rows)


def association_diagnostics() -> pd.DataFrame:
    real=pd.read_csv(ROOT/"GENAI/reports/ctgan/real_association_matrix.csv",index_col=0)
    mats={g:pd.read_csv(ROOT/f"GENAI/reports/{g}/synthetic_association_matrix.csv",index_col=0) for g in GENERATORS}
    rows=[]; cols=list(real.columns)
    for i,a in enumerate(cols):
        for b in cols[i+1:]:
            r=float(real.loc[a,b]); vals={g:float(mats[g].loc[a,b]) for g in GENERATORS}
            rows.append({"feature_a":a,"feature_b":b,"real_association":r,"ctgan_association":vals["ctgan"],"ctgan_absolute_error":abs(r-vals["ctgan"]),"tvae_association":vals["tvae"],"tvae_absolute_error":abs(r-vals["tvae"]),"generator_error_difference_ctgan_minus_tvae":abs(r-vals["ctgan"])-abs(r-vals["tvae"])})
    return pd.DataFrame(rows)


def privacy_diagnostics() -> pd.DataFrame:
    train=read_partition("train"); cats=CATEGORICAL+[load_config()["target_column"]]
    loo=_distances(train,train,cats,train); np.fill_diagonal(loo,np.inf); base=loo.min(axis=1)
    levels={f"train_loo_q{int(q*100)}":float(np.quantile(base,q)) for q in [.25,.5,.75,.95]}
    rows=[]
    for g in GENERATORS:
        syn=pd.read_csv(ROOT/f"GENAI/data/synthetic/{g}/synthetic_{g}.csv",dtype={c:str for c in cats})
        d=_distances(syn,train,cats,train).min(axis=1)
        rec={"generator":g.upper(),"synthetic_rows":len(d),"minimum":float(d.min()),"q25":float(np.quantile(d,.25)),"median":float(np.median(d)),"q75":float(np.quantile(d,.75)),"maximum":float(d.max())}
        for name,threshold in levels.items(): rec[f"proportion_below_{name}"]=float(np.mean(d<threshold))
        rec.update(levels); rec["note"]="Descriptive comparison to training leave-one-out distances; no privacy threshold or guarantee."
        rows.append(rec)
    return pd.DataFrame(rows)


def nearest_record_pairs() -> pd.DataFrame:
    """List closest pairs using source-row indices, never patient/file identifiers."""
    train=read_partition("train"); cats=CATEGORICAL+[load_config()["target_column"]]
    idx=pd.read_csv(ROOT/"GENAI/data/processed/split_indices.csv").query("partition == 'train'").sort_values("source_row_index_zero_based")["source_row_index_zero_based"].to_numpy()
    rows=[]
    for g in GENERATORS:
        syn=pd.read_csv(ROOT/f"GENAI/data/synthetic/{g}/synthetic_{g}.csv",dtype={c:str for c in cats})
        matrix=_distances(syn,train,cats,train); closest=matrix.argmin(axis=1); distance=matrix.min(axis=1)
        for si in np.argsort(distance)[:10]:
            rows.append({"generator":g.upper(),"synthetic_row_index_zero_based":int(si),"nearest_training_partition_row_index_zero_based":int(closest[si]),"nearest_source_row_index_zero_based":int(idx[closest[si]]),"distance":float(distance[si])})
    return pd.DataFrame(rows)


def run() -> dict:
    OUT.mkdir(parents=True,exist_ok=True)
    lineage=verify_lineage()
    num,dist=numerical_diagnostics(); cat=categorical_diagnostics(); target=target_diagnostic(); assoc=association_diagnostics(); privacy=privacy_diagnostics(); pairs=nearest_record_pairs()
    num.to_csv(OUT/"numerical_feature_diagnostics.csv",index=False)
    dist.to_csv(OUT/"numerical_distribution_diagnostics.csv",index=False)
    cat.to_csv(OUT/"categorical_feature_diagnostics.csv",index=False)
    target.to_csv(OUT/"target_distribution_diagnostic.csv",index=False)
    assoc.to_csv(OUT/"association_diagnostic.csv",index=False)
    privacy.to_csv(OUT/"privacy_distance_diagnostic.csv",index=False)
    pairs.to_csv(OUT/"nearest_record_pairs.csv",index=False)
    (OUT/"lineage_verification.json").write_text(json.dumps(lineage,indent=2),encoding="utf-8")
    return {"lineage":lineage,"numerical":num,"distributions":dist,"categorical":cat,"target":target,"association":assoc,"privacy":privacy}


if __name__=="__main__":
    result=run()
    print("Diagnostics written to",OUT)
    print("Numerical weakest CTGAN:",result["numerical"].nsmallest(10,"ctgan_similarity")[["feature","ctgan_similarity","tvae_similarity"]].to_string(index=False))
    print("Privacy:",result["privacy"].to_string(index=False))
