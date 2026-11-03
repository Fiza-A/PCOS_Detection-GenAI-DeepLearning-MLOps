"""Generate measured CTGAN/TVAE comparison without forcing a winner."""
import json

import pandas as pd

from phase2_data import ROOT


def compare(results: dict) -> dict:
    rows=[]
    for g in ["ctgan","tvae"]:
        v=results[g]
        rows.append({"generator":g.upper(),"hard_technical_pass":v["validation"]["hard_technical_checks_pass"],"numerical_mean_1_minus_KS":v["fidelity"]["aggregate_mean_one_minus_ks"],"categorical_mean_JS_similarity":v["fidelity"]["aggregate_mean_js_similarity"],"association_similarity_1_minus_error":v["correlation"]["aggregate_association_similarity_1_minus_error"],"target_train_distribution":v["validation"]["target_counts_training"],"target_synthetic_distribution":v["validation"]["target_counts_synthetic"],"duplicate_rate":v["diversity"]["duplicate_rate"],"unique_rows":v["diversity"]["unique_synthetic_rows"],"exact_training_overlap":v["diversity"]["exact_record_overlap_training"],"nearest_distance_median":v["diversity"]["nearest_training_distance_quantiles"]["0.5"],"quality_gate_status":v["gate"]["status"]})
    table=pd.DataFrame(rows); out=ROOT/"GENAI/reports/comparison"; out.mkdir(parents=True,exist_ok=True); table.to_csv(out/"comparison.csv",index=False)
    numeric_best=max(rows,key=lambda x:x["numerical_mean_1_minus_KS"])["generator"]
    cat_best=max(rows,key=lambda x:x["categorical_mean_JS_similarity"])["generator"]
    assoc_best=max(rows,key=lambda x:x["association_similarity_1_minus_error"])["generator"]
    result={"comparison":rows,"dimension_leaders":{"numerical_fidelity":numeric_best,"categorical_fidelity":cat_best,"association_similarity":assoc_best},"decision":"Evidence is multi-dimensional and both outputs remain REVIEW_REQUIRED; no generator is approved or declared universally superior.","limitations":"Results describe these fitted models and this small training partition only. They do not establish clinical validity, generalization, privacy, or downstream predictive utility."}
    (out/"comparison.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    return result
