"""Evidence-separated technical quality gate; no invented acceptance thresholds."""
import json

from phase2_data import ROOT, load_config


def assess(generator: str, validation: dict, fidelity: dict, correlation: dict, diversity: dict) -> dict:
    cfg=load_config(); hard=bool(validation["hard_technical_checks_pass"])
    status="REJECTED" if not hard else "REVIEW_REQUIRED"
    criteria={
        "technical_schema_and_type_checks":{"status":"PASS" if hard else "FAIL","evidence":validation},
        "numerical_fidelity":{"status":"REVIEW_REQUIRED","value":fidelity["aggregate_mean_one_minus_ks"],"reason":"No scientifically justified acceptance threshold is established in project sources."},
        "categorical_fidelity":{"status":"REVIEW_REQUIRED","value":fidelity["aggregate_mean_js_similarity"],"reason":"No scientifically justified acceptance threshold is established in project sources."},
        "association_fidelity":{"status":"REVIEW_REQUIRED","value":correlation["aggregate_association_similarity_1_minus_error"],"reason":"No scientifically justified acceptance threshold is established in project sources."},
        "diversity_and_preliminary_privacy":{"status":"REVIEW_REQUIRED","evidence":diversity,"reason":"Metrics are descriptive; no approved decision thresholds or formal privacy guarantee."},
        "clinical_plausibility":{"status":"REVIEW_REQUIRED","reason":"Not established from available project sources."}}
    result={"generator":generator.upper(),"status":status,"approved_for_downstream_use":False,"reason":"No approved evidence thresholds are defined, so technical completion cannot by itself qualify this output for approval." if hard else "A hard technical validity check failed.","criteria":criteria,"threshold_policy":cfg["quality_gate"]}
    out=ROOT/"GENAI/reports/quality_gate"; out.mkdir(parents=True,exist_ok=True); (out/f"{generator.lower()}.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    return result
