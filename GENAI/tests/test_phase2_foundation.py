import hashlib
import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"GENAI/src"))
from phase2_data import prepare
from quality_gate import assess
from phase2_data import load_config


class PhaseTwoFoundationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source=ROOT/"GENAI/data/raw/PCOS_data.csv"
        cls.source_hash=hashlib.sha256(cls.source.read_bytes()).hexdigest()
        cls.meta=prepare()

    def test_raw_integrity_and_split_determinism(self):
        self.assertEqual(hashlib.sha256(self.source.read_bytes()).hexdigest(),self.source_hash)
        prior=self.meta["source_row_indices_zero_based"]
        self.assertEqual(prior,prepare()["source_row_indices_zero_based"])

    def test_partition_sizes_stratification_and_disjointness(self):
        self.assertEqual(self.meta["row_counts"],{"train":378,"validation":81,"test":82})
        sets=[set(v) for v in self.meta["source_row_indices_zero_based"].values()]
        self.assertFalse(sets[0]&sets[1]); self.assertFalse(sets[0]&sets[2]); self.assertFalse(sets[1]&sets[2])

    def test_generator_inputs_exclude_identifiers_and_retain_target(self):
        self.assertNotIn("Sl. No",self.meta["generator_features"])
        self.assertNotIn("Patient File No.",self.meta["generator_features"])
        self.assertIn(load_config()["target_column"],self.meta["generator_features"])

    def test_category_codes_remain_categorical_strings(self):
        from phase2_data import read_partition
        train=read_partition("train")
        self.assertEqual(train["Blood Group"].dtype,object)
        self.assertIn("15",set(train["Blood Group"]))

    def test_gate_does_not_approve_without_evidence_policy(self):
        result=assess("ctgan",{"hard_technical_checks_pass":True},{"aggregate_mean_one_minus_ks":.7,"aggregate_mean_js_similarity":.8},{"aggregate_association_similarity_1_minus_error":.9},{})
        self.assertEqual(result["status"],"REVIEW_REQUIRED")
        self.assertFalse(result["approved_for_downstream_use"])

    def test_saved_generator_lineage_excludes_validation_and_test(self):
        for generator in ["ctgan", "tvae"]:
            p=ROOT/f"GENAI/data/synthetic/{generator}/run_metadata.json"
            self.assertTrue(p.is_file())
            import json
            metadata=json.loads(p.read_text(encoding="utf-8"))
            self.assertEqual(metadata["fit_data_source"],"GENAI/data/processed/train.csv only")
            self.assertFalse(metadata["validation_or_test_passed_to_fit"])
            self.assertEqual(metadata["training_rows"],378)
            self.assertEqual(metadata["synthetic_rows"],378)


if __name__=="__main__": unittest.main()
