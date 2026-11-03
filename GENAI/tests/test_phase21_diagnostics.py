import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"GENAI/src"))
from diagnose_phase2 import verify_lineage, numerical_diagnostics, privacy_diagnostics


class Phase21DiagnosticTests(unittest.TestCase):
    def test_lineage_counts_and_disjointness(self):
        result=verify_lineage()
        self.assertEqual(result["raw_rows"],541)
        self.assertEqual(result["split_row_counts_indices"],{"train":378,"validation":81,"test":82})
        self.assertEqual(set(result["pairwise_split_intersections"].values()),{0})
        self.assertTrue(result["raw_sha256_matches_split_metadata"])
        for generator in result["generator_lineage"].values():
            self.assertEqual(generator["exact_record_overlap_recomputed"],{"train":0,"validation":0,"test":0})

    def test_numerical_scores_recomputed_match_saved_evidence(self):
        values,_=numerical_diagnostics()
        self.assertEqual(len(values),31)
        for g in ["ctgan","tvae"]:
            self.assertLess((values[f"{g}_similarity"]-values[f"{g}_stored_similarity"]).abs().max(),1e-12)

    def test_privacy_distances_are_descriptive_and_nonnegative(self):
        values=privacy_diagnostics()
        self.assertEqual(set(values.generator),{"CTGAN","TVAE"})
        self.assertTrue((values["minimum"]>=0).all())
        self.assertTrue((values["proportion_below_train_loo_q50"].between(0,1)).all())


if __name__=="__main__": unittest.main()
