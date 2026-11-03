import hashlib
import unittest
from pathlib import Path

import pandas as pd

from GENAI.src.audit_dataset import DATA, TARGET, audit


class PhaseOneAuditTests(unittest.TestCase):
    def test_source_exists_and_target_is_present(self):
        self.assertTrue(DATA.is_file())
        self.assertIn(TARGET, pd.read_csv(DATA, nrows=0).columns)

    def test_inventory_covers_each_column(self):
        inventory, report = audit()
        self.assertEqual(len(inventory), report["dataset"]["columns"])
        self.assertEqual(inventory["original_name"].tolist(), pd.read_csv(DATA, nrows=0).columns.tolist())

    def test_audit_is_reproducible_and_read_only(self):
        before = hashlib.sha256(DATA.read_bytes()).hexdigest()
        first = audit()[1]
        second = audit()[1]
        after = hashlib.sha256(DATA.read_bytes()).hexdigest()
        self.assertEqual(first, second)
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
