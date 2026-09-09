"""Case category isolation and runner selection checks."""
import json
import hashlib
import tempfile
import unittest
from pathlib import Path
from tests.common.cases import CASES, CASE_GROUPS, CaseSpec
from tests.common.mock_cases import load_mock_suite, materialize_mock_fixture
from tests.st.scripts.audit_coverage import audit
from tests.st.scripts.run_tests import selected_case_ids


class MatrixTests(unittest.TestCase):
    def test_mock_measured_contract(self):
        expected = [
            (1, 2, 6, 4, 4, 32, 16, 8, 2, .1768, "float16"),
            (2, 3, 8, 4, 4, 32, 16, 8, 3, .1768, "float16"),
            (1, 4, 10, 8, 8, 32, 16, 8, 3, .125, "float32"),
            (1, 8, 16, 8, 8, 64, 32, 16, 3, .125, "float16"),
            (2, 4, 16, 8, 8, 64, 32, 16, 3, .125, "float16"),
            (1, 8, 32, 16, 16, 64, 32, 16, 4, .125, "float32"),
        ]
        keys = ("b", "s1", "s2", "n1", "nidx1", "d_query", "d_query_index", "d_rope", "k", "scale_value", "core_dtype")
        for number, (item, shape) in enumerate(zip(load_mock_suite()["cases"], expected), 1):
            spec = item["spec"]
            self.assertEqual(tuple(spec[key] for key in keys), shape)
            self.assertEqual((spec["layout"], spec["statistics_layout"], spec["repetitions"]), ("BSND", "BNSG", 1))
            self.assertTrue(all(spec[key] == spec["core_dtype"] for key in
                                ("weights_dtype", "query_rope_dtype", "key_rope_dtype")))
            if number == 2:
                self.assertEqual((spec["actual_seq_dtype"], spec["actual_q"], spec["actual_k"], spec["omit_lengths"]),
                                 ("int32", [2, 3], [5, 8], False))
            else:
                self.assertEqual((spec["actual_q"], spec["actual_k"], spec["omit_lengths"]), ([], [], True))

    def test_mock_input_and_golden_bytes_are_frozen(self):
        # Fingerprints of the validated six fixtures, including all inputs,
        # four goldens and the unselected-key check. Paths/labels are excluded.
        expected = [
            "ae05c28eadff0fd16d2b7b7f4226673c3c2f36f7417769291629281e02a3c8b7",
            "9c3351a990427a30385d1e3609033ac2e38d87b502276a08f52cf05247923ad5",
            "1c86a7cf92e834664be4dc86a7310425aeecae5d7a4055b4aaf3fdaba14ed041",
            "ac1168c447dc4e88aa1ffbdbc721ac0c8f4ceca74f0fcb3fe7570ac9faa4f548",
            "2919c4896887686db4e732488f89389fdcb31c51fe25fcb07b94d51f0e03554d",
            "e6c9368bd962b2acfa181a8998ce0f04dd0f1b3927fcbe6f38eaddaee135a913",
        ]
        with tempfile.TemporaryDirectory() as directory:
            for item, fingerprint in zip(load_mock_suite()["cases"], expected):
                case = CaseSpec(**item["spec"])
                folder = materialize_mock_fixture(case, directory)
                value = hashlib.sha256()
                for path in sorted(folder.glob("*.bin")):
                    value.update(path.name.encode() + b"\0" + path.read_bytes())
                self.assertEqual(value.hexdigest(), fingerprint, case.case_id)
                before = (folder / ".complete").stat().st_mtime_ns
                self.assertEqual(materialize_mock_fixture(case, directory), folder)
                self.assertEqual((folder / ".complete").stat().st_mtime_ns, before)

    def test_coverage(self):
        self.assertEqual(audit()["issues"], [])

    def test_categories_and_selection(self):
        groups = {name: {case.case_id for case in cases}
                  for name, cases in CASE_GROUPS.items()}
        self.assertTrue(groups["coverage"])
        self.assertTrue(groups["correctness"])
        self.assertFalse(groups["coverage"] & groups["correctness"])
        mock = json.loads((Path(__file__).parents[1] / "st/cases/mock.json").read_text())
        self.assertEqual([row["official_case"] for row in mock["cases"]], list(range(1, 7)))
        mock_ids = {row["spec"]["case_id"] for row in mock["cases"]}
        self.assertEqual(len(mock_ids), 6)
        self.assertFalse(mock_ids & {case.case_id for case in CASES})
        for group, identifiers in groups.items():
            self.assertEqual(set(selected_case_ids(group, ())), identifiers)
        with self.assertRaisesRegex(ValueError, "outside coverage"):
            selected_case_ids("coverage", ("sharp_main",))
        with self.assertRaisesRegex(ValueError, "outside correctness"):
            selected_case_ids("correctness", ("baseline_f16",))
        self.assertEqual(set(selected_case_ids("full", ())),
                         groups["coverage"] | groups["correctness"])


if __name__ == "__main__":
    unittest.main()
