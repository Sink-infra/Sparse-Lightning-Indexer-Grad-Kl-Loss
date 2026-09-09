import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
import numpy as np

from tests.common.cases import get_case
from tests.common.tensor_io import encode_logical, decode_logical
from tests.st.scripts.gen_data import generate_inputs, materialize_runner_fixture, fixture_cache_key


class GenerationTests(unittest.TestCase):
    def test_seed_and_storage_types(self):
        for dtype in ("float16", "bfloat16"):
            case = replace(get_case("baseline_f16"), core_dtype=dtype, weights_dtype=dtype)
            first, second = generate_inputs(case), generate_inputs(case)
            for name, value in first.items():
                self.assertEqual(value.shape, case.input_shapes()[name])
                np.testing.assert_array_equal(value, second[name])
            self.assertEqual(first["query"].dtype, np.uint16 if dtype == "bfloat16" else np.float16)

    def test_fixture_cache_detects_modified_data(self):
        case = get_case("baseline_f16")
        with tempfile.TemporaryDirectory() as directory:
            path, created = materialize_runner_fixture(case, Path(directory))
            self.assertTrue(created)
            self.assertFalse(materialize_runner_fixture(case, Path(directory))[1])
            (path / "golden_loss.bin").write_bytes(b"bad")
            self.assertTrue(materialize_runner_fixture(case, Path(directory))[1])
            self.assertEqual((path / "golden_loss.bin").stat().st_size, 4)
        self.assertNotEqual(fixture_cache_key(case), fixture_cache_key(replace(case, seed=12)))

    def test_large_cases_require_explicit_materialization(self):
        with self.assertRaises(MemoryError):
            generate_inputs(get_case("sequence_key_max_512k"), max_bytes=1024)

    def test_bfloat16_roundtrip_and_ties(self):
        values = np.asarray([1.0, -1.0, 1.00390625, 1.01171875, 0.0], dtype=np.float32)
        actual = decode_logical(encode_logical(values, "bfloat16"), "bfloat16")
        np.testing.assert_array_equal(actual, [1.0, -1.0, 1.0, 1.015625, 0.0])


if __name__ == "__main__":
    unittest.main()
