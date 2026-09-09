"""Independent gradient and invariant checks; no NPU required."""
import unittest
from dataclasses import replace
import numpy as np
from tests.common.cases import get_case
from tests.common.reference import indexer_kl_backward, reference_backward
from tests.st.scripts.gen_data import generate_inputs


class ReferenceTests(unittest.TestCase):
    def test_gradients_with_repeated_indices(self):
        rng = np.random.default_rng(17)
        q = np.abs(rng.normal(size=(2, 3))) + 0.2
        k = np.abs(rng.normal(size=(4, 3))) + 0.2
        w = np.asarray([0.25, -0.35])
        indices = np.asarray([0, 2, 2, 3])
        target = np.asarray([0.1, 0.2, 0.3, 0.4])
        expected = indexer_kl_backward(q, k, w, indices, target)
        for arg, name in enumerate(("dQueryIndex", "dKeyIndex", "dWeights")):
            values = [q, k, w]
            numeric = np.zeros_like(values[arg])
            for index in np.ndindex(numeric.shape):
                plus, minus = [x.copy() for x in values], [x.copy() for x in values]
                plus[arg][index] += 1e-6
                minus[arg][index] -= 1e-6
                numeric[index] = (indexer_kl_backward(*plus, indices, target)["loss"][0] -
                                  indexer_kl_backward(*minus, indices, target)["loss"][0]) / 2e-6
            np.testing.assert_allclose(expected[name], numeric, rtol=2e-6, atol=2e-8)
        np.testing.assert_array_equal(expected["dKeyIndex"][1], 0)

    def test_sharp_kl_uses_log_softmax_without_clipping(self):
        result = indexer_kl_backward(np.asarray([[1.0]]), np.asarray([[0.0], [100.0]]),
                                    np.asarray([1.0]), np.asarray([0, 1]), np.asarray([0.5, 0.5]))
        self.assertAlmostEqual(result["loss"][0], 50.0 - np.log(2.0), places=12)

    def test_padding_scatter_and_zero_weights(self):
        base = replace(get_case("baseline_f16"), b=2, s1=3, s2=17,
                       actual_q=(3, 1), actual_k=(17, 13), index_pattern="endpoints")
        result = reference_backward(generate_inputs(base), base)
        np.testing.assert_array_equal(result["dQueryIndex"][1, 1:], 0)
        np.testing.assert_array_equal(result["dWeights"][1, 1:], 0)
        np.testing.assert_array_equal(result["dKeyIndex"][:, 1:10], 0)
        case = replace(base, value_profile="weights_zero")
        result = reference_backward(generate_inputs(case), case)
        np.testing.assert_array_equal(result["dQueryIndex"], 0)
        np.testing.assert_array_equal(result["dKeyIndex"], 0)
        self.assertGreater(np.max(np.abs(result["dWeights"])), 0)


if __name__ == "__main__":
    unittest.main()
