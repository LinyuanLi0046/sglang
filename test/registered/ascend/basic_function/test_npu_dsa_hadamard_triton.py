import unittest

import torch

from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=20, suite="stage-b-test-1-npu-a2", nightly=False)

try:
    import torch_npu  # noqa: F401

    _HAS_NPU = torch.npu.is_available()
except (AttributeError, ImportError):
    _HAS_NPU = False


def _normalized_hadamard_128() -> torch.Tensor:
    matrix = torch.ones((1, 1), dtype=torch.float32)
    while matrix.shape[0] < 128:
        matrix = torch.cat(
            (
                torch.cat((matrix, matrix), dim=1),
                torch.cat((matrix, -matrix), dim=1),
            ),
            dim=0,
        )
    return matrix * (128**-0.5)


@unittest.skipUnless(_HAS_NPU, "Ascend NPU is required")
class TestDSAIndexerHadamard(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from sglang.srt.layers.attention.dsa.dsa_indexer import (
            NUM_VECTOR_CORES,
            rotate_activation,
        )

        cls.num_vector_cores = NUM_VECTOR_CORES
        cls.rotate_activation = staticmethod(rotate_activation)
        cls.reference_matrix = _normalized_hadamard_128()

    def _check_against_reference(self, shape):
        torch.manual_seed(0)
        x = torch.randn(shape, dtype=torch.bfloat16, device="npu")
        actual = self.rotate_activation(x)
        expected = (
            x.cpu()
            .to(torch.float32)
            .reshape(-1, 128)
            .matmul(self.reference_matrix)
            .reshape(shape)
        )

        self.assertEqual(actual.shape, x.shape)
        self.assertEqual(actual.dtype, torch.bfloat16)
        torch.testing.assert_close(
            actual.cpu().to(torch.float32),
            expected,
            rtol=1e-2,
            atol=5e-2,
        )

    def test_small_decode_shapes(self):
        self._check_against_reference((1, 128))
        self._check_against_reference((2, 32, 128))

    def test_large_prefill_shape_and_tail(self):
        rows = self.num_vector_cores * 8 + 3
        self._check_against_reference((rows, 128))

    def test_basis_vector_preserves_sylvester_order(self):
        x = torch.zeros((1, 128), dtype=torch.bfloat16, device="npu")
        x[0, 1] = 1
        actual = self.rotate_activation(x).cpu().to(torch.float32)
        expected = self.reference_matrix[1:2]
        torch.testing.assert_close(actual, expected, rtol=0, atol=5e-4)

    def test_qk_dot_product_is_preserved(self):
        torch.manual_seed(1)
        q = torch.randn((4, 32, 128), dtype=torch.bfloat16, device="npu")
        k = torch.randn((4, 128), dtype=torch.bfloat16, device="npu")
        q_rotated = self.rotate_activation(q).cpu().to(torch.float32)
        k_rotated = self.rotate_activation(k).cpu().to(torch.float32)

        before = (
            q.cpu().to(torch.float32)
            * k.cpu().to(torch.float32).unsqueeze(1)
        ).sum(dim=-1)
        after = (q_rotated * k_rotated.unsqueeze(1)).sum(dim=-1)
        torch.testing.assert_close(after, before, rtol=2e-2, atol=2e-1)

    def test_transform_is_approximately_involutory(self):
        torch.manual_seed(2)
        x = torch.randn((17, 128), dtype=torch.bfloat16, device="npu")
        restored = self.rotate_activation(self.rotate_activation(x))
        torch.testing.assert_close(
            restored.cpu().to(torch.float32),
            x.cpu().to(torch.float32),
            rtol=2e-2,
            atol=1e-1,
        )

    def test_empty_input(self):
        x = torch.empty((0, 128), dtype=torch.bfloat16, device="npu")
        actual = self.rotate_activation(x)
        self.assertEqual(actual.shape, x.shape)
        self.assertEqual(actual.dtype, torch.bfloat16)

    def test_rejects_unsupported_shape_and_layout(self):
        with self.assertRaisesRegex(ValueError, "last dimension 128"):
            self.rotate_activation(
                torch.empty((1, 64), dtype=torch.bfloat16, device="npu")
            )

        noncontiguous = torch.empty(
            (128, 2), dtype=torch.bfloat16, device="npu"
        ).transpose(0, 1)
        self.assertFalse(noncontiguous.is_contiguous())
        with self.assertRaisesRegex(ValueError, "contiguous tensor"):
            self.rotate_activation(noncontiguous)


if __name__ == "__main__":
    unittest.main()
