import unittest

import torch

from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=20, suite="stage-b-test-1-npu-a2", nightly=False)

try:
    import torch_npu  # noqa: F401

    _HAS_NPU = torch.npu.is_available()
except (AttributeError, ImportError):
    _HAS_NPU = False

if _HAS_NPU:
    get_device_name = getattr(torch.npu, "get_device_name", None)
    if get_device_name is None:
        get_device_name = torch_npu.npu.get_device_name
    _IS_ASCEND_950 = "Ascend950" in str(get_device_name(0)).replace(" ", "")
else:
    _IS_ASCEND_950 = False


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


@unittest.skipUnless(
    _IS_ASCEND_950,
    "The fused E4M3FN Hadamard GEMM quantizer requires Ascend 950",
)
class TestDSAIndexerHadamardGemmQuant(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from sglang.srt.layers.attention.dsa.dsa_indexer import (
            _quantize_npu_indexer_activation,
        )

        cls.fused_quant = staticmethod(_quantize_npu_indexer_activation)
        cls.hadamard = _normalized_hadamard_128().to(
            device="npu",
            dtype=torch.bfloat16,
        )

    def _check_against_vllm_sequence(self, shape):
        torch.manual_seed(3)
        x = torch.randn(shape, dtype=torch.bfloat16, device="npu")
        actual_q, actual_scale = self.fused_quant(
            x,
            self.hadamard,
            torch.float8_e4m3fn,
        )

        rotated = x.reshape(-1, 128) @ self.hadamard
        expected_q, expected_scale = torch_npu.npu_dynamic_quant(
            rotated,
            dst_type=torch.float8_e4m3fn,
        )
        expected_q = expected_q.reshape(shape)
        expected_scale = expected_scale.reshape(shape[:-1])

        self.assertEqual(actual_q.shape, x.shape)
        self.assertEqual(actual_q.dtype, torch.float8_e4m3fn)
        self.assertTrue(actual_q.is_contiguous())
        self.assertEqual(actual_scale.shape, x.shape[:-1])
        self.assertEqual(actual_scale.dtype, torch.float32)
        self.assertTrue(actual_scale.is_contiguous())
        torch.testing.assert_close(
            actual_scale,
            expected_scale,
            rtol=2e-3,
            atol=1e-5,
        )

        actual_dequant = (
            actual_q.float() * actual_scale.unsqueeze(-1)
        ).cpu()
        expected_dequant = (
            expected_q.float() * expected_scale.unsqueeze(-1)
        ).cpu()
        torch.testing.assert_close(
            actual_dequant,
            expected_dequant,
            rtol=2e-2,
            atol=5e-2,
        )

    def test_k_and_q_shapes_cover_all_block_sizes(self):
        self._check_against_vllm_sequence((1, 128))
        self._check_against_vllm_sequence((3, 1, 128))
        self._check_against_vllm_sequence((1, 32, 128))
        self._check_against_vllm_sequence((3, 32, 128))

    def test_zero_row_and_per_row_scales(self):
        x = torch.zeros((4, 128), dtype=torch.bfloat16, device="npu")
        x[1, 0] = 1
        x[2, 0] = 16
        x[3] = torch.linspace(
            -2,
            2,
            128,
            dtype=torch.bfloat16,
            device="npu",
        )

        quantized, scales = self.fused_quant(
            x,
            self.hadamard,
            torch.float8_e4m3fn,
        )
        self.assertEqual(scales[0].item(), 0.0)
        self.assertTrue(torch.count_nonzero(quantized[0].float()).item() == 0)
        self.assertGreater(scales[2].item(), scales[1].item())

        rotated = x @ self.hadamard
        expected_q, expected_scales = torch_npu.npu_dynamic_quant(
            rotated,
            dst_type=torch.float8_e4m3fn,
        )
        torch.testing.assert_close(scales, expected_scales, rtol=2e-3, atol=1e-5)
        torch.testing.assert_close(
            quantized.float() * scales[:, None],
            expected_q.float() * expected_scales[:, None],
            rtol=2e-2,
            atol=5e-2,
        )

    def test_empty_input(self):
        x = torch.empty((0, 32, 128), dtype=torch.bfloat16, device="npu")
        quantized, scales = self.fused_quant(
            x,
            self.hadamard,
            torch.float8_e4m3fn,
        )
        self.assertEqual(quantized.shape, x.shape)
        self.assertEqual(quantized.dtype, torch.float8_e4m3fn)
        self.assertEqual(scales.shape, (0, 32))
        self.assertEqual(scales.dtype, torch.float32)

    def test_rejects_unsupported_inputs(self):
        with self.assertRaisesRegex(TypeError, "requires BF16 input"):
            self.fused_quant(
                torch.empty((1, 128), dtype=torch.float16, device="npu"),
                self.hadamard,
                torch.float8_e4m3fn,
            )

        with self.assertRaisesRegex(ValueError, "last dimension 128"):
            self.fused_quant(
                torch.empty((1, 64), dtype=torch.bfloat16, device="npu"),
                self.hadamard,
                torch.float8_e4m3fn,
            )

        noncontiguous = torch.empty(
            (128, 2),
            dtype=torch.bfloat16,
            device="npu",
        ).transpose(0, 1)
        with self.assertRaisesRegex(ValueError, "contiguous tensor"):
            self.fused_quant(
                noncontiguous,
                self.hadamard,
                torch.float8_e4m3fn,
            )

        with self.assertRaisesRegex(ValueError, "only supports"):
            self.fused_quant(
                torch.empty((1, 128), dtype=torch.bfloat16, device="npu"),
                self.hadamard,
                torch.int8,
            )


if __name__ == "__main__":
    unittest.main()
