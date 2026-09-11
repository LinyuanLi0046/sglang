"""Single-NPU mode-consistency tests for the framework's WeLM fused QKV op.

Run from the repository root, with its python/ directory on PYTHONPATH:

    NPU_DEVICE=npu:4 python test/manual/ascend/test_welmv4_fused_qkv.py -v -f

Requires the CANNBotDSL SDK and its supported NPU/toolchain (ARCH 3510).
No server, checkpoint, distributed initialization or small operator wheel is
needed. The operator is imported from sglang.srt.layers. Missing NPU support
is reported as SKIP; SDK import/compilation failures on NPU remain errors.

The tests compare contiguous loads with per-row positions, including ragged
prefill and target verify, and check dense outputs, raw mirror outputs and KV
scatter. They do not establish independent GEMM/norm/RoPE accuracy or
model/Graph correctness, and do not measure latency. Compiled callables are
reused across M.
"""

import importlib.util
import os
import unittest


class TestWeLMv4FusedQKV(unittest.TestCase):
    CACHE_ROWS = 4096
    MAX_POSITION = 8192
    CACHE_SENTINEL = -7.0
    OUTPUT_NAMES = ("q", "k", "v", "mirror_k", "mirror_v", "k_cache", "v_cache")

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if any(
            importlib.util.find_spec(name) is None for name in ("torch", "torch_npu")
        ):
            raise unittest.SkipTest("PyTorch with Ascend NPU support is required")

        import torch
        import torch_npu  # noqa: F401

        if not torch.npu.is_available():
            raise unittest.SkipTest("No Ascend NPU is available")

        cls.torch = torch
        cls.device = os.environ.get("NPU_DEVICE", "npu:0")
        torch.npu.set_device(cls.device)

        # Deliberately do not hide missing/broken SDK or operator imports.
        from sglang.srt.layers import fused_qkv_proj_norm_rope_cache

        cls.operator = fused_qkv_proj_norm_rope_cache
        cls.programs = {}
        torch.manual_seed(1234)
        cls.weights = {
            width: (torch.randn((width, 2048), device=cls.device) * 0.02).to(
                torch.bfloat16
            )
            for width in (2048, 2560)
        }
        cls.gamma = (
            1 + 0.1 * torch.randn((1, 256), device=cls.device)
        ).to(torch.bfloat16)
        angle = (
            torch.arange(cls.MAX_POSITION, dtype=torch.float32, device=cls.device)[
                :, None
            ]
            * torch.linspace(0.001, 0.1, 32, device=cls.device)[None, :]
        )
        cls.cos_sin = torch.cat((angle.cos(), angle.sin()), dim=1).contiguous()

    @classmethod
    def tearDownClass(cls):
        cls.programs.clear()
        cls.weights.clear()
        del cls.gamma, cls.cos_sin
        super().tearDownClass()

    def _program(self, width, mode):
        key = (width, mode)
        if key not in self.programs:
            self.programs[key] = self.operator.compile_aot(
                width,
                self.CACHE_ROWS,
                self.MAX_POSITION,
                return_v=True,
                positions_contiguous=mode == "contiguous",
            )
        return self.programs[key]

    def _inputs(self, lengths, *, prefixes=None, pad=True):
        torch = self.torch
        real_m = sum(lengths)
        m = (real_m + 3) // 4 * 4 if pad else real_m
        if prefixes is None:
            prefixes = [17 + (i % 8) * 800 for i in range(len(lengths))]
        self.assertEqual(len(prefixes), len(lengths))
        hidden = torch.randn((m, 2048), device=self.device).to(torch.bfloat16)
        hidden[real_m:] = 0

        # Build test metadata on CPU; this is not framework/forward code.
        positions = torch.zeros(m, dtype=torch.int64, device="cpu")
        offset = 0
        for length, prefix in zip(lengths, prefixes):
            self.assertGreaterEqual(prefix, 0)
            self.assertLessEqual(prefix + length, self.MAX_POSITION)
            positions[offset : offset + length] = torch.arange(
                prefix, prefix + length, device="cpu"
            )
            offset += length

        # Real rows use nonconsecutive, unique cache slots. Padding uses 0.
        slots = torch.zeros(m, dtype=torch.int64, device="cpu")
        slots[:real_m] = torch.arange(real_m, device="cpu") * 2 + 1
        self.assertLess(int(slots.max()), self.CACHE_ROWS)
        # Preserve one real write even in M=1; otherwise test skip-write too.
        if real_m > 1:
            slots[real_m // 2] = -1
        return hidden, positions.to(self.device), slots.to(self.device)

    def _invoke(self, width, mode, inputs):
        torch = self.torch
        hidden, positions, slots = inputs
        m = hidden.shape[0]
        mirror_m = m if width == 2560 else 1
        shapes = ((m, 1536), (m, 256), (m, 256), (mirror_m, 256), (mirror_m, 256))
        # NaN catches missing dense writes; ordinary-layer mirror stand-ins
        # are intentionally unwritten and excluded from assertions.
        outputs = [
            torch.full(shape, float("nan"), dtype=torch.bfloat16, device=self.device)
            for shape in shapes
        ]
        caches = [
            torch.full(
                (self.CACHE_ROWS, 256),
                self.CACHE_SENTINEL,
                dtype=torch.bfloat16,
                device=self.device,
            )
            for _ in range(2)
        ]
        args = (
            hidden,
            self.weights[width],
            self.gamma,
            positions,
            self.cos_sin,
            slots,
            *outputs,
            *caches,
            1e-6,
        )
        program = self._program(width, mode)
        program(*args)
        torch.npu.synchronize()

        # Test assertions run on CPU, outside the operator execution.
        result = [tensor.cpu() for tensor in outputs + caches]
        for i, name in enumerate(self.OUTPUT_NAMES):
            if width == 2048 and name.startswith("mirror"):
                continue
            self.assertTrue(torch.isfinite(result[i]).all().item(), name)
        self._check_cache(result, slots.cpu())
        return result

    def _check_cache(self, result, slots):
        torch = self.torch
        # Positive slots belong to real tokens; duplicate dummy slot 0 is
        # ignored here because the order of padding writes is unspecified.
        valid = slots > 0
        for dense, cache in ((result[1], result[5]), (result[2], result[6])):
            torch.testing.assert_close(
                cache[slots[valid]], dense[valid], rtol=0, atol=0
            )
            untouched = torch.ones(self.CACHE_ROWS, dtype=torch.bool, device="cpu")
            untouched[slots[slots >= 0]] = False
            self.assertTrue((cache[untouched] == self.CACHE_SENTINEL).all().item())

    def _assert_equal(self, width, reference, actual):
        for i, name in enumerate(self.OUTPUT_NAMES):
            if width == 2048 and name.startswith("mirror"):
                continue
            self.torch.testing.assert_close(
                actual[i], reference[i], rtol=0, atol=0, msg=name
            )

    def _compare_modes(self, width, mode, inputs):
        reference = self._invoke(width, "row", inputs)
        actual = self._invoke(width, mode, inputs)
        self._assert_equal(width, reference, actual)

    def _compare_requests(self, width, lengths, *, prefixes=None):
        inputs = self._inputs(lengths, prefixes=prefixes)
        packed = self._invoke(width, "row", inputs)
        hidden, positions, slots = inputs
        requests = [i for i, length in enumerate(lengths) if length > 0]
        selected = {requests[0], requests[len(requests) // 2], requests[-1]}
        for request in sorted(selected):
            start = sum(lengths[:request])
            end = start + lengths[request]
            reference = self._invoke(
                width,
                "contiguous",
                (hidden[start:end], positions[start:end], slots[start:end]),
            )
            for i in range(5 if width == 2560 else 3):
                self.torch.testing.assert_close(
                    packed[i][start:end],
                    reference[i],
                    rtol=0,
                    atol=0,
                    msg=f"request={request}, {self.OUTPUT_NAMES[i]}",
                )

    def test_contiguous_prefill_matches_row_positions(self):
        for width in (2048, 2560):
            for m, pad in ((1, False), (129, False), (129, True), (641, True)):
                with self.subTest(width=width, m=m, pad=pad):
                    self._compare_modes(
                        width, "contiguous", self._inputs((m,), pad=pad)
                    )

    def test_ragged_prefill_matches_per_request_contiguous(self):
        cases = (
            (0, 1),
            (1, 1),
            (31, 34),
            (320, 320),
            (319, 322),
            (1, 255, 0, 513),
            (257, 513, 255),
            (63, 64, 65, 127, 128, 194),
            (0, 641, 0),
            (1,) * 641,
        )
        for width in (2048, 2560):
            for case_id, lengths in enumerate(cases):
                with self.subTest(width=width, case=case_id, m=sum(lengths)):
                    self._compare_requests(width, lengths)

    def test_target_verify_positions_match_per_request_contiguous(self):
        # Verify packs B independent D-token runs. The combined call must use
        # per-row positions even though each individual request is contiguous.
        for width in (2048, 2560):
            for bs, draft_tokens in ((1, 2), (2, 3), (56, 2), (56, 3)):
                with self.subTest(width=width, bs=bs, draft_tokens=draft_tokens):
                    inputs = self._inputs((draft_tokens,) * bs, pad=False)
                    packed = self._invoke(width, "row", inputs)
                    hidden, positions, slots = inputs
                    for request in sorted({0, bs // 2, bs - 1}):
                        start = request * draft_tokens
                        end = start + draft_tokens
                        reference = self._invoke(
                            width,
                            "contiguous",
                            (
                                hidden[start:end],
                                positions[start:end],
                                slots[start:end],
                            ),
                        )
                        for i in range(5 if width == 2560 else 3):
                            self.torch.testing.assert_close(
                                packed[i][start:end],
                                reference[i],
                                rtol=0,
                                atol=0,
                                msg=f"request={request}, {self.OUTPUT_NAMES[i]}",
                            )

    def test_cos_sin_table_tail(self):
        for width in (2048, 2560):
            with self.subTest(width=width, mode="contiguous"):
                self._compare_modes(
                    width,
                    "contiguous",
                    self._inputs(
                        (257,), prefixes=(self.MAX_POSITION - 257,), pad=False
                    ),
                )
            with self.subTest(width=width, mode="row", bs=2):
                self._compare_requests(
                    width,
                    (257, 384),
                    prefixes=(self.MAX_POSITION - 257, self.MAX_POSITION - 384),
                )

    def test_arbitrary_decode_positions_and_repeated_launch(self):
        for width in (2048, 2560):
            for m in (1, 2, 56):
                with self.subTest(width=width, m=m):
                    inputs = self._inputs(
                        (1,) * m,
                        prefixes=[
                            self.MAX_POSITION - 1 - (i * 131 % 4096) for i in range(m)
                        ],
                        pad=False,
                    )
                    first = self._invoke(width, "row", inputs)
                    second = self._invoke(width, "row", inputs)
                    self._assert_equal(width, first, second)


if __name__ == "__main__":
    unittest.main()
