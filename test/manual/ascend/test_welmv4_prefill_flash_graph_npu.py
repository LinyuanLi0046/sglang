"""Native BF16 Flash graph regression; run in the serving Ascend environment.

Tests schedule+Flash capture, Full/SWA/sinks, negative-slot cache writes,
variable prefix/B within a T graph, and two T writers feeding one Mirror[B].
Queued replay snapshots are checked against FP32 attention and cache references.
This is an operator/adapter test, not an end-to-end TP/EP model benchmark.
"""

import copy
import types
import unittest

try:
    import torch
    import torch_npu
    from torch_npu.contrib import transfer_to_npu  # noqa: F401

    HAS_NPU = torch.npu.is_available()
except ImportError:
    HAS_NPU = False

if HAS_NPU:
    import cann_ops_transformer

    from sglang.srt.hardware_backend.npu.attention.ascend_backend import AscendAttnBackend
    from sglang.srt.model_executor.runner.welm_prefill_graph import (
        WelmPrefillGraphAdapter,
        padded_flash_cu_seqlens,
    )
    from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
        BreakableCUDAGraph,
        BreakableCUDAGraphCapture,
    )


@unittest.skipUnless(HAS_NPU, "requires torch_npu and an Ascend device")
class TestNativeFlashPrefillGraph(unittest.TestCase):
    def test_prompt_full_and_swa_with_variable_b_and_prefix(self):
        for swa in (False, True):
            with self.subTest(swa=swa):
                self._check(swa=swa, mirror=False)

    def test_mirror_full_and_swa_with_variable_t_and_prefix(self):
        for swa in (False, True):
            with self.subTest(swa=swa):
                self._check(swa=swa, mirror=True)

    def test_padded_mirror_full_and_swa_with_variable_b(self):
        for swa in (False, True):
            for with_sinks in (False, True):
                with self.subTest(swa=swa, sinks=with_sinks):
                    self._check(swa=swa, mirror=True, padded=True, with_sinks=with_sinks)

    @staticmethod
    def _reference(q, key, value, req_tokens, lengths, kv_lengths, layer, sinks, mirror):
        nq, nkv, dim = layer.tp_q_head_num, layer.tp_k_head_num, layer.qk_head_dim
        out = torch.zeros_like(q)
        offset = 0
        for i, (e, length) in enumerate(zip(lengths, kv_lengths)):
            start, width = (i, 1) if mirror else (offset, e)
            qr = q[start : start + width].view(width, nq, dim).float()
            kr = key.view(-1, nkv, dim)[req_tokens[i, :length]].float()
            vr = value.view(-1, nkv, dim)[req_tokens[i, :length]].float()
            kr = kr.repeat_interleave(nq // nkv, dim=1)
            vr = vr.repeat_interleave(nq // nkv, dim=1)
            scores = torch.einsum("qhd,khd->hqk", qr, kr) * layer.scaling
            qp = torch.arange(length - width, length, device=q.device)
            kp = torch.arange(length, device=q.device)
            visible = kp[None, :] <= qp[:, None]
            if layer.sliding_window_size >= 0:
                visible &= kp[None, :] >= qp[:, None] - layer.sliding_window_size
            scores.masked_fill_(~visible[None], float("-inf"))
            # Attention sinks contribute to the denominator, with zero V.
            if sinks is None:
                probs = scores.softmax(-1)
            else:
                sink_column = sinks[:, None, None].expand(nq, width, 1)
                probs = torch.cat((scores, sink_column), dim=-1).softmax(-1)[..., :-1]
            result = torch.einsum("hqk,khd->qhd", probs, vr)
            out[start : start + width] = result.reshape(width, -1).to(out.dtype)
            offset += e
        return out

    def _check(self, *, swa, mirror, padded=False, with_sinks=True):
        device = torch.device("npu", torch.npu.current_device())
        torch.manual_seed(17)
        capacity, bcap, page, context = 128, 4, 64, 256
        nq, nkv, dim = 6, 1, 256
        qrows = (bcap if padded else 2) if mirror else capacity
        q = torch.randn(qrows, nq * dim, device=device, dtype=torch.bfloat16)
        k = torch.randn(capacity, nkv * dim, device=device, dtype=q.dtype)
        v = torch.randn_like(k)
        key = torch.randn(36, page, nkv, dim, device=device, dtype=q.dtype)
        value = torch.randn_like(key)
        key[0].zero_()
        value[0].zero_()
        full_locs = torch.full((capacity,), -1, device=device, dtype=torch.int64)
        swa_locs = torch.full_like(full_locs, -1)
        full_tokens = torch.arange(64, 64 + bcap * context, device=device).view(
            bcap, context
        )
        swa_shift = 1088
        backend = AscendAttnBackend.__new__(AscendAttnBackend)
        backend.device = device
        backend.page_size = page
        backend.max_context_len = context
        backend.is_hybrid_swa = backend.use_sliding_window_kv_pool = True
        backend.forward_metadata, backend.graph_mode = object(), False
        original_metadata = backend.forward_metadata
        backend.req_to_token_pool = types.SimpleNamespace(req_to_token=full_tokens)
        backend.full_to_swa_index_mapping = (
            torch.arange(64 + bcap * context, device=device) + swa_shift
        )
        backend.token_to_kv_pool = types.SimpleNamespace(
            get_key_buffer=lambda _: key, get_value_buffer=lambda _: value
        )
        backend._welm_flash_attn = cann_ops_transformer.flash_attn
        backend._welm_flash_attn_metadata = cann_ops_transformer.flash_attn_metadata
        backend.welm_flash_attn_mask = torch.ones(
            2048, 2048, device=device, dtype=torch.int8
        ).triu_(1)
        layer = types.SimpleNamespace(
            layer_id=int(swa),
            tp_q_head_num=nq,
            tp_k_head_num=nkv,
            tp_v_head_num=nkv,
            qk_head_dim=dim,
            v_head_dim=dim,
            sliding_window_size=63 if swa else -1,
            scaling=dim**-0.5,
        )
        sinks = (torch.linspace(-1, 1, nq, device=device, dtype=torch.float32)
                 if with_sinks else None)
        base = backend.create_welm_prefill_graph_metadata(bcap)
        state = copy.copy(base)
        state.welm_flash_schedules = {}
        if mirror:
            state.block_tables = base.block_tables[:qrows]
            state.block_tables_swa = base.block_tables_swa[:qrows]
            state.welm_flash_seqused_kv = base.welm_flash_seqused_kv[:qrows]
            state.welm_flash_cu_seqlens_q = torch.arange(
                qrows + 1, dtype=torch.int32, device=device
            )
            state.welm_flash_seqused_q = torch.ones(qrows, dtype=torch.int32, device=device)
            state.welm_flash_mirror_q_lengths = (
                state.welm_flash_cu_seqlens_q,
                state.welm_flash_seqused_q,
            )
        else:
            state.welm_flash_cu_seqlens_q = torch.zeros(
                bcap + 1, dtype=torch.int32, device=device
            )
        adapter = WelmPrefillGraphAdapter.__new__(WelmPrefillGraphAdapter)
        adapter.backend, adapter.current_flash_metadata = backend, state
        adapter.pad_mirror = padded
        adapter.mirror_q_used = state.welm_flash_seqused_q
        adapter.full_write_locs, adapter.swa_write_locs = full_locs, swa_locs

        def prepare(lengths, kv_lengths):
            fb = types.SimpleNamespace(
                batch_size=len(lengths),
                seq_lens_cpu=torch.tensor(kv_lengths, device="cpu"),
                seq_lens=torch.tensor(kv_lengths, device=device),
                extend_seq_lens=torch.tensor(lengths, device=device),
                req_pool_indices=torch.arange(len(lengths), device=device),
            )
            backend.prepare_welm_prefill_graph_metadata(base, fb)
            adapter._prepare_mirror_requests(len(lengths))
            if not mirror:
                host = torch.tensor(
                    padded_flash_cu_seqlens(lengths, capacity, bcap),
                    dtype=torch.int32,
                    device="cpu",
                    pin_memory=True,
                )
                state.welm_flash_cu_seqlens_q.copy_(host, non_blocking=True)
            new_locs = torch.cat(
                [
                    full_tokens[i, length - e : length]
                    for i, (e, length) in enumerate(zip(lengths, kv_lengths))
                ]
            )
            full_locs.fill_(-1)
            swa_locs.fill_(-1)
            full_locs[: sum(lengths)].copy_(new_locs)
            swa_locs[: sum(lengths)].copy_(new_locs + swa_shift)

        def body():
            state.welm_flash_schedules.clear()
            return adapter.flash(
                layer,
                q,
                None if mirror else k,
                None if mirror else v,
                mirror=mirror,
                save_kv_cache=True,
                sinks=sinks,
            )

        pool = torch.npu.graph_pool_handle()

        def capture(fn):
            for _ in range(2):
                fn()
            torch.npu.synchronize()
            graph = BreakableCUDAGraph()
            with BreakableCUDAGraphCapture(cuda_graph=graph, pool=pool):
                result = fn()
            self.assertEqual(len(graph._segments), 1)
            self.assertEqual(len(graph._break_fns), 0)
            return graph, result

        prepare([17, 23], [145, 177])
        writers = {}
        if mirror:
            for t in (128, 64):
                writers[t], _ = capture(
                    lambda t=t: backend.write_welm_prefill_graph_kv(
                        layer, k[:t], v[:t], full_locs[:t], swa_locs[:t]
                    )
                )
        graph, output = capture(body)
        if mirror:
            cases = (
                ([17, 23], [145, 177]),
                ([61, 42], [189, 170]),
                ([1, 1], [193, 129]),
                ([17, 23], [145, 177]),
            )
            if padded:
                cases = (
                    ([17, 1, 1, 1], [145, 63, 64, 65]),
                    ([61, 1, 1], [189, 170, 65]),
                    ([1], [193]),
                    ([17, 23, 1, 1], [145, 177, 64, 65]),
                    ([1, 1], [64, 65]),
                )
        else:
            cases = (
                ([17, 23], [145, 177]),
                ([95], [223]),
                ([1, 7, 3, 54], [193, 135, 195, 182]),
                ([17, 23], [145, 177]),
            )
        checks = []
        for step, (lengths, kv_lengths) in enumerate(cases):
            prepare(lengths, kv_lengths)
            q.fill_((step + 1) / 13)
            k.copy_(torch.randn_like(k))
            v.copy_(torch.randn_like(v))
            slots = swa_locs if swa else full_locs
            real = sum(lengths)
            expected_key, expected_value = key.clone(), value.clone()
            expected_key.view(-1, nkv, dim)[slots[:real]] = k[:real].view(real, nkv, dim)
            expected_value.view(-1, nkv, dim)[slots[:real]] = v[:real].view(real, nkv, dim)
            if mirror:
                writers[64 if real <= 64 else 128].replay()
            if padded:
                # A replay must initialize every physical row, including a
                # request that was real in the preceding batch but is now idle.
                output.fill_(float("nan"))
            graph.replay()
            reference = self._reference(
                q,
                expected_key,
                expected_value,
                full_tokens + (swa_shift if swa else 0),
                lengths,
                kv_lengths,
                layer,
                sinks,
                mirror,
            )
            checks.append((output.clone(), reference, False))
            checks.extend(
                ((key.clone(), expected_key, True), (value.clone(), expected_value, True))
            )
            self.assertIs(backend.forward_metadata, original_metadata)
            self.assertFalse(backend.graph_mode)
        # No per-batch host fence: expose stale buffers and queued-copy races.
        torch.npu.synchronize()
        for actual, expected, exact in checks:
            torch.testing.assert_close(
                actual,
                expected,
                rtol=0 if exact else 0.02,
                atol=0 if exact else 0.02,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
