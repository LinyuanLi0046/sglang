"""Ascend smoke test for captured WeLM attention and phase handoff lifetimes.

The attention callable below deliberately uses small tensor operations: this
checks NPU graph/side-stream lifetimes, not native Flash numerics or HCCL.
Native operators are covered by test_welmv4_prefill_flash_graph_npu.py.
Run in an installed SGLang Ascend environment: python <this file>.
"""

import types
import unittest

try:
    import torch
    import torch_npu  # noqa: F401
    from torch_npu.contrib import transfer_to_npu  # noqa: F401

    HAS_NPU = torch.npu.is_available()
except ImportError:
    HAS_NPU = False

if HAS_NPU:
    from sglang.srt.model_executor.forward_batch_info import ForwardMode
    from sglang.srt.model_executor.model_runner_components.ngram_embedding_manager import (
        NgramEmbeddingManager,
    )
    from sglang.srt.model_executor.runner.welm_prefill_graph import (
        WelmPrefillGraphAdapter,
        padded_rope_tiles,
    )
    from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
        BreakableCUDAGraph,
        BreakableCUDAGraphCapture,
    )


@unittest.skipUnless(HAS_NPU, "requires torch_npu and an Ascend device")
class TestWeLMNpuCapturedAttention(unittest.TestCase):
    def test_replay_reads_live_rows_and_metadata_with_captured_side_stream(self):
        for mirror in (False, True):
            with self.subTest(mirror=mirror):
                self._check_replays(mirror)

    def _check_replays(self, mirror):
        capacity, batch_size, width = 8, 2, 4
        device = torch.device("npu", torch.npu.current_device())
        x = torch.ones((capacity, width), device=device, dtype=torch.bfloat16)
        kv_input = torch.ones_like(x)
        weight = torch.eye(width, device=device, dtype=x.dtype)
        swa = torch.arange(capacity, device=device, dtype=torch.int64)
        valid = torch.zeros((capacity, 1), device=device, dtype=torch.bool)
        kv_cache = torch.zeros_like(x)
        flash_state = types.SimpleNamespace(bias=torch.zeros(1, device=device))
        adapter = WelmPrefillGraphAdapter.__new__(WelmPrefillGraphAdapter)
        adapter.device = device
        adapter.rope_tiles = {}
        adapter.max_requests = batch_size
        adapter.tail_indices = torch.zeros(batch_size, dtype=torch.int64, device=device)
        adapter.full_write_locs = adapter.swa_write_locs = swa
        adapter.current_flash_metadata = flash_state
        adapter.backend = types.SimpleNamespace(
            forward_metadata=types.SimpleNamespace(swa_out_cache_loc=swa),
            token_to_kv_pool=types.SimpleNamespace(
                get_key_buffer=lambda _: kv_cache, get_value_buffer=lambda _: kv_cache
            ),
            write_welm_prefill_graph_kv=lambda layer, k, v, *slots: kv_cache.copy_(
                torch.where(valid, k, 0)
            ),
        )

        def set_batch(lengths, bias):
            adapter.current_batch = types.SimpleNamespace(
                extend_seq_lens_cpu=list(lengths),
                batch_size=batch_size,
                out_cache_loc=swa,
                num_token_non_padded_cpu=sum(lengths),
            )
            flash_state.bias.fill_(bias)
            valid[:sum(lengths)].fill_(True)
            valid[sum(lengths):].fill_(False)

        def attention(q, k, v, layer, sinks, *, graph_metadata, mirror_prefill):
            out = q + k.sum(dim=0) + graph_metadata.bias
            return out if mirror_prefill else torch.where(valid, out, 0)

        adapter.backend._forward_welm_flash_attention = attention
        layer = types.SimpleNamespace(layer_id=0)

        side = torch.npu.Stream()

        def body():
            main = torch.npu.current_stream()
            side.wait_stream(main)
            with torch.npu.stream(side):
                k = kv_input + 1
            q = x @ weight
            main.wait_stream(side)
            k.record_stream(main)
            if mirror:
                # Stand-in for the T graph's consumer cache write.
                kv_cache.copy_(torch.where(valid, k, 0))
                q = q[:batch_size]
            return adapter.flash(
                layer, q, k, k, mirror=mirror, save_kv_cache=True
            ) * 2

        set_batch([3, 5], 0)
        for _ in range(2):
            body()
        torch.npu.synchronize()
        graph = BreakableCUDAGraph()
        capture_stream = torch.npu.Stream()
        with BreakableCUDAGraphCapture(
            graph, pool=torch.npu.graph_pool_handle(), stream=capture_stream
        ):
            output = body()
        self.assertEqual(len(graph._segments), 2)
        self.assertEqual(len(graph._break_fns), 1)

        # Same captured shape, different live metadata; return to the first
        # request layout to expose stale cache/bridge contents.
        snapshots = []
        for lengths, value, bias in (([2, 3], 1, 2), ([1, 2], 3, 7), ([2, 3], 5, 1)):
            x.fill_(value)
            kv_input.fill_(value + 1)
            set_batch(lengths, bias)
            adapter._prepare_tiles(adapter.current_batch, capacity)
            tiles = adapter.current_batch.welmv4_rope_segment_tile_starts.clone()
            graph.replay()
            actual = output.clone()
            rows = batch_size if mirror else sum(lengths)
            expected = torch.zeros_like(output)
            expected[:rows] = (
                (x @ weight)[:rows] + (kv_input + 1)[: sum(lengths)].sum(dim=0) + bias
            ) * 2
            snapshots.append(
                (actual, expected, tiles, padded_rope_tiles(lengths, capacity))
            )
            self.assertIs(adapter.backend.forward_metadata.swa_out_cache_loc, swa)
        # No per-batch host fence: exercise the same queued-work lifetime as
        # scheduler overlap. Only the test's final comparison waits for device.
        torch.npu.synchronize()
        for actual, expected, tiles, expected_tiles in snapshots:
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            self.assertEqual(tiles.cpu().tolist(), expected_tiles)


@unittest.skipUnless(HAS_NPU, "requires torch_npu and an Ascend device")
class TestNgramHistoryStaging(unittest.TestCase):
    def test_queued_history_updates_on_one_or_two_streams(self):
        for separate_streams in (False, True):
            with self.subTest(separate_streams=separate_streams):
                self._check_history(separate_streams)

    def _check_history(self, separate_streams):
        device = torch.device("npu", torch.npu.current_device())
        table = torch.full((8, 16), -1, dtype=torch.int32, device=device)
        rows = torch.tensor([1, 3, 5], dtype=torch.int64, device=device)
        manager = NgramEmbeddingManager(enabled=True, table=table, n=3, k=0)
        schedule = torch.npu.current_stream()
        forward = torch.npu.Stream() if separate_streams else schedule
        snapshots, batches = [], []
        expected = torch.full((8, 16), -1, dtype=torch.int32)
        for step in range(8):
            # Protect shared-table writes from the previous consumer, using
            # only a stream dependency; do not drain the CPU between batches.
            if separate_streams:
                schedule.wait_stream(forward)
            reqs = [
                types.SimpleNamespace(
                    prefix_indices=list(range(start)),
                    extend_range=types.SimpleNamespace(length=length),
                    origin_input_ids=list(range(base, base + 8)), output_ids=[],
                )
                for start, length, base in (
                    (0, 3, 10 + step * 100),
                    (1, 2, 20 + step * 100),
                    (6, 2, 30 + step * 100),
                )
            ]
            batch = types.SimpleNamespace(
                reqs=reqs, forward_mode=ForwardMode.EXTEND, req_pool_indices=rows,
            )
            batches.append(batch)  # Keep cross-stream mask consumers alive.
            manager.prepare_for_forward(batch, chunked_req=reqs[1])
            expected[1, :3] = torch.arange(10, 13) + step * 100
            expected[3, :3] = torch.arange(20, 23) + step * 100
            expected[5, 4:8] = torch.arange(34, 38) + step * 100
            reqs[0].origin_input_ids[:] = [-999] * 8
            if separate_streams:
                forward.wait_stream(schedule)
            with torch.npu.stream(forward):
                snapshots.append((table.clone(), expected.clone()))
                snapshots.append((
                    batch.ne_skip_token_table_update.clone(),
                    torch.tensor([False, True, False]),
                ))
        if separate_streams:
            schedule.wait_stream(forward)
        req = types.SimpleNamespace(
            origin_input_ids=[71, 72], output_ids=[73], req_pool_idx=6, rid="pd",
            ngram_token_table_needs_init=True,
        )
        batch = types.SimpleNamespace(reqs=[req], forward_mode=ForwardMode.DECODE)
        manager.prepare_for_forward(batch, chunked_req=None)
        manager.prepare_for_forward(batch, chunked_req=None)
        self.assertFalse(req.ngram_token_table_needs_init)
        expected[6, :3] = torch.tensor([71, 72, 73])
        if separate_streams:
            forward.wait_stream(schedule)
        with torch.npu.stream(forward):
            snapshots.append((table.clone(), expected))
        torch.npu.synchronize()
        for actual, reference in snapshots:
            torch.testing.assert_close(actual.cpu(), reference, rtol=0, atol=0)


@unittest.skipUnless(HAS_NPU, "requires torch_npu and an Ascend device")
class TestSplitPrefillGraphs(unittest.TestCase):
    def test_single_operand_rope_matches_reference_without_double_rotation(self):
        from sglang.srt.layers.welmv4_npu_op import welmv4_inplace_rope_single_npu

        device = torch.device("npu", torch.npu.current_device())
        head_dim, rope_dim, capacity = 256, 64, 128
        angles = torch.arange(512 * 32, device=device, dtype=torch.float32).reshape(512, 32) / 1000
        cache = torch.cat((angles.cos(), angles.sin()), dim=-1)
        checks = []
        for heads, lengths, segmented in ((1, [65, 17], True), (2, [127], True),
                                           (4, [128], False), (6, [4], False),
                                           (12, [2], False), (24, [1], False)):
            rows = capacity if segmented else sum(lengths)
            tensor = torch.randn(rows, heads * head_dim, device=device, dtype=torch.bfloat16)
            original = tensor.clone()
            positions_cpu = [pos + request * 37 for request, length in enumerate(lengths)
                             for pos in range(length)] + [0] * (rows - sum(lengths))
            positions = torch.tensor(positions_cpu, device=device, dtype=torch.int64)
            tiles = torch.tensor(padded_rope_tiles(lengths, rows, 4), device=device,
                                 dtype=torch.int32) if segmented else None
            real = sum(lengths)
            expected = original.clone().view(rows, heads, head_dim)
            x = original[:real].view(real, heads, head_dim)[..., -rope_dim:].float()
            cos, sin = cache[positions[:real]].chunk(2, dim=-1)
            x1, x2 = x.chunk(2, dim=-1)
            expected[:real, :, -rope_dim:] = torch.cat(
                (x1 * cos[:, None] - x2 * sin[:, None],
                 x1 * sin[:, None] + x2 * cos[:, None]), dim=-1
            ).to(tensor.dtype)
            welmv4_inplace_rope_single_npu(
                tensor, positions, cache, head_dim=head_dim, rope_dim=rope_dim,
                segment_tile_starts=tiles,
            )
            checks.append((tensor.clone(), expected.reshape_as(tensor)))
        torch.npu.synchronize()
        for actual, expected in checks:
            torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.02)

    def test_two_graph_families_reuse_buffers_across_queued_t_b_changes(self):
        from sglang.srt.model_executor.runner_backend.breakable_cuda_graph_backend import (
            BreakableCudaGraphBackend,
        )

        device = torch.device("npu", torch.npu.current_device())
        x = torch.zeros((128, 4), dtype=torch.bfloat16, device=device)
        kv = torch.zeros_like(x)
        tails = torch.zeros(4, dtype=torch.int64, device=device)
        handoff = torch.zeros((4, 4), dtype=x.dtype, device=device)
        backend = BreakableCudaGraphBackend.__new__(BreakableCudaGraphBackend)
        backend._device_module = torch.npu
        backend._tp_group = types.SimpleNamespace(barrier=lambda: None)
        backend._pool = torch.npu.graph_pool_handle()
        backend._capture_stream = torch.npu.Stream()
        backend._shared_output_buffer = None
        backend._debug_eager = False
        backend._graphs, backend._outputs, backend._capture_inputs = {}, {}, {}
        adapter = WelmPrefillGraphAdapter.__new__(WelmPrefillGraphAdapter)
        adapter.prune = True
        adapter.capture_keys, adapter.captured_states = set(), {}
        adapter.runner = types.SimpleNamespace(backend=backend)
        valid = torch.zeros((128, 1), dtype=torch.bool, device=device)
        adapter.current_flash_metadata = object()
        adapter.backend = types.SimpleNamespace(
            token_to_kv_pool=types.SimpleNamespace(
                get_key_buffer=lambda _: kv, get_value_buffer=lambda _: kv
            ),
            _forward_welm_flash_attention=lambda q, k, *args, **kw: q + k.sum(0),
        )
        attention = types.SimpleNamespace(layer_id=0)

        def set_batch(lengths):
            adapter.current_batch = types.SimpleNamespace(
                batch_size=len(lengths), extend_seq_lens_cpu=list(lengths),
                out_cache_loc=torch.zeros(128, dtype=torch.int64, device=device),
            )
            valid[:sum(lengths)].fill_(True)
            valid[sum(lengths):].fill_(False)

        for capacity in (128, 64):
            set_batch([capacity])

            def prompt(capacity=capacity):
                kv.zero_()
                kv[:capacity].copy_(torch.where(valid[:capacity], x[:capacity] * 2, 0))
                handoff.copy_(x[:capacity].index_select(0, tails))
                return None

            key = adapter.key(capacity)
            backend.capture_one(key, prompt, capture_inputs=(x, kv, handoff, tails))
            adapter.capture_keys.add(key)
            self.assertIsNone(backend._shared_output_buffer)
        for bs in (4, 2, 1):
            set_batch([1] * bs)

            def mirror(bs=bs):
                return adapter.flash(attention, handoff[:bs].clone(), None, None,
                                     mirror=True, save_kv_cache=True) * 3

            key = adapter.mirror_key(bs)
            backend.capture_one(key, mirror, capture_inputs=(handoff, kv))
            adapter.capture_keys.add(key)
        self.assertEqual(len(backend._graphs), 5)
        snapshots = []
        cases = ((64, [3, 5], 1), (128, [91], 2), (64, [1, 2, 3, 4], 3),
                 (128, [42, 53], 4), (64, [3, 5], 5))
        for capacity, lengths, value in cases:
            set_batch(lengths)
            indices, offset = [], 0
            for length in lengths:
                offset += length
                indices.append(offset - 1)
            host = torch.tensor(indices + [0] * (4 - len(indices)), dtype=torch.int64,
                                device="cpu", pin_memory=True)
            tails.copy_(host, non_blocking=True)
            x.fill_(value)
            result = adapter.replay(adapter.key(capacity), adapter.current_batch)
            snapshots.append((result.clone(), (value + 2 * value * sum(lengths)) * 3))
        # No per-batch host fence; final copies expose overwritten handoffs.
        torch.npu.synchronize()
        for actual, value in snapshots:
            torch.testing.assert_close(actual, torch.full_like(actual, value), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
