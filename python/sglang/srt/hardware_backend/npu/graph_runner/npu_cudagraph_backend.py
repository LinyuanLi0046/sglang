"""NPUCudaGraphBackend — Ascend NPU full-graph capture (torch.npu.NPUGraph).

Mirrors FullCudaGraphBackend with two differences:
  - Captures via torch.npu.graph(...) into torch.npu.NPUGraph.
  - replay_with_input_update(shape_key, seq_lens, attr_name) rebinds
    the recorded graph's input bindings for variable seq_lens at replay
    time (NPU's NPUGraph.update(...) API).

torch.npu is imported lazily inside methods so the module loads on
non-NPU hosts.
"""

from __future__ import annotations

import threading
from contextlib import AbstractContextManager, contextmanager
from dataclasses import replace
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

import numpy as np
import torch

from sglang.srt.constants import GPU_MEMORY_TYPE_CUDA_GRAPH
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    set_graph_pool_id,
)
from sglang.srt.hardware_backend.npu.compile_backend import (
    NpuCompileConfig,
    NpuCompileSession,
    prepare_npu_compile_regions,
    validate_npugraph_ex_config,
)
from sglang.srt.model_executor.runner.shape_key import ShapeKey
from sglang.srt.model_executor.runner_backend.base_cuda_graph_backend import (
    BaseCudaGraphBackend,
)
from sglang.srt.utils import empty_context, get_bool_env_var
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.runner.base_cuda_graph_runner import (
        BaseCudaGraphRunner,
    )


class NPUCudaGraphBackend(BaseCudaGraphBackend):
    """One torch.npu.NPUGraph per shape; attention metadata captured
    inside the graph. replay_with_input_update substitutes fresh
    seq_lens without re-recording."""

    def __init__(
        self,
        cuda_graph_runner: BaseCudaGraphRunner,
        *,
        enable_memory_saver: bool = False,
    ) -> None:
        self._graphs: Dict[Any, Any] = {}
        self._outputs: Dict[Any, Any] = {}
        self._pool = None
        self._device_module = cuda_graph_runner.device_module
        self._device_id = self._device_module.current_device()
        self._tp_group = cuda_graph_runner.model_runner.tp_group
        self._capture_stream = None
        self._memory_saver_adapter: Optional[Any] = TorchMemorySaverAdapter.create(
            enable=enable_memory_saver
            and get_bool_env_var("SGLANG_MEMORY_SAVER_CUDA_GRAPH")
        )
        self._enable_torch_compile = getattr(
            cuda_graph_runner, "enable_torch_compile", False
        )
        self._compile_config = NpuCompileConfig.from_env()
        self._compile_bs = set()
        self._compile_regions = ()
        model_runner = cuda_graph_runner.model_runner
        compile_target = self._compile_config.static_kernel and (
            not model_runner.is_draft_worker
            and (
                cuda_graph_runner.capture_forward_mode.is_decode()
                or cuda_graph_runner.capture_forward_mode.is_target_verify()
            )
        )
        if compile_target:
            # Bind after weight postprocessing, and never initialize target-model
            # regions on the NextN worker which inherits the same model class.
            initialize = getattr(model_runner.model, "init_npu_compile_regions", None)
            if initialize is not None:
                initialize(capture_forward_mode=cuda_graph_runner.capture_forward_mode)
            self._compile_config = replace(
                self._compile_config,
                super_kernel_options=getattr(
                    model_runner.model, "npu_super_kernel_options", ()
                ),
            )
            # An explicit empty declaration is a model's opt-out (e.g. both
            # WeLM region switches are off). Unintegrated models still error.
            compile_target = (
                getattr(model_runner.model, "npu_compile_regions", None) != ()
            )
        if compile_target:
            from sglang.srt.distributed.parallel_state import get_world_group

            args = model_runner.server_args
            if getattr(cuda_graph_runner, "ragged_verify_mode", False):
                raise ValueError("Regional NPU compilation requires fixed-width Verify")
            validate_npugraph_ex_config(
                config=self._compile_config,
                device=model_runner.device,
                graph_enabled=True,
                enable_torch_compile=self._enable_torch_compile,
                world_size=get_world_group().world_size,
            )
            # compile_bs is empty when whole-model compilation is disabled.
            # Select local buckets directly from the actual capture table.
            self._compile_bs = {
                bs
                for bs in cuda_graph_runner.capture_bs
                if bs <= args.torch_compile_max_bs
            }
            if not self._compile_bs:
                raise ValueError(
                    "No Decode/Verify capture bucket fits --torch-compile-max-bs"
                )
            self._compile_regions = prepare_npu_compile_regions(
                model_runner.model, self._compile_config
            )
        # Event of the last replay; the next rebind waits on it so the device is
        # done reading the host-side seq_lens array before we rewrite it.
        self._rebind_fence = None

    @contextmanager
    def capture_session(self, stream):
        if self._pool is None:
            self._pool = self._device_module.graph_pool_handle()
        set_graph_pool_id(self._pool)
        self._capture_stream = stream
        try:
            yield
        finally:
            self._capture_stream = None

    def capture_one(
        self,
        shape_key: ShapeKey,
        forward_fn: Callable[[], Any],
        capture_inputs: Optional[Any] = None,
        post_warmup_hook: Optional[Callable[[], None]] = None,
    ) -> None:
        compile_session = (
            NpuCompileSession(self._compile_regions, self._compile_config)
            if shape_key.size in self._compile_bs
            else None
        )
        with compile_session.activate() if compile_session else empty_context():
            self._capture_one(shape_key, forward_fn, post_warmup_hook, compile_session)

    def _capture_one(
        self,
        shape_key: ShapeKey,
        forward_fn: Callable[[], Any],
        post_warmup_hook: Optional[Callable[[], None]],
        compile_session: Optional[NpuCompileSession],
    ) -> None:
        import torch_npu  # noqa: F401  (verifies NPU availability)

        # Two warmups so kernels are loaded and one-time setup is paid before capture.
        # post_warmup_hook lets the attention backend reset state that warmup mutated.
        for _ in range(2):
            self._device_module.synchronize()
            self._tp_group.barrier()
            forward_fn()
            if post_warmup_hook is not None:
                post_warmup_hook()

        graph = torch.npu.NPUGraph()

        if self._enable_torch_compile:
            skip_guard_context = torch.compiler.set_stance(skip_guard_eval_unsafe=True)
        else:
            skip_guard_context = empty_context()

        graph_ctx: Callable[..., AbstractContextManager]
        if (
            self._memory_saver_adapter is not None
            and self._memory_saver_adapter.enabled
        ):
            graph_ctx = partial(
                self._memory_saver_adapter.cuda_graph,
                tag=GPU_MEMORY_TYPE_CUDA_GRAPH,
            )
        else:
            graph_ctx = torch.npu.graph

        with (
            skip_guard_context,
            graph_ctx(
                graph,
                pool=self._pool,
                stream=self._capture_stream,
                auto_dispatch_capture=True,
            ),
            compile_session.capture() if compile_session else empty_context(),
        ):
            out = forward_fn()

        if compile_session is not None:
            compile_session.optimize(graph)

        self._graphs[shape_key] = graph
        self._outputs[shape_key] = out

    def can_run(self, forward_batch: ForwardBatch, shape_key: ShapeKey) -> bool:
        return shape_key in self._graphs

    @contextmanager
    def replay_session(self):
        yield

    def replay(
        self,
        shape_key: ShapeKey,
        static_forward_batch: ForwardBatch,
        **kwargs,
    ) -> Any:
        self._graphs[shape_key].replay()
        return self._outputs[shape_key]

    def replay_with_input_update(
        self,
        shape_key: ShapeKey,
        seq_lens: Any,
        attr_name: str = None,
        attr_type: Any = None,
        cpu_update_input: list = None,
    ) -> Any:
        """Rebind seq_lens on the recorded NPU graph in a background
        thread, then replay. Used when the model is not deepseek-nsa.

        Two calling conventions:
        1. (legacy) seq_lens + attr_name + attr_type:
           Constructs cpu_update_input=[{attr_name: seq_lens}] internally.
        2. cpu_update_input: A list of {attr_name: seq_lens} dicts,
           one per speculative step.  Used by EAGLE draft runners.
        """
        if cpu_update_input is None:
            if isinstance(attr_type, torch.Tensor):
                seq_lens = torch.from_numpy(np.array(seq_lens).astype(np.int32))
            cpu_update_input = [{attr_name: seq_lens}]

        graph = self._graphs[shape_key]

        if self._rebind_fence is not None:
            self._rebind_fence.synchronize()

        def _update():
            self._device_module.set_device(self._device_id)
            graph.update(cpu_update_input=cpu_update_input)

        thread = threading.Thread(target=_update)
        thread.start()
        graph.replay()
        thread.join()

        fence = self._device_module.Event()
        fence.record()
        self._rebind_fence = fence
        return self._outputs[shape_key]

    def cleanup(self) -> None:
        self._graphs.clear()
        self._outputs.clear()
        self._pool = None
