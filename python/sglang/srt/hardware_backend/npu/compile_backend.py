"""Regional npugraph_ex compilation inside an SGLang-owned NPU graph.

This module has no torch/torch_npu/npugraph_ex import-time dependency. Models
must explicitly expose a tuple of NpuCompileRegion objects as
``model.npu_compile_regions``; no model is opted in implicitly.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass
from importlib import import_module
from typing import Callable

from sglang.srt.environ import envs


@dataclass(frozen=True)
class NpuCompileConfig:
    static_kernel: bool = False
    super_kernel: bool = False

    def __post_init__(self):
        if self.super_kernel and not self.static_kernel:
            raise ValueError(
                "SGLANG_NPU_ENABLE_SUPERKERNEL requires "
                "SGLANG_NPU_ENABLE_STATIC_KERNEL=1"
            )

    @classmethod
    def from_env(cls):
        return cls(
            static_kernel=envs.SGLANG_NPU_ENABLE_STATIC_KERNEL.get(),
            super_kernel=envs.SGLANG_NPU_ENABLE_SUPERKERNEL.get(),
        )


def validate_npugraph_ex_config(
    *,
    config: NpuCompileConfig,
    device: str,
    graph_enabled: bool,
    enable_torch_compile: bool,
    world_size: int,
):
    """Validate regional compilation without importing the NPU toolchain."""
    if not config.static_kernel:
        return
    if device != "npu" or not graph_enabled:
        raise ValueError(
            "Regional static kernel compilation requires NPU Decode/Verify Graph"
        )
    if enable_torch_compile:
        raise ValueError(
            "Regional static kernel compilation cannot be combined with "
            "--enable-torch-compile (whole-model compilation)"
        )
    if world_size > 1:
        try:
            local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "0"))
        except ValueError:
            local_world_size = 0
        if not 0 < local_world_size <= world_size:
            raise ValueError(
                "Multi-rank static kernel compilation requires LOCAL_WORLD_SIZE "
                "to be the number of ranks of this job on this node "
                "(a positive integer no greater than world_size)"
            )


def get_npugraph_ex_backend(config: NpuCompileConfig):
    """Build the backend; SGLang retains ownership of capture and replay."""
    import npugraph_ex

    module_name = "npugraph_ex.configs.compiler_config"
    try:
        config_module = import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name != module_name:
            raise
        config_module = None
    process_options = getattr(config_module, "_process_kwargs_options", None)
    if process_options is None:
        config_module = import_module("npugraph_ex.configs.npugraphex_config")
        process_options = config_module._process_kwargs_options

    compiler_config = npugraph_ex.CompilerConfig()
    options = {
        "force_eager": True,
        "inplace_pass": False,
        "clone_input": False,
        "clone_output": False,
        "static_kernel_compile": config.static_kernel,
    }
    if config.super_kernel:
        # Required during static compilation as well as after outer capture.
        options["super_kernel_optimize"] = True
    process_options(compiler_config, {"options": options})
    # init_npu_backend() already sets jit_compile=False once per worker.
    return npugraph_ex.get_npu_backend(compiler_config=compiler_config)


class NpuCompileRegion:
    """A persistent callable, eager except during an opted-in graph capture.

    ``fn`` must contain only the selected tensor computation. Keep mode checks,
    communication, shared-buffer notifications and cache ownership outside it.
    The owning model must retain this object for the lifetime of its graphs.
    """

    def __init__(
        self,
        name: str,
        fn: Callable,
        *,
        compile_fn: Callable | None = None,
        external_scope: bool = False,
    ):
        if not name.strip():
            raise ValueError("An NPU compile region needs a nonempty scope name")
        self.name = name
        self._eager = fn
        self._compile_fn = compile_fn if compile_fn is not None else fn
        self._external_scope = external_scope
        self._compiled = None
        self._config = None
        self._session = None

    @property
    def is_active(self):
        return self._session is not None

    @contextmanager
    def scope(self, *, exclude: bool = False):
        """Mark a group of tensor calls without putting stream waits into FX."""
        session = self._session
        if session is None or not session.capturing or not session.config.super_kernel:
            yield
            return
        import torch

        name = None if exclude else self.name
        torch.npu.super_kernel_scope_begin(name)
        try:
            yield
        finally:
            torch.npu.super_kernel_scope_end(name)

    def prepare(self, config: NpuCompileConfig, backend):
        if self._compiled is not None:
            if self._config != config:
                raise ValueError("Recreate NPU compile regions when changing config")
            return
        import torch

        self._compiled = torch.compile(
            torch.no_grad()(self._compile_fn),
            backend=backend,
            fullgraph=True,
            dynamic=False,
        )
        self._config = config

    def __call__(self, *args, **kwargs):
        session = self._session
        if session is None:
            return self._eager(*args, **kwargs)
        if session.capturing:
            if self not in session.warmed:
                raise RuntimeError(
                    f"NPU region {self.name!r} was not warmed before capture"
                )
            session.captured.add(self)
        else:
            session.warmed.add(self)

        if not self._external_scope:
            with self.scope():
                return self._compiled(*args, **kwargs)
        return self._compiled(*args, **kwargs)


class NpuCompileSession:
    """Warmup/capture state for one bucket; never used on the replay path."""

    def __init__(self, regions, config: NpuCompileConfig):
        self.regions = tuple(regions)
        self.config = config
        self.capturing = False
        self.warmed = set()
        self.captured = set()

    @contextmanager
    def activate(self):
        if any(region._session is not None for region in self.regions):
            raise RuntimeError("Concurrent/nested NPU regional capture is unsupported")
        for region in self.regions:
            region._session = self
        try:
            yield self
        finally:
            for region in self.regions:
                region._session = None

    @contextmanager
    def capture(self):
        self.capturing = True
        try:
            yield
        finally:
            self.capturing = False

    def optimize(self, graph):
        if not self.captured:
            raise RuntimeError("No registered NPU compile region ran during capture")
        if self.config.super_kernel:
            # In-place optimization of an already captured graph, before it
            # is published for replay. Do not replace graph with the return value.
            graph.super_kernel_optimize()


def prepare_npu_compile_regions(model, config: NpuCompileConfig):
    """Resolve explicit model opt-in and prepare reusable compiled callables."""
    regions = tuple(getattr(model, "npu_compile_regions", ()))
    if not regions:
        raise ValueError(
            "Regional static kernel compilation was enabled, but this model "
            "has no npu_compile_regions integration"
        )
    if any(not isinstance(region, NpuCompileRegion) for region in regions):
        raise TypeError(
            "model.npu_compile_regions must contain NpuCompileRegion objects"
        )
    if len({region.name for region in regions}) != len(regions):
        raise ValueError("NPU compile region scope names must be unique within a model")
    if config.super_kernel:
        import torch
        import torch_npu  # noqa: F401

        if not all(
            callable(getattr(torch.npu, name, None))
            for name in ("super_kernel_scope_begin", "super_kernel_scope_end")
        ) or not callable(getattr(torch.npu.NPUGraph, "super_kernel_optimize", None)):
            raise RuntimeError(
                "SuperKernel requires torch_npu scope APIs and "
                "NPUGraph.super_kernel_optimize"
            )
    from sglang.srt.compilation.torch_compile_decoration import (
        set_torch_compile_cache_config,
    )

    # Whole-model compilation is disabled on this path, so the decode runner
    # does not configure Dynamo. Reuse its cache limits before region warmup,
    # without enabling the unrelated Inductor/Triton configuration.
    set_torch_compile_cache_config()
    backend = get_npugraph_ex_backend(config)
    for region in regions:
        region.prepare(config, backend)
    return regions
