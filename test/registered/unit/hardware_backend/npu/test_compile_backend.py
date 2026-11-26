"""CPU-only checks for regional compile configuration and capture lifecycle."""

import importlib.util
import os
import sys
import unittest
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.hardware_backend.npu import compile_backend as cb
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestNpuCompileBackend(unittest.TestCase):
    def test_config_validation(self):
        defaults = dict(
            config=cb.NpuCompileConfig(static_kernel=True),
            device="npu",
            graph_enabled=True,
            enable_torch_compile=False,
            world_size=1,
        )
        cb.validate_npugraph_ex_config(**defaults)
        for invalid in (
            dict(device="cuda"),
            dict(graph_enabled=False),
            dict(enable_torch_compile=True),
        ):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                cb.validate_npugraph_ex_config(**(defaults | invalid))
        with self.assertRaisesRegex(ValueError, "requires"):
            cb.NpuCompileConfig(super_kernel=True)
        for local_size in ("", "0", "-1", "invalid", "9", "4"):
            with self.subTest(local_size=local_size), patch.dict(
                os.environ, {"LOCAL_WORLD_SIZE": local_size}
            ):
                context = (
                    nullcontext()
                    if local_size == "4"
                    else self.assertRaises(ValueError)
                )
                with context:
                    cb.validate_npugraph_ex_config(**(defaults | dict(world_size=8)))

    def test_backend_layouts_and_transitive_import_failure(self):
        process_options = Mock()
        module = SimpleNamespace(_process_kwargs_options=process_options)
        package = SimpleNamespace(CompilerConfig=Mock(), get_npu_backend=Mock())
        current = "npugraph_ex.configs.compiler_config"
        missing = ModuleNotFoundError(name=current)
        for imports in ([module], [missing, module], [SimpleNamespace(), module]):
            with self.subTest(imports=imports), patch.dict(
                sys.modules, {"npugraph_ex": package}
            ), patch.object(cb, "import_module", side_effect=imports):
                cb.get_npugraph_ex_backend(cb.NpuCompileConfig(True, True))
                options = process_options.call_args.args[1]["options"]
                self.assertEqual(
                    options,
                    dict(
                        force_eager=True,
                        inplace_pass=False,
                        clone_input=False,
                        clone_output=False,
                        static_kernel_compile=True,
                        super_kernel_optimize=True,
                    ),
                )
        with patch.dict(sys.modules, {"npugraph_ex": package}), patch.object(
            cb, "import_module", return_value=module
        ):
            cb.get_npugraph_ex_backend(cb.NpuCompileConfig(True))
            self.assertNotIn(
                "super_kernel_optimize", process_options.call_args.args[1]["options"]
            )
        failure = ModuleNotFoundError(name="missing_transitive_dependency")
        with patch.dict(sys.modules, {"npugraph_ex": package}), patch.object(
            cb, "import_module", side_effect=failure
        ), self.assertRaises(ModuleNotFoundError) as raised:
            cb.get_npugraph_ex_backend(cb.NpuCompileConfig(True))
        self.assertIs(raised.exception, failure)


class TestNpuRegionalCapture(unittest.TestCase):
    def setUp(self):
        self.events = []
        events = self.events
        self.configure_cache = Mock()

        class Graph:
            def super_kernel_optimize(self):
                events.append("optimize")

            def replay(self):
                events.append("replay")

        @contextmanager
        def graph_context(graph, **kwargs):
            events.append("capture_begin")
            yield
            events.append("capture_end")

        def compile_fn(fn, **kwargs):
            self.configure_cache.assert_called_with()

            def compiled(*args, **kw):
                events.append("compiled")
                return fn(*args, **kw)

            return compiled

        self.torch = SimpleNamespace(
            compile=Mock(side_effect=compile_fn),
            no_grad=lambda: lambda fn: fn,
            npu=SimpleNamespace(
                NPUGraph=Graph,
                graph=graph_context,
                super_kernel_scope_begin=lambda name: events.append("scope_begin"),
                super_kernel_scope_end=lambda name: events.append("scope_end"),
            ),
        )
        self.enterContext(
            patch.dict(
                sys.modules, {"torch": self.torch, "torch_npu": SimpleNamespace()}
            )
        )
        self.backend_factory = self.enterContext(
            patch.object(cb, "get_npugraph_ex_backend")
        )
        self.region = cb.NpuCompileRegion("activation", lambda x: x + 1)
        self.runner = SimpleNamespace(
            device_module=SimpleNamespace(
                current_device=lambda: 0, synchronize=lambda: None
            ),
            enable_torch_compile=False,
            capture_forward_mode=SimpleNamespace(is_decode=lambda: True),
            capture_bs=[1, 4],
            model_runner=SimpleNamespace(
                model=SimpleNamespace(npu_compile_regions=(self.region,)),
                server_args=SimpleNamespace(torch_compile_max_bs=2),
                tp_group=SimpleNamespace(barrier=lambda: None),
                device="npu",
                is_draft_worker=False,
            ),
        )
        # Load the real NPU backend, replacing only unrelated runtime dependencies.
        stubs = {
            "numpy": SimpleNamespace(),
            "sglang.srt.compilation.torch_compile_decoration": SimpleNamespace(
                set_torch_compile_cache_config=self.configure_cache
            ),
            "sglang.srt.constants": SimpleNamespace(GPU_MEMORY_TYPE_CUDA_GRAPH="graph"),
            "sglang.srt.distributed.device_communicators.pynccl_allocator": (
                SimpleNamespace(set_graph_pool_id=lambda _: None)
            ),
            "sglang.srt.distributed.parallel_state": SimpleNamespace(
                get_world_group=lambda: SimpleNamespace(world_size=1)
            ),
            "sglang.srt.model_executor.runner.shape_key": SimpleNamespace(
                ShapeKey=object
            ),
            "sglang.srt.model_executor.runner_backend.base_cuda_graph_backend": (
                SimpleNamespace(BaseCudaGraphBackend=object)
            ),
            "sglang.srt.utils": SimpleNamespace(
                empty_context=nullcontext, get_bool_env_var=lambda _: False
            ),
            "sglang.srt.utils.torch_memory_saver_adapter": SimpleNamespace(
                TorchMemorySaverAdapter=SimpleNamespace(create=lambda **kw: None)
            ),
        }
        self.enterContext(patch.dict(sys.modules, stubs))
        path = Path(cb.__file__).parent / "graph_runner" / "npu_cudagraph_backend.py"
        spec = importlib.util.spec_from_file_location("_npu_backend_test", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.backend_class = module.NPUCudaGraphBackend

    def make_backend(self, config):
        with patch.object(cb.NpuCompileConfig, "from_env", return_value=config):
            return self.backend_class(self.runner)

    def test_static_and_superkernel_capture_replay_and_bucket_fallback(self):
        for super_kernel in (False, True):
            with self.subTest(super_kernel=super_kernel):
                self.region = cb.NpuCompileRegion("activation", lambda x: x + 1)
                self.runner.model_runner.model.npu_compile_regions = (self.region,)
                with patch.object(
                    self.torch.npu,
                    "super_kernel_scope_begin",
                    self.torch.npu.super_kernel_scope_begin if super_kernel else None,
                ):
                    backend = self.make_backend(
                        cb.NpuCompileConfig(True, super_kernel)
                    )
                # ShapeKey is hashable in production.
                key = type("Key", (), {"size": 1})()
                self.events.clear()
                backend.capture_one(key, lambda: self.region(2))
                expected = ["compiled", "compiled", "capture_begin"]
                expected += (
                    ["scope_begin", "compiled", "scope_end"]
                    if super_kernel
                    else ["compiled"]
                )
                expected += ["capture_end"] + (
                    ["optimize"] if super_kernel else []
                )
                self.assertEqual(self.events, expected)
                self.assertEqual(backend.replay(key, None), 3)
                self.assertEqual(self.events[-1], "replay")
                self.assertIsNone(self.region._session)
                self.assertEqual(self.region(4), 5)
                self.events.clear()
                backend.capture_one(
                    type("Key", (), {"size": 4})(), lambda: self.region(2)
                )
                self.assertEqual(self.events, ["capture_begin", "capture_end"])

    def test_disabled_path_needs_no_model_regions_or_compiler(self):
        self.runner.model_runner.model.npu_compile_regions = ()
        backend = self.make_backend(cb.NpuCompileConfig())
        backend.capture_one(type("Key", (), {"size": 1})(), lambda: 7)
        self.backend_factory.assert_not_called()
        self.torch.compile.assert_not_called()
        self.configure_cache.assert_not_called()
        self.assertEqual(self.events, ["capture_begin", "capture_end"])

    def test_optimize_failure_restores_eager_and_does_not_publish_graph(self):
        backend = self.make_backend(cb.NpuCompileConfig(True, True))
        key = type("Key", (), {"size": 1})()
        with patch.object(
            self.torch.npu.NPUGraph,
            "super_kernel_optimize",
            side_effect=RuntimeError("failed"),
        ):
            with self.assertRaisesRegex(RuntimeError, "failed"):
                backend.capture_one(key, lambda: self.region(2))
        self.assertNotIn(key, backend._graphs)
        self.assertIsNone(self.region._session)
        self.assertEqual(self.region(4), 5)

    def test_missing_registration_bucket_and_actual_execution_are_errors(self):
        config = cb.NpuCompileConfig(True)
        self.runner.model_runner.model.npu_compile_regions = ()
        with self.assertRaisesRegex(ValueError, "no npu_compile_regions"):
            self.make_backend(config)
        self.runner.model_runner.model.npu_compile_regions = (self.region,)
        self.runner.model_runner.server_args.torch_compile_max_bs = 0
        with self.assertRaisesRegex(ValueError, "No Decode capture bucket"):
            self.make_backend(config)
        self.runner.model_runner.server_args.torch_compile_max_bs = 2
        backend = self.make_backend(config)
        with self.assertRaisesRegex(RuntimeError, "No registered"):
            backend.capture_one(type("Key", (), {"size": 1})(), lambda: 7)
        self.assertFalse(backend._graphs)
        self.assertIsNone(self.region._session)

    def test_unsupported_mode_and_superkernel_api_fail_before_compilation(self):
        self.runner.model_runner.is_draft_worker = True
        with self.assertRaisesRegex(ValueError, "Decode only"):
            self.make_backend(cb.NpuCompileConfig(True))
        self.runner.model_runner.is_draft_worker = False
        with patch.object(self.torch.npu, "super_kernel_scope_begin", None):
            with self.assertRaisesRegex(RuntimeError, "scope APIs"):
                self.make_backend(cb.NpuCompileConfig(True, True))
        self.backend_factory.assert_not_called()
        self.configure_cache.assert_not_called()


if __name__ == "__main__":
    unittest.main()
