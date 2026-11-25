"""Exercise the real async protocol with controlled transport completion.

Run directly with Python; this protocol has no torch, server or device dependency.
"""

import importlib.util
import unittest
from concurrent.futures import Future
from pathlib import Path
from threading import Event

_PATH = (
    Path(__file__).resolve().parents[4]
    / "python/sglang/srt/hardware_backend/npu/eplb/coordinator.py"
)
_SPEC = importlib.util.spec_from_file_location("npu_eplb_coordinator", _PATH)
protocol = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(protocol)


class Control:
    def __init__(self):
        self.peer_ready = True

    def vote(self, epoch, phase, layer, ready, error):
        return ready and self.peer_ready, error


class Runtime:
    def __init__(self, *, decode=False):
        self.decode = decode
        self.release = Event()
        self.release.set()
        self.installed = []
        self.staged = []
        self.closed = False
        self.synchronized = False
        self.fail_stage = False

    def prepare(self, layers):
        return layers

    def record_source_ready(self):
        return Future() if self.decode else None

    def stage(self, layer, submitted):
        if self.fail_stage:
            raise RuntimeError("injected transport failure")
        self.staged.append(layer)
        if submitted is not None:
            submitted.set_result(None)
        if not self.release.wait(timeout=2):
            raise TimeoutError("test transport was not released")
        return lambda: self.installed.append(layer)

    def install(self, layer, writeback):
        writeback()

    def synchronize(self):
        self.synchronized = True

    def close(self):
        self.closed = True


class TestNpuEplbCoordinator(unittest.TestCase):
    def make_coordinator(self, **kwargs):
        runtime, control = Runtime(**kwargs), Control()
        coordinator = protocol.AsyncEplbCoordinator(control, runtime, timeout=3)
        self.addCleanup(coordinator._executor.shutdown, wait=True)
        self.addCleanup(runtime.release.set)
        return coordinator, runtime, control

    def start_stage(self, coordinator, layers=(0,)):
        coordinator.start(lambda: layers)
        coordinator._future.result(timeout=1)
        self.assertFalse(coordinator.poll())

    def test_install_requires_all_ranks_and_next_boundary(self):
        coordinator, runtime, control = self.make_coordinator()
        self.start_stage(coordinator)
        coordinator._future.result(timeout=1)
        self.assertEqual(runtime.installed, [])
        control.peer_ready = False
        self.assertFalse(coordinator.poll())
        self.assertEqual(runtime.installed, [])
        control.peer_ready = True
        self.assertTrue(coordinator.poll())
        self.assertEqual(runtime.installed, [0])
        coordinator.close()

    def test_decode_waits_for_submission_but_not_transfer_completion(self):
        coordinator, runtime, _ = self.make_coordinator(decode=True)
        runtime.release.clear()
        self.start_stage(coordinator)
        # poll returned while the transfer is still held by the test transport.
        self.assertFalse(coordinator._future.done())
        self.assertEqual(runtime.installed, [])
        runtime.release.set()
        coordinator._future.result(timeout=1)
        self.assertTrue(coordinator.poll())
        coordinator.close()

    def test_layers_install_once_across_repeated_epochs(self):
        coordinator, runtime, _ = self.make_coordinator()
        for _ in range(2):
            self.start_stage(coordinator, (2, 5))
            coordinator._future.result(timeout=1)
            self.assertFalse(coordinator.poll())
            coordinator._future.result(timeout=1)
            self.assertTrue(coordinator.poll())
        self.assertEqual(runtime.installed, [2, 5, 2, 5])
        self.assertEqual(coordinator.epoch, 2)
        coordinator.close()

    def test_close_drains_staged_layer_without_installing_it(self):
        coordinator, runtime, _ = self.make_coordinator()
        self.start_stage(coordinator)
        coordinator._future.result(timeout=1)
        coordinator.close()
        self.assertTrue(runtime.synchronized)
        self.assertTrue(runtime.closed)
        self.assertEqual(runtime.installed, [])

    def test_planner_failure_drains_without_changing_live_layout(self):
        coordinator, runtime, _ = self.make_coordinator()

        def fail():
            raise ValueError("injected planner failure")

        coordinator.start(fail)
        with self.assertRaises(ValueError):
            coordinator._future.result(timeout=1)
        with self.assertRaises(protocol.RebalanceAborted):
            coordinator.poll()
        self.assertEqual(coordinator.phase, protocol.Phase.DISABLED)
        self.assertEqual(runtime.installed, [])
        coordinator.close()

    def test_transfer_failure_is_fatal_before_install(self):
        coordinator, runtime, _ = self.make_coordinator()
        runtime.fail_stage = True
        self.start_stage(coordinator)
        with self.assertRaises(RuntimeError):
            coordinator._future.result(timeout=1)
        with self.assertRaises(protocol.UnsafeRebalanceFailure):
            coordinator.poll()
        self.assertEqual(coordinator.phase, protocol.Phase.FAILED)
        self.assertFalse(runtime.closed)
        self.assertEqual(runtime.installed, [])
        protocol._FAILED_OWNERS.remove(coordinator)


if __name__ == "__main__":
    unittest.main()
