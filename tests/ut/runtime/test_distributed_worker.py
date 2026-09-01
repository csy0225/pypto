# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for ``DistributedWorker`` (the ``prepare()`` reuse handle).

Runs without a device or the ``simpler`` package by patching the module-level
setup helpers in :mod:`pypto.runtime.distributed_runner`, so construction does
no real compile/fork. The tests cover both ordinary prepared dispatch and the
persistent contract: bounded asynchronous submission, retained per-program
domains, handle-owned input lifetimes, and complete cleanup before publication.
"""

import ctypes
import gc
import importlib.util
import json
import sys
import threading
import weakref
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch
from pypto.ir.compiled_program import _ParamInfo
from pypto.ir.distributed_compiled_program import DistributedConfig
from pypto.pypto_core import DataType
from pypto.pypto_core.ir import ParamDirection
from pypto.runtime import DeviceTensor, StackedDeviceTensor
from pypto.runtime.bench import (
    _L3_SWIMLANE_GRAPH_BEGIN,
    _L3_SWIMLANE_GRAPH_END,
    _L3_SWIMLANE_TIMING_BEGIN,
    _L3_SWIMLANE_TIMING_END,
)
from pypto.runtime.distributed_runner import (
    DistributedRunHandle,
    DistributedWorker,
    _assemble_chip_callables,
    _clear_dfx_dispatch_dirs,
    _collect_l3_swimlane,
    _construct_worker,
    _make_call_config,
    _prepare_reused_dep_gen_dirs,
    _reset_dfx_dispatch_state,
    _submit_chip,
)
from pypto.runtime.runner import _CHIP_SWIMLANE_RECORDS_NAME, RunConfig


def _param(name: str, shape: list[int], direction: ParamDirection = ParamDirection.In) -> _ParamInfo:
    return _ParamInfo(name=name, direction=direction, shape=shape, dtype=DataType.FP32)


def _fake_compiled(param_infos, output_indices):
    """A minimal stand-in for DistributedCompiledProgram used by DistributedWorker."""
    compiled = MagicMock(name="DistributedCompiledProgram")
    compiled._get_metadata.return_value = (param_infos, output_indices, [])
    compiled._distributed_config = DistributedConfig()
    compiled.platform = "a2a3sim"
    return compiled


class _ImmediateNativeHandle:
    """Minimal Simpler RunHandle stand-in used by prepared-worker tests."""

    done = True

    def result(self, timeout=None):
        del timeout


class _ControlledNativeHandle:
    """RunHandle stand-in with an explicit terminal-completion gate."""

    def __init__(
        self,
        error: BaseException | None = None,
        on_result: Callable[[], None] | None = None,
    ) -> None:
        self._terminal = threading.Event()
        self.result_started = threading.Event()
        self.error = error
        self._on_result = on_result

    @property
    def done(self) -> bool:
        return self._terminal.is_set()

    def complete(self, error: BaseException | None = None) -> None:
        self.error = error
        self._terminal.set()

    def result(self, timeout=None):
        self.result_started.set()
        if not self._terminal.wait(timeout):
            raise TimeoutError("native handle timed out")
        if self._on_result is not None:
            self._on_result()
        if self.error is not None:
            raise self.error


class _ObservedLock:
    """A regular mutex that exposes when a second thread blocks on admission."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.blocked = threading.Event()

    def __enter__(self) -> "_ObservedLock":
        if self._lock.locked():
            self.blocked.set()
        self._lock.acquire()
        return self

    def __exit__(self, *_exc: Any) -> None:
        self._lock.release()


class _NamedHostRange:
    """What ``wrap_fork_inherited`` returns: a host range named in place, never copied."""

    def __init__(self, data_ptr, nbytes, owner, buffer_id, *, access=None, backend_kind=None) -> None:
        self.base = data_ptr
        self.nbytes = nbytes
        self.owner = owner
        self.buffer_id = buffer_id
        self.access = access
        self.backend_kind = backend_kind


class _FakeBuffer:
    def __init__(self, base: int, nbytes: int, *, host: bool = False, owner_worker_id: int = 0) -> None:
        self.nbytes = nbytes
        self.owner_worker_id = owner_worker_id
        self._backing = (ctypes.c_ubyte * nbytes)() if host else None
        self.base = ctypes.addressof(self._backing) if self._backing is not None else base
        self.closed = False
        self.identity = SimpleNamespace(
            owner_instance_id=b"owner-id",
            buffer_id=max(1, int(self.base)),
            generation=1,
        )
        self.address_space = 1
        self.access = 2
        self.backend_kind = 4
        self.owner_worker_path_id = owner_worker_id
        self.body = int(self.base).to_bytes(8, "little")

    def tensor(self, shapes, dtype):
        return shapes, dtype


def _alloc_by_worker(worker_mock, buffers):
    """Make the fake allocator return a buffer chosen by ``worker_id``, not by call order.

    A ``side_effect`` list is consumed in call order, which couples the fixture to the order
    the shards happen to be uploaded in. Since `alloc_stacked_tensor` uploads the shards
    concurrently, that order is not defined — and the real
    ``alloc_child_tensor(worker_id, ...)`` returns the buffer belonging to the worker it is
    asked about, so keying on the worker is also the more faithful fake.
    """
    worker_mock.alloc_child_tensor.side_effect = lambda worker_id, *args, **kwargs: buffers[worker_id]
    return buffers


@pytest.fixture
def patched_setup():
    """Patch every setup helper so DistributedWorker() does no real work.

    Yields a dict of the mocks so individual tests can assert call counts.
    The worker mock records malloc/copy_to/free for alloc_tensor checks.
    """
    simpler = ModuleType("simpler")
    simpler.__path__ = []
    task_interface = ModuleType("simpler.task_interface")
    setattr(task_interface, "DataType", SimpleNamespace(UINT8=object()))
    setattr(simpler, "task_interface", task_interface)
    # `simpler.buffer` is the zero-copy naming path: a test can tell a named range from a
    # staged copy by which of these two the worker was handed.
    buffer_mod = ModuleType("simpler.buffer")
    setattr(buffer_mod, "AccessMode", SimpleNamespace(READ="READ", READWRITE="READWRITE"))
    setattr(buffer_mod, "BackendKind", SimpleNamespace(FORK_SHM="FORK_SHM", FORK_COW="FORK_COW"))
    setattr(buffer_mod, "mint_owner_instance_id", lambda: b"owner-id")
    setattr(
        buffer_mod,
        "wrap_fork_inherited",
        lambda data_ptr, nbytes, owner, buffer_id, **kwargs: _NamedHostRange(
            data_ptr, nbytes, owner, buffer_id, **kwargs
        ),
    )
    setattr(simpler, "buffer", buffer_mod)

    worker = MagicMock(name="Worker(level=3)")
    worker.chip_contexts = []
    worker._live_domains = {}
    worker._building_run_resources = None
    worker.alloc_child_tensor.return_value = _FakeBuffer(0xDEAD0000, 1 << 20)
    worker.device_memset_available = False
    worker.create_buffer.side_effect = lambda nbytes: _FakeBuffer(0, nbytes, host=True)
    worker.submit.side_effect = lambda fn: (fn(worker._orch, None, None), _ImmediateNativeHandle())[1]

    mod = "pypto.runtime.distributed_runner"
    chip_callables = ({"chip_orch": object()}, "rt_name", False)
    with (
        patch.dict(
            sys.modules,
            {
                "simpler": simpler,
                "simpler.task_interface": task_interface,
                "simpler.buffer": buffer_mod,
            },
        ),
        patch(f"{mod}._assemble_chip_callables", return_value=chip_callables) as assemble,
        patch(f"{mod}._load_orch_entry", return_value=(MagicMock(name="entry_fn"), None)) as load_entry,
        patch(f"{mod}._load_sub_worker_fns", return_value={}) as load_subs,
        patch(f"{mod}._load_required_callbacks", return_value=set()) as load_required,
        patch(f"{mod}._construct_worker", return_value=worker) as construct,
        patch(f"{mod}._register_callables", return_value=({}, {"chip_orch": 0})) as register,
        patch(f"{mod}._make_call_config", return_value=MagicMock(name="CallConfig")) as make_call_config,
        patch(f"{mod}._dispatch") as dispatch,
        patch(f"{mod}._submit_dispatch", return_value=_ImmediateNativeHandle()) as submit_dispatch,
    ):
        yield {
            "worker": worker,
            "assemble": assemble,
            "load_entry": load_entry,
            "load_subs": load_subs,
            "load_required": load_required,
            "construct": construct,
            "register": register,
            "make_call_config": make_call_config,
            "dispatch": dispatch,
            "submit_dispatch": submit_dispatch,
        }


def _resident(
    rt: DistributedWorker,
    shape: tuple[int, ...],
    *,
    worker_id: int = 0,
) -> DeviceTensor:
    """Allocate a uniquely addressed resident tensor through the public API."""
    nbytes = torch.empty((), dtype=torch.float32).element_size()
    for dim in shape:
        nbytes *= dim
    base = 0x10000000 + (len(rt._device_buffers) + 1) * 0x100000
    rt._w.alloc_child_tensor.return_value = _FakeBuffer(base, nbytes)
    return rt.alloc_tensor(shape, torch.float32, worker_id=worker_id)


class TestSetupOnce:
    def test_setup_runs_once_dispatch_many(self, patched_setup):
        m = patched_setup
        compiled = _fake_compiled([_param("a", [128, 128]), _param("b", [128, 128])], [])

        rt = DistributedWorker(compiled)
        # All expensive setup happened exactly once at construction.
        m["assemble"].assert_called_once()
        m["construct"].assert_called_once()
        m["register"].assert_called_once()
        m["worker"].init.assert_called_once()
        assert m["worker"].__dict__["_pypto_tensor_owner_ref"]() is rt
        # Simpler's public init owns eager hierarchy startup.
        m["worker"]._start_hierarchical.assert_not_called()

        a = _resident(rt, (128, 128))
        b = _resident(rt, (128, 128))
        rt(a, b)
        rt(a, b)
        rt(a, b)

        # Setup still once; dispatch ran per call.
        assert m["submit_dispatch"].call_count == 3
        m["assemble"].assert_called_once()
        m["construct"].assert_called_once()
        assert m["worker"].init.call_count == 1
        rt.close()

    def test_prepared_descriptor_key_tracks_payload_and_tensor_metadata(self, patched_setup):
        compiled = _fake_compiled([_param("a", [4, 4])], [])
        rt = DistributedWorker(compiled)
        host = torch.zeros(4, 4, dtype=torch.float32).share_memory_()

        before = rt._prepared_task_args_descriptor_key((host,))
        host.add_(1)
        assert rt._prepared_task_args_descriptor_key((host,)) == before
        host.transpose_(0, 1)
        assert rt._prepared_task_args_descriptor_key((host,)) != before
        rt.close()

    def test_prepared_descriptor_key_tracks_device_and_stacked_lifecycle(self, patched_setup):
        compiled = _fake_compiled([_param("a", [4])], [])
        rt = DistributedWorker(compiled)
        first = _resident(rt, (4,), worker_id=0)
        second = _resident(rt, (4,), worker_id=1)
        stacked = StackedDeviceTensor((first, second), (2, 4), (0, 1))

        before = rt._prepared_task_args_descriptor_key((first, stacked))
        assert before is not None
        first.buffer.identity.generation += 1
        assert rt._prepared_task_args_descriptor_key((first, stacked)) != before
        first.buffer.closed = True
        assert rt._prepared_task_args_descriptor_key((first, stacked)) is None
        first.buffer.closed = False
        rt.close()

    def test_prepared_descriptor_key_fails_open_for_mutable_and_unknown_values(self, patched_setup):
        compiled = _fake_compiled([_param("a", [4])], [])
        rt = DistributedWorker(compiled)
        cyclic: list[object] = []
        cyclic.append(cyclic)

        assert rt._prepared_task_args_descriptor_key(([1],)) is None
        assert rt._prepared_task_args_descriptor_key(((1, 2),)) is None
        assert rt._prepared_task_args_descriptor_key((cyclic,)) is None
        assert rt._prepared_task_args_descriptor_key((object(),)) is None
        rt.close()

    def test_prepared_descriptor_key_fails_open_on_broken_buffer(self, patched_setup):
        class BrokenBuffer:
            closed = False

            @property
            def identity(self):
                raise RuntimeError("descriptor unavailable")

        assert DistributedWorker._prepared_buffer_descriptor_key(BrokenBuffer()) is None

    def test_prepared_descriptor_key_fails_open_for_raw_simpler_tensor(self, patched_setup):
        compiled = _fake_compiled([_param("a", [4])], [])
        rt = DistributedWorker(compiled)
        raw = object()

        with patch("pypto.runtime.distributed_runner._is_simpler_tensor", return_value=True):
            assert rt._prepared_task_args_descriptor_key((raw,)) is None
        rt.close()

    @pytest.mark.parametrize(
        "case",
        [
            "torch_metadata",
            "device_generation",
            "device_closed",
            "stacked_placement",
            "scalar_value",
        ],
    )
    def test_prepared_descriptor_key_is_not_weaker_than_full_signature(
        self,
        patched_setup,
        case,
    ):
        from pypto.runtime import device_tensor, tensor_arg

        rt = DistributedWorker(_fake_compiled([_param("a", [4])], []))
        if case == "torch_metadata":
            value = torch.zeros(4, 4, dtype=torch.float32).share_memory_()
            values = [value]

            def mutate():
                value.transpose_(0, 1)

        elif case in {"device_generation", "device_closed"}:
            value = _resident(rt, (4,))
            values = [value]

            def mutate():
                if case == "device_generation":
                    value.buffer.identity.generation += 1
                else:
                    value.buffer.closed = True

        elif case == "stacked_placement":
            first = _resident(rt, (4,), worker_id=0)
            second = _resident(rt, (4,), worker_id=1)
            values = [StackedDeviceTensor((first, second), (2, 4), (0, 1))]

            def mutate():
                values[0] = StackedDeviceTensor((first, second), (2, 4), (1, 0))

        else:
            values = [7]

            def mutate():
                values[0] = 8

        wire_tensor = type("WireTensor", (), {})
        modules = (SimpleNamespace(Tensor=wire_tensor), device_tensor, None)
        with patch.object(tensor_arg, "_modules", return_value=modules):
            before_value = values[0]
            before_signature = tensor_arg._task_args_signature((before_value,))
            before_key = rt._prepared_task_args_descriptor_key((before_value,))
            mutate()
            after_value = values[0]
            after_signature = tensor_arg._task_args_signature((after_value,))
            after_key = rt._prepared_task_args_descriptor_key((after_value,))

        assert before_signature is not None
        assert after_signature is not None
        assert before_signature != after_signature
        assert before_key is None or after_key is None or before_key != after_key
        rt.close()

    def test_unpublished_frame_release_preserves_signature_generation(self, patched_setup):
        compiled = _fake_compiled([_param("a", [4])], [])
        rt = DistributedWorker(compiled)
        frame = rt._dispatch_frames[0]
        frame.in_use = True
        frame.task_args_descriptor_key = ("old",)
        frame.task_args_signature_generation = 7
        frame.task_args_signature_token = ("token", 7)

        rt._release_unpublished_dispatch_frame(frame)

        assert frame.task_args_descriptor_key is None
        assert frame.task_args_signature_generation == 7
        assert frame.task_args_signature_token is None
        rt.close()

    def test_failed_submit_retry_advances_signature_token(self, patched_setup):
        compiled = _fake_compiled([_param("a", [4])], [])
        rt = DistributedWorker(compiled)
        arg = _resident(rt, (4,))
        submit_prepared = rt._submit_prepared_native
        signature_tokens: list[Any] = []

        def flaky_submit(state, tensors, call_config, keepalive, cleanup, token):
            signature_tokens.append(token)
            if len(signature_tokens) == 1:
                raise RuntimeError("injected pre-submit failure")
            return submit_prepared(
                state,
                tensors,
                call_config,
                keepalive,
                cleanup,
                token,
            )

        with patch.object(rt, "_submit_prepared_native", side_effect=flaky_submit):
            with pytest.raises(RuntimeError, match="injected pre-submit failure"):
                rt.submit(compiled, arg)
            handle = rt.submit(compiled, arg)
            handle.result()

        assert signature_tokens[0] is not None
        assert signature_tokens[1] is not None
        assert signature_tokens[1] != signature_tokens[0]
        assert rt._dispatch_frames[0].task_args_signature_generation == 2
        rt.close()


class TestAsyncDispatchHandle:
    def test_submit_returns_handle_and_retires_frame_on_result(self, patched_setup):
        m = patched_setup
        compiled = _fake_compiled([_param("a", [16, 16])], [])
        native = _ControlledNativeHandle()
        m["submit_dispatch"].return_value = native
        rt = DistributedWorker(compiled)

        handle = rt.submit(compiled, _resident(rt, (16, 16)))

        assert isinstance(handle, DistributedRunHandle)
        assert handle.done is False
        assert len(rt._active_dispatch_handles) == 1
        native.complete()
        handle.result()
        assert handle.done is True
        assert not rt._active_dispatch_handles
        assert all(not frame.in_use for frame in rt._dispatch_frames)
        rt.close()

    def test_timeout_keeps_frame_owned_until_later_completion(self, patched_setup):
        m = patched_setup
        compiled = _fake_compiled([_param("a", [16, 16])], [])
        native = _ControlledNativeHandle()
        m["submit_dispatch"].return_value = native
        rt = DistributedWorker(compiled)
        handle = rt.submit(compiled, _resident(rt, (16, 16)))

        with pytest.raises(TimeoutError):
            handle.result(timeout=0.0)
        assert handle.done is False
        assert any(frame.in_use for frame in rt._dispatch_frames)
        with pytest.raises(ValueError, match="non-negative finite"):
            handle.result(timeout=-1.0)

        native.complete()
        handle.result()
        assert all(not frame.in_use for frame in rt._dispatch_frames)
        rt.close()

    def test_handle_keeps_input_alive_until_completion(self, patched_setup):
        m = patched_setup
        compiled = _fake_compiled([_param("a", [16, 16])], [])
        native = _ControlledNativeHandle()
        m["submit_dispatch"].return_value = native
        rt = DistributedWorker(compiled)
        arg = torch.zeros((16, 16), dtype=torch.float32).share_memory_()
        arg_ref = weakref.ref(arg)

        handle = rt.submit(compiled, arg)
        m["submit_dispatch"].reset_mock()
        del arg
        gc.collect()
        assert arg_ref() is not None

        native.complete()
        handle.result()
        gc.collect()
        assert arg_ref() is None
        rt.close()

    def test_third_submit_drains_oldest_before_allocating_metadata(self, patched_setup):
        m = patched_setup
        compiled = _fake_compiled([_param("a", [16, 16])], [])
        natives = [_ControlledNativeHandle() for _ in range(3)]
        m["submit_dispatch"].side_effect = natives
        rt = DistributedWorker(compiled)
        arg = _resident(rt, (16, 16))
        first = rt.submit(compiled, arg)
        second = rt.submit(compiled, arg)
        assert m["make_call_config"].call_count == 3  # prepare + two accepted dispatches

        third_result: list[DistributedRunHandle] = []
        third_error: list[BaseException] = []

        def submit_third() -> None:
            try:
                third_result.append(rt.submit(compiled, arg))
            except BaseException as exc:  # noqa: BLE001 - asserted below
                third_error.append(exc)

        caller = threading.Thread(target=submit_third)
        caller.start()
        assert natives[0].result_started.wait(timeout=2)
        # Backpressure happens before a CallConfig or frame-local tensor map is
        # created for the third dispatch.
        assert m["make_call_config"].call_count == 3
        assert not third_result

        natives[0].complete()
        caller.join(timeout=2)
        assert not caller.is_alive()
        assert not third_error
        assert len(third_result) == 1
        assert first.done is True
        assert m["make_call_config"].call_count == 4

        natives[1].complete()
        natives[2].complete()
        second.result()
        third_result[0].result()
        rt.close()

    def test_backpressure_does_not_rethrow_an_older_handle_error(self, patched_setup):
        m = patched_setup
        compiled = _fake_compiled([_param("a", [16, 16])], [])
        natives = [_ControlledNativeHandle() for _ in range(3)]
        m["submit_dispatch"].side_effect = natives
        rt = DistributedWorker(compiled)
        arg = _resident(rt, (16, 16))
        first = rt.submit(compiled, arg)
        second = rt.submit(compiled, arg)

        natives[0].complete(RuntimeError("first dispatch failed"))
        third = rt.submit(compiled, arg)

        with pytest.raises(RuntimeError, match="first dispatch failed"):
            first.result()
        natives[1].complete()
        natives[2].complete()
        second.result()
        third.result()
        rt.close()

    def test_in_flight_dispatches_use_distinct_host_scratch(self, patched_setup):
        m = patched_setup
        allocated: list[object] = []

        def alloc_intermediates(tensors):
            scratch = object()
            allocated.append(scratch)
            tensors["scratch"] = scratch

        m["load_entry"].return_value = (MagicMock(name="entry_fn"), alloc_intermediates)
        compiled = _fake_compiled([_param("a", [16, 16])], [])
        natives = [_ControlledNativeHandle(), _ControlledNativeHandle()]
        m["submit_dispatch"].side_effect = natives
        rt = DistributedWorker(compiled)
        arg = _resident(rt, (16, 16))

        first = rt.submit(compiled, arg)
        second = rt.submit(compiled, arg)

        assert len(allocated) == 2
        first_tensors = m["submit_dispatch"].call_args_list[0].args[2]
        second_tensors = m["submit_dispatch"].call_args_list[1].args[2]
        assert first_tensors["scratch"] is allocated[0]
        assert second_tensors["scratch"] is allocated[1]
        natives[0].complete()
        natives[1].complete()
        first.result()
        second.result()
        rt.close()

    def test_dfx_submit_waits_for_earlier_dispatch(self, patched_setup, tmp_path):
        m = patched_setup
        compiled = _fake_compiled([_param("a", [16, 16])], [])
        compiled.output_dir = tmp_path
        natives = [_ControlledNativeHandle(), _ImmediateNativeHandle()]
        m["submit_dispatch"].side_effect = natives
        rt = DistributedWorker(compiled)
        arg = _resident(rt, (16, 16))
        first = rt.submit(compiled, arg)
        submitted: list[DistributedRunHandle] = []

        with patch("pypto.runtime.distributed_runner._clear_dfx_dispatch_dirs") as clear:
            caller = threading.Thread(
                target=lambda: submitted.append(
                    rt.submit(compiled, arg, config=RunConfig(platform="a2a3sim", enable_dep_gen=True))
                )
            )
            caller.start()
            assert natives[0].result_started.wait(timeout=2)
            clear.assert_not_called()
            assert m["submit_dispatch"].call_count == 1

            natives[0].complete()
            caller.join(timeout=2)
            assert not caller.is_alive()
            clear.assert_called_once_with(tmp_path / "dfx_outputs")

        first.result()
        submitted[0].result()
        rt.close()

    def test_failed_handle_recycles_frame_and_caches_error(self, patched_setup):
        m = patched_setup
        compiled = _fake_compiled([_param("a", [16, 16])], [])
        failed = _ControlledNativeHandle()
        m["submit_dispatch"].return_value = failed
        rt = DistributedWorker(compiled)
        handle = rt.submit(compiled, _resident(rt, (16, 16)))
        failed.complete(RuntimeError("dispatch failed"))

        with pytest.raises(RuntimeError, match="dispatch failed"):
            handle.result()
        with pytest.raises(RuntimeError, match="dispatch failed"):
            handle.result()
        assert all(not frame.in_use for frame in rt._dispatch_frames)
        rt.close()

    def test_close_drains_outstanding_handle_before_worker_close(self, patched_setup):
        m = patched_setup
        compiled = _fake_compiled([_param("a", [16, 16])], [])
        native = _ControlledNativeHandle()
        m["submit_dispatch"].return_value = native
        rt = DistributedWorker(compiled)
        arg = _resident(rt, (16, 16))
        rt.submit(compiled, arg)

        closed = threading.Event()
        closer = threading.Thread(target=lambda: (rt.close(), closed.set()))
        closer.start()
        assert native.result_started.wait(timeout=2)
        assert not closed.is_set()
        m["worker"].close.assert_not_called()

        native.complete()
        closer.join(timeout=2)
        assert closed.is_set()
        m["worker"].close.assert_called_once_with()
        with pytest.raises(RuntimeError, match="after close"):
            rt.submit(compiled, DeviceTensor(0x1000, (16, 16), torch.float32))

    def test_interrupted_native_handle_publication_keeps_frame_until_close(self, patched_setup):
        m = patched_setup
        compiled = _fake_compiled([_param("a", [16, 16])], [])
        native = _ControlledNativeHandle()
        m["worker"]._accepted_run_handles = set()
        m["worker"]._hierarchical_start_cv = threading.Condition()

        def interrupt_after_acceptance(*_args):
            with m["worker"]._hierarchical_start_cv:
                m["worker"]._accepted_run_handles.add(native)
            raise KeyboardInterrupt("interrupted after native acceptance")

        m["submit_dispatch"].side_effect = interrupt_after_acceptance
        rt = DistributedWorker(compiled)
        arg = _resident(rt, (16, 16))

        with pytest.raises(KeyboardInterrupt, match="after native acceptance"):
            rt.submit(compiled, arg)

        assert len(rt._active_dispatch_handles) == 1
        assert sum(frame.in_use for frame in rt._dispatch_frames) == 1
        closer = threading.Thread(target=rt.close)
        closer.start()
        assert native.result_started.wait(timeout=2)
        assert closer.is_alive()

        native.complete()
        closer.join(timeout=2)
        assert not closer.is_alive()
        assert not rt._active_dispatch_handles
        assert all(not frame.in_use for frame in rt._dispatch_frames)
        m["worker"].close.assert_called_once_with()


class TestPerTaskRingSizing:
    """A per-dispatch ``RunConfig`` sizes that dispatch's runtime ring buffers.

    ``_make_call_config`` runs once at construction to build the program's
    prewarm baseline. Every accepted asynchronous dispatch receives a fresh
    snapshot; a ``RunConfig`` adds that dispatch's overrides.
    """

    # ``_submit_dispatch(w, entry_fn, tensors, chip_cids, sub_ids, call_config, ...)``
    _CALL_CONFIG_ARG = 5

    def test_no_config_snapshots_prepared_baseline(self, patched_setup):
        m = patched_setup
        compiled = _fake_compiled([_param("a", [16, 16])], [])
        rt = DistributedWorker(compiled)
        # Construction builds the baseline exactly once.
        assert m["make_call_config"].call_count == 1
        baseline = m["make_call_config"].return_value
        fresh = MagicMock(name="FreshCallConfig")
        m["make_call_config"].return_value = fresh

        rt(_resident(rt, (16, 16)))

        assert m["make_call_config"].call_count == 2
        assert fresh is not baseline
        assert m["submit_dispatch"].call_args.args[self._CALL_CONFIG_ARG] is fresh
        rt.close()

    def test_per_dispatch_config_rebuilds_call_config(self, patched_setup):
        from pypto.runtime import RunConfig  # noqa: PLC0415

        m = patched_setup
        compiled = _fake_compiled([_param("a", [16, 16])], [])
        rt = DistributedWorker(compiled)
        assert m["make_call_config"].call_count == 1  # baseline at construction

        rc = RunConfig(platform="a2a3sim", ring_task_window=64, ring_heap=4 * 1024 * 1024)
        rt(_resident(rt, (16, 16)), config=rc)

        # A per-dispatch config rebuilds from (program DistributedConfig, rc).
        assert m["make_call_config"].call_count == 2
        rebuild = m["make_call_config"].call_args
        assert rebuild.args[0] is compiled._distributed_config
        assert rebuild.args[1] is rc
        # The freshly built config (not None) is what reaches _dispatch.
        assert (
            m["submit_dispatch"].call_args.args[self._CALL_CONFIG_ARG] is m["make_call_config"].return_value
        )
        rt.close()

    def test_run_method_forwards_per_dispatch_config(self, patched_setup):
        from pypto.runtime import RunConfig  # noqa: PLC0415

        m = patched_setup
        compiled = _fake_compiled([_param("a", [16, 16])], [])
        rt = DistributedWorker(compiled)

        rc = RunConfig(platform="a2a3sim", ring_dep_pool=256)
        rt.run(compiled, _resident(rt, (16, 16)), config=rc)

        # rt.run(...) honors the same per-dispatch ring sizing as rt(...).
        assert m["make_call_config"].call_count == 2
        assert m["make_call_config"].call_args.args[1] is rc
        rt.close()

    def test_prepared_swimlane_reuses_deps_without_co_enabling_dep_gen(
        self,
        patched_setup,
        tmp_path,
    ):
        m = patched_setup
        compiled = _fake_compiled([_param("a", [16, 16])], [])
        compiled.platform = "a2a3"
        compiled.output_dir = tmp_path
        dep_dir = tmp_path / "dfx_outputs" / "rank0" / "d0"
        dep_dir.mkdir(parents=True)
        (dep_dir / "deps.json").write_text("{}", encoding="utf-8")
        (dep_dir / _CHIP_SWIMLANE_RECORDS_NAME).write_text("stale", encoding="utf-8")
        rt = DistributedWorker(compiled)

        rc = RunConfig(
            platform="a2a3",
            enable_l2_swimlane=True,
            l2_swimlane_reuse_dep_gen=True,
        )
        with (
            patch(
                "pypto.runtime.distributed_runner._clear_dfx_dispatch_dirs",
            ) as clear,
            patch(
                "pypto.runtime.distributed_runner._collect_l3_swimlane",
            ),
        ):
            rt(_resident(rt, (16, 16)), config=rc)

        clear.assert_not_called()
        rebuild = m["make_call_config"].call_args
        assert rebuild.kwargs["co_enable_swimlane_dep_gen"] is False
        assert (dep_dir / "deps.json").is_file()
        assert not (dep_dir / _CHIP_SWIMLANE_RECORDS_NAME).exists()
        rt.close()


class TestPreparedSwimlaneTwoPass:
    """Prepared onboard L3 captures deps, then measures without dep_gen."""

    _CALL_CONFIG_ARG = 5

    def test_onboard_reuses_worker_for_graph_then_clean_timing(self, patched_setup, tmp_path, capsys):
        m = patched_setup
        compiled = _fake_compiled([_param("a", [16, 16])], [])
        compiled.platform = "a2a3"
        compiled.output_dir = tmp_path
        rt = DistributedWorker(compiled)

        # Ignore the baseline config constructed during prepare(); this test
        # observes the two configs built for the caller-visible dispatch.
        m["make_call_config"].reset_mock()
        deps_call_config = MagicMock(name="DepsCallConfig")
        timing_call_config = MagicMock(name="TimingCallConfig")
        m["make_call_config"].side_effect = [deps_call_config, timing_call_config]
        events: list[str] = []
        m["submit_dispatch"].side_effect = lambda *args: (
            events.append("deps" if args[self._CALL_CONFIG_ARG] is deps_call_config else "timing"),
            _ImmediateNativeHandle(),
        )[1]

        run_config = RunConfig(
            platform="a2a3",
            enable_chip_swimlane=1,  # AICore-timing level, not just "on"
            enable_pmu=3,
            enable_scope_stats=True,
            enable_dump_args=2,
        )
        with (
            patch(
                "pypto.runtime.distributed_runner._clear_dfx_dispatch_dirs",
                side_effect=lambda _path: events.append("clear"),
            ) as clear,
            patch(
                "pypto.runtime.distributed_runner._collect_l3_swimlane",
                side_effect=lambda _output, _platform: events.append("collect"),
            ) as collect,
        ):
            rt(_resident(rt, (16, 16)), config=run_config)

        assert events == ["clear", "deps", "timing", "collect"]
        assert m["submit_dispatch"].call_count == 2
        assert all(call.args[0] is m["worker"] for call in m["submit_dispatch"].call_args_list)
        assert m["construct"].call_count == 1
        assert m["worker"].init.call_count == 1
        clear.assert_called_once_with(tmp_path / "dfx_outputs")
        collect.assert_called_once_with(tmp_path, "a2a3")

        assert m["make_call_config"].call_count == 2
        deps_build, timing_build = m["make_call_config"].call_args_list
        deps_config = deps_build.args[1]
        assert deps_config.enable_chip_swimlane == 0
        assert deps_config.enable_dep_gen is True
        assert deps_config.enable_pmu == 0
        assert deps_config.enable_scope_stats is False
        assert deps_config.enable_dump_args == 0

        timing_config = timing_build.args[1]
        assert timing_config.enable_chip_swimlane == 1
        assert timing_config.enable_dep_gen is False
        assert timing_config.enable_pmu == 3
        assert timing_config.enable_scope_stats is True
        assert timing_config.enable_dump_args == 2
        assert timing_build.kwargs["co_enable_swimlane_dep_gen"] is False
        captured = capsys.readouterr()
        assert [line for line in captured.err.splitlines() if "l3_swimlane_pass=" in line] == [
            _L3_SWIMLANE_GRAPH_BEGIN,
            _L3_SWIMLANE_GRAPH_END,
            _L3_SWIMLANE_TIMING_BEGIN,
            _L3_SWIMLANE_TIMING_END,
        ]
        rt.close()

    def test_simulator_keeps_single_pass(self, patched_setup, tmp_path):
        m = patched_setup
        compiled = _fake_compiled([_param("a", [16, 16])], [])
        compiled.output_dir = tmp_path
        rt = DistributedWorker(compiled)

        m["make_call_config"].reset_mock()
        call_config = MagicMock(name="SimCallConfig")
        m["make_call_config"].return_value = call_config
        run_config = RunConfig(
            platform="a2a3sim",
            enable_chip_swimlane=1,  # AICore-timing level, not just "on"
        )
        with patch("pypto.runtime.distributed_runner._collect_l3_swimlane") as collect:
            rt(_resident(rt, (16, 16)), config=run_config)

        m["submit_dispatch"].assert_called_once()
        assert m["submit_dispatch"].call_args.args[self._CALL_CONFIG_ARG] is call_config
        m["make_call_config"].assert_called_once_with(
            compiled._distributed_config,
            run_config,
            dfx_base=tmp_path / "dfx_outputs",
        )
        collect.assert_called_once_with(tmp_path, "a2a3sim")
        rt.close()

    def test_dep_gen_without_swimlane_keeps_single_pass(self, patched_setup, tmp_path):
        m = patched_setup
        compiled = _fake_compiled([_param("a", [16, 16])], [])
        compiled.platform = "a2a3"
        compiled.output_dir = tmp_path
        rt = DistributedWorker(compiled)

        m["make_call_config"].reset_mock()
        run_config = RunConfig(platform="a2a3", enable_dep_gen=True)
        with patch("pypto.runtime.distributed_runner._collect_l3_swimlane") as collect:
            rt(_resident(rt, (16, 16)), config=run_config)

        m["submit_dispatch"].assert_called_once()
        m["make_call_config"].assert_called_once_with(
            compiled._distributed_config,
            run_config,
            dfx_base=tmp_path / "dfx_outputs",
        )
        collect.assert_not_called()
        rt.close()

    def test_persistent_route_waits_for_graph_then_timing_requests(self, patched_setup, tmp_path):
        m = patched_setup
        compiled = _fake_compiled([_param("a", [16, 16])], [])
        compiled.platform = "a2a3"
        compiled.output_dir = tmp_path
        rt = DistributedWorker(compiled)

        # Exercise _dispatch_prepared's persistent branch without starting a
        # background thread: the two-pass fallback still fences both requests.
        rt._persistent = True
        m["make_call_config"].reset_mock()
        deps_call_config = MagicMock(name="DepsCallConfig")
        timing_call_config = MagicMock(name="TimingCallConfig")
        m["make_call_config"].side_effect = [deps_call_config, timing_call_config]
        with (
            patch.object(
                rt,
                "_submit_persistent",
                return_value=_ImmediateNativeHandle(),
            ) as submit_persistent,
            patch("pypto.runtime.distributed_runner._collect_l3_swimlane"),
        ):
            rt(
                _resident(rt, (16, 16)),
                config=RunConfig(platform="a2a3", enable_chip_swimlane=True),
            )

        assert [call.args[2] for call in submit_persistent.call_args_list] == [
            deps_call_config,
            timing_call_config,
        ]
        rt.close()


class TestArenaPrewarm:
    """``init`` prewarms the prebuilt runtime-arena cache with the ring sizing the
    first dispatch will use, so the ~800ms cold build lands at prepare() time
    rather than inside the first (usually timed) dispatch.
    """

    def test_prewarms_with_prepared_baseline_when_no_config(self, patched_setup):
        m = patched_setup
        compiled = _fake_compiled([_param("a", [16, 16])], [])
        rt = DistributedWorker(compiled)

        # No worker RunConfig → the program's baseline CallConfig (the same one
        # config-less dispatches reuse) is what init prewarms with; no rebuild.
        assert m["make_call_config"].call_count == 1
        assert m["worker"].init.call_args.kwargs["prewarm_config"] is m["make_call_config"].return_value
        rt.close()

    def test_prewarms_with_worker_config_ring_sizing(self, patched_setup):
        from pypto.runtime import RunConfig  # noqa: PLC0415

        m = patched_setup
        compiled = _fake_compiled([_param("a", [16, 16])], [])
        rc = RunConfig(platform="a2a3sim", ring_heap=4 * 1024 * 1024)

        rt = DistributedWorker(compiled, rc)

        # A worker RunConfig builds a second CallConfig from (program
        # DistributedConfig, rc) — the same construction a dispatch with rc uses,
        # so the prewarmed arena's sizing key matches that dispatch's.
        assert m["make_call_config"].call_count == 2
        prewarm_build = m["make_call_config"].call_args
        assert prewarm_build.args[0] is compiled._distributed_config
        assert prewarm_build.args[1] is rc
        assert m["worker"].init.call_args.kwargs["prewarm_config"] is m["make_call_config"].return_value
        rt.close()


class TestPerCallValidation:
    def test_accepts_device_tensor_allocated_by_same_worker(self, patched_setup):
        submitted_tensors: dict[str, Any] = {}

        def submit_dispatch(*args):
            submitted_tensors.update(args[2])
            return _ImmediateNativeHandle()

        patched_setup["submit_dispatch"].side_effect = submit_dispatch
        compiled = _fake_compiled([_param("a", [128, 128]), _param("b", [128, 128])], [])
        rt = DistributedWorker(compiled)
        rt(_resident(rt, (128, 128)), _resident(rt, (128, 128)))
        patched_setup["submit_dispatch"].assert_called_once()
        assert set(submitted_tensors) == {"a", "b"}
        rt.close()

    def test_rejects_raw_device_tensor_before_dispatch(self, patched_setup):
        compiled = _fake_compiled([_param("a", [128, 128])], [])
        rt = DistributedWorker(compiled)

        with pytest.raises(
            TypeError, match=r"raw-pointer DeviceTensor.*same DistributedWorker\.alloc_tensor"
        ):
            rt(DeviceTensor(0x1000, (128, 128), torch.float32))

        patched_setup["submit_dispatch"].assert_not_called()
        rt.close()

    def test_rejects_device_tensor_owned_by_another_worker(self, patched_setup):
        compiled = _fake_compiled([_param("a", [128, 128])], [])
        owner = DistributedWorker(compiled)
        foreign = _resident(owner, (128, 128))
        consumer = DistributedWorker(compiled)

        with pytest.raises(ValueError, match="not a live allocation owned by this DistributedWorker"):
            consumer(foreign)

        patched_setup["submit_dispatch"].assert_not_called()
        consumer.close()
        owner.close()

    def test_rejects_stacked_tensor_with_raw_shard(self, patched_setup):
        compiled = _fake_compiled([_param("a", [1, 128, 128])], [])
        rt = DistributedWorker(compiled)
        raw = DeviceTensor(0x1000, (128, 128), torch.float32)
        stacked = StackedDeviceTensor((raw,), (1, 128, 128), (0,))

        with pytest.raises(TypeError, match=r"shard 0.*raw-pointer DeviceTensor"):
            rt(stacked)

        patched_setup["submit_dispatch"].assert_not_called()
        rt.close()

    def test_accepts_shared_host_torch_tensor(self, patched_setup):
        compiled = _fake_compiled([_param("a", [128, 128]), _param("b", [128, 128])], [])
        rt = DistributedWorker(compiled)
        host_a = torch.zeros(128, 128, dtype=torch.float32).share_memory_()
        rt(host_a, _resident(rt, (128, 128)))
        patched_setup["submit_dispatch"].assert_called_once()
        rt.close()

    def test_rejects_non_shared_host_torch_tensor(self, patched_setup):
        compiled = _fake_compiled([_param("a", [128, 128]), _param("b", [128, 128])], [])
        rt = DistributedWorker(compiled)
        with pytest.raises(TypeError, match="shared memory"):
            rt(torch.zeros(128, 128), _resident(rt, (128, 128)))
        rt.close()

    def test_releasing_compatibility_refs_does_not_disable_staged_uploads(self, patched_setup):
        compiled = _fake_compiled([_param("weight", [4, 4])], [])
        weight = torch.zeros(4, 4, dtype=torch.float32)
        rt = DistributedWorker(compiled, inherited_host_tensors=[weight])

        rt.release_inherited_host_tensor_refs()
        rt.release_inherited_host_tensor_refs()

        assert rt._inherited_host_tensors == ()
        rt.alloc_tensor(weight.shape, weight.dtype, init=weight)
        patched_setup["worker"].copy_to.assert_called_once()
        rt.close()

    def test_registered_tensor_still_requires_shared_memory_for_dispatch(self, patched_setup):
        compiled = _fake_compiled([_param("buffer", [128, 128])], [])
        buffer = torch.zeros(128, 128, dtype=torch.float32)
        rt = DistributedWorker(compiled, inherited_host_tensors=[buffer])

        with pytest.raises(TypeError, match="shared memory"):
            rt(buffer)

        rt.close()

    @pytest.mark.parametrize(
        ("weight", "expected_exception"),
        [
            (object(), TypeError),
            (torch.zeros(128, 128, dtype=torch.float32).t(), ValueError),
            (torch.empty(1, device="meta"), ValueError),
        ],
    )
    def test_rejects_invalid_prefork_tensor_registration(self, patched_setup, weight, expected_exception):
        compiled = _fake_compiled([_param("weight", [128, 128])], [])

        with pytest.raises(expected_exception, match=r"torch\.Tensor|contiguous CPU"):
            DistributedWorker(compiled, inherited_host_tensors=[weight])

    def test_scalar_param_forwarded_as_is(self, patched_setup):
        # Scalar params (shape=None, e.g. seq_len) bypass tensor validation and
        # are forwarded verbatim to the entry — common in serving dispatch.
        submitted_tensors: dict[str, Any] = {}

        def submit_dispatch(*args):
            submitted_tensors.update(args[2])
            return _ImmediateNativeHandle()

        patched_setup["submit_dispatch"].side_effect = submit_dispatch
        scalar = _ParamInfo(name="seq_len", direction=ParamDirection.In, shape=None, dtype=DataType.FP32)
        compiled = _fake_compiled([scalar, _param("kv", [16, 16])], [])
        rt = DistributedWorker(compiled)
        rt(7, _resident(rt, (16, 16)))
        assert submitted_tensors["seq_len"] == 7
        rt.close()

    def test_rejects_wrong_arg_count(self, patched_setup):
        compiled = _fake_compiled([_param("a", [128, 128]), _param("b", [128, 128])], [])
        rt = DistributedWorker(compiled)
        with pytest.raises(TypeError, match="expects 2 arguments"):
            rt(DeviceTensor(0x1000, (128, 128), torch.float32))
        rt.close()

    def test_validates_device_tensor_shape(self, patched_setup):
        compiled = _fake_compiled([_param("a", [128, 128]), _param("b", [128, 128])], [])
        rt = DistributedWorker(compiled)
        wrong = _resident(rt, (64, 64))
        valid = _resident(rt, (128, 128))
        with pytest.raises(TypeError, match="shape"):
            rt(wrong, valid)
        rt.close()


class TestDeviceMemoryApi:
    def test_pointer_reuse_old_tensor_cannot_free_new_allocation(self, patched_setup):
        old_buffer = _FakeBuffer(0xDEAD0000, 1024)
        new_buffer = _FakeBuffer(0xDEAD0000, 1024)
        worker = patched_setup["worker"]
        worker.alloc_child_tensor.side_effect = [old_buffer, new_buffer]
        rt = DistributedWorker(_fake_compiled([_param("a", [16, 16])], []))

        old = rt.alloc_tensor((16, 16), torch.float32)
        rt.free_tensor(old)
        new = rt.alloc_tensor((16, 16), torch.float32)

        worker.free.reset_mock()
        with pytest.raises(ValueError, match="stale DeviceTensor"):
            rt.free_tensor(old)
        worker.free.assert_not_called()
        assert rt._buffer_for_ptr(new.data_ptr) is new_buffer

        rt.free_tensor(new)
        worker.free.assert_called_once_with(new_buffer)
        rt.close()

    def test_free_tensor_failure_retains_buffer_for_retry(self, patched_setup):
        rt = DistributedWorker(_fake_compiled([_param("a", [16, 16])], []))
        dev = rt.alloc_tensor((16, 16), torch.float32)
        worker = patched_setup["worker"]
        worker.free.side_effect = RuntimeError("admission failed")

        with pytest.raises(RuntimeError, match="admission failed"):
            rt.free_tensor(dev)
        assert (0, dev.data_ptr) in rt._owned_tensors
        assert rt._buffer_for_ptr(dev.data_ptr) is dev.buffer

        worker.free.side_effect = None
        rt.free_tensor(dev)
        assert (0, dev.data_ptr) not in rt._owned_tensors
        assert worker.free.call_count == 2
        rt.close()

    def test_copy_rejects_interior_pointer_with_precise_guidance(self, patched_setup):
        rt = DistributedWorker(_fake_compiled([_param("a", [16, 16])], []))
        ptr = rt.malloc(64)
        host = (ctypes.c_ubyte * 32)()
        with pytest.raises(ValueError, match="interior pointer"):
            rt.copy_to(ptr + 32, ctypes.addressof(host), 32)
        rt.free(ptr)
        rt.close()

    def test_import_ipc_all_delegates_to_simpler_worker(self, patched_setup):
        compiled = _fake_compiled([_param("a", [16, 16])], [])
        rt = DistributedWorker(compiled)
        keys = {8: b"a" * 256, 9: b"b" * 256}
        patched_setup["worker"].import_ipc_all.return_value = {
            8: 0x10000008,
            9: 0x10000009,
        }

        assert rt.import_ipc_all(keys) == {8: 0x10000008, 9: 0x10000009}
        patched_setup["worker"].import_ipc_all.assert_called_once_with(keys, region_bytes=None)
        rt.close()

    def test_import_ipc_all_rejects_after_close(self, patched_setup):
        compiled = _fake_compiled([_param("a", [16, 16])], [])
        rt = DistributedWorker(compiled)
        rt.close()

        with pytest.raises(RuntimeError, match="import_ipc_all"):
            rt.import_ipc_all({8: b"a" * 256})

    def test_alloc_tensor_forwards_malloc_and_copy(self, patched_setup):
        compiled = _fake_compiled([_param("a", [16, 16])], [])
        rt = DistributedWorker(compiled)
        # init must be a CPU, contiguous, shared-memory tensor (read by the
        # forked chip worker via the inherited mapping).
        host = torch.arange(256, dtype=torch.float32).view(16, 16).share_memory_()

        dev = rt.alloc_tensor((16, 16), torch.float32, init=host)

        assert isinstance(dev, DeviceTensor)
        assert dev.data_ptr == 0xDEAD0000
        assert dev.shape == (16, 16)
        child = patched_setup["worker"].alloc_child_tensor.return_value
        assert dev.buffer is child
        alloc_args = patched_setup["worker"].alloc_child_tensor.call_args.args
        assert alloc_args[:2] == (0, (16 * 16 * 4,))
        # PyPTO stages the raw host-pointer API through a self-describing host
        # Buffer because forked L3 children cannot dereference parent pointers.
        copy_args = patched_setup["worker"].copy_to.call_args.args
        assert copy_args[0] is child
        staging = copy_args[1]
        assert ctypes.string_at(staging.base, 16 * 16 * 4) == ctypes.string_at(host.data_ptr(), 16 * 16 * 4)
        patched_setup["worker"].release_buffer.assert_called_once_with(staging)
        rt.close()

    def test_alloc_tensor_accepts_ordinary_post_prepare_init(self, patched_setup):
        compiled = _fake_compiled([_param("a", [16, 16])], [])
        rt = DistributedWorker(compiled)
        dev = rt.alloc_tensor((16, 16), torch.float32, init=torch.zeros(16, 16, dtype=torch.float32))
        assert dev.buffer is patched_setup["worker"].alloc_child_tensor.return_value
        patched_setup["worker"].copy_to.assert_called_once()
        rt.close()

    def test_alloc_tensor_rolls_back_on_copy_failure(self, patched_setup):
        compiled = _fake_compiled([_param("a", [16, 16])], [])
        rt = DistributedWorker(compiled)
        patched_setup["worker"].copy_to.side_effect = RuntimeError("boom")
        host = torch.zeros(16, 16, dtype=torch.float32).share_memory_()

        with pytest.raises(RuntimeError, match="boom"):
            rt.alloc_tensor((16, 16), torch.float32, init=host)

        # malloc'd pointer is freed on the failure path.
        child = patched_setup["worker"].alloc_child_tensor.return_value
        patched_setup["worker"].free.assert_called_once_with(child)
        staging = patched_setup["worker"].copy_to.call_args.args[1]
        patched_setup["worker"].release_buffer.assert_called_once_with(staging)
        rt.close()

    def test_alloc_tensor_forwards_nonzero_worker_id(self, patched_setup):
        # A non-default worker_id is supported through alloc_child_tensor and
        # tracked under (worker_id, ptr) for per-worker auto-free.
        compiled = _fake_compiled([_param("a", [16, 16])], [])
        rt = DistributedWorker(compiled)
        dev = rt.alloc_tensor((16, 16), torch.float32, worker_id=1)
        alloc_args = patched_setup["worker"].alloc_child_tensor.call_args.args
        assert alloc_args[:2] == (1, (16 * 16 * 4,))
        assert (1, dev.data_ptr) in rt._owned_tensors
        rt.free_tensor(dev, worker_id=1)
        patched_setup["worker"].free.assert_called_once_with(dev.buffer)
        rt.close()


def _compiled_2cards():
    compiled = _fake_compiled([_param("b", [2, 4, 4])], [])
    compiled._distributed_config = DistributedConfig(device_ids=[0, 1])
    return compiled


class TestAllocStackedTensor:
    """``alloc_stacked_tensor`` uploads each leading-dim shard to its worker once."""

    def test_identity_uploads_shard_per_worker(self, patched_setup):
        buffers = {0: _FakeBuffer(0xA000, 64), 1: _FakeBuffer(0xB000, 64)}
        _alloc_by_worker(patched_setup["worker"], buffers)
        rt = DistributedWorker(_compiled_2cards())
        host = torch.arange(2 * 4 * 4, dtype=torch.float32).view(2, 4, 4).share_memory_()

        stacked = rt.alloc_stacked_tensor(host)  # default worker_ids = range(2)

        assert stacked.full_shape == (2, 4, 4)
        assert stacked.worker_ids == (0, 1)
        assert tuple(s.shape for s in stacked.shards) == ((4, 4), (4, 4))
        worker = patched_setup["worker"]
        # shard 0 -> worker 0, shard 1 -> worker 1. The pairing is asserted; the order the
        # two uploads are issued in is not, because they now run concurrently.
        nbytes = 4 * 4 * 4
        allocs = worker.alloc_child_tensor.call_args_list
        assert sorted(call.args[:2] for call in allocs) == [(0, (nbytes,)), (1, (nbytes,))]
        assert sorted(call.args[0].base for call in worker.copy_to.call_args_list) == sorted(
            buffer.base for buffer in buffers.values()
        )
        # Tracked per (worker_id, ptr) for auto-free.
        assert (0, 0xA000) in rt._owned_tensors
        assert (1, 0xB000) in rt._owned_tensors
        rt.close()

    def test_registered_inherited_storage_uploads_without_shared_memory(self, patched_setup):
        buffers = {0: _FakeBuffer(0xA000, 64), 1: _FakeBuffer(0xB000, 64)}
        _alloc_by_worker(patched_setup["worker"], buffers)
        host = torch.arange(2 * 4 * 4, dtype=torch.float32).view(2, 4, 4)
        rt = DistributedWorker(_compiled_2cards(), inherited_host_tensors=[host])

        stacked = rt.alloc_stacked_tensor(host)

        assert stacked.worker_ids == (0, 1)
        worker = patched_setup["worker"]
        assert sorted(call.args[0].base for call in worker.copy_to.call_args_list) == sorted(
            buffer.base for buffer in buffers.values()
        )
        rt.close()

    def test_permuted_worker_ids_place_shards(self, patched_setup):
        buffers = {1: _FakeBuffer(0xA000, 64), 0: _FakeBuffer(0xB000, 64)}
        _alloc_by_worker(patched_setup["worker"], buffers)
        rt = DistributedWorker(_compiled_2cards())
        host = torch.zeros(2, 4, 4, dtype=torch.float32).share_memory_()

        stacked = rt.alloc_stacked_tensor(host, worker_ids=[1, 0])

        assert stacked.worker_ids == (1, 0)
        worker = patched_setup["worker"]
        nbytes = 4 * 4 * 4
        # shard 0 -> worker 1, shard 1 -> worker 0: the placement, not the issue order.
        allocs = worker.alloc_child_tensor.call_args_list
        assert sorted(call.args[:2] for call in allocs) == [(0, (nbytes,)), (1, (nbytes,))]
        assert (1, 0xA000) in rt._owned_tensors
        assert (0, 0xB000) in rt._owned_tensors
        rt.close()

    def test_free_stacked_tensor_releases_each_shard(self, patched_setup):
        buffers = {1: _FakeBuffer(0xA000, 64), 0: _FakeBuffer(0xB000, 64)}
        _alloc_by_worker(patched_setup["worker"], buffers)
        rt = DistributedWorker(_compiled_2cards())
        host = torch.zeros(2, 4, 4, dtype=torch.float32).share_memory_()
        stacked = rt.alloc_stacked_tensor(host, worker_ids=[1, 0])

        patched_setup["worker"].free.reset_mock()
        rt.free_stacked_tensor(stacked)

        worker = patched_setup["worker"]
        worker.free.assert_any_call(buffers[1])
        worker.free.assert_any_call(buffers[0])
        assert (1, 0xA000) not in rt._owned_tensors
        assert (0, 0xB000) not in rt._owned_tensors
        rt.close()

    def test_close_auto_frees_stacked_shards(self, patched_setup):
        buffers = {0: _FakeBuffer(0xA000, 64), 1: _FakeBuffer(0xB000, 64)}
        _alloc_by_worker(patched_setup["worker"], buffers)
        rt = DistributedWorker(_compiled_2cards())
        host = torch.zeros(2, 4, 4, dtype=torch.float32).share_memory_()
        rt.alloc_stacked_tensor(host)  # leak — close() must release both shards

        patched_setup["worker"].free.reset_mock()
        rt.close()
        worker = patched_setup["worker"]
        worker.free.assert_any_call(buffers[0])
        worker.free.assert_any_call(buffers[1])

    def test_worker_ids_out_of_range_rejected(self, patched_setup):
        rt = DistributedWorker(_compiled_2cards())
        host = torch.zeros(2, 4, 4, dtype=torch.float32).share_memory_()
        with pytest.raises(ValueError, match="out of range"):
            rt.alloc_stacked_tensor(host, worker_ids=[0, 5])
        rt.close()

    def test_empty_leading_dim_rejected(self, patched_setup):
        # B == 0 must fail cleanly (before any malloc), not build an empty
        # StackedDeviceTensor that IndexErrors on .dtype / __repr__.
        rt = DistributedWorker(_compiled_2cards())
        host = torch.zeros(0, 4, 4, dtype=torch.float32).share_memory_()
        with pytest.raises(ValueError, match="at least one shard"):
            rt.alloc_stacked_tensor(host)
        patched_setup["worker"].alloc_child_tensor.assert_not_called()
        rt.close()

    def test_worker_ids_length_mismatch_rejected(self, patched_setup):
        rt = DistributedWorker(_compiled_2cards())
        host = torch.zeros(2, 4, 4, dtype=torch.float32).share_memory_()
        with pytest.raises(ValueError, match="entries"):
            rt.alloc_stacked_tensor(host, worker_ids=[0])
        rt.close()

    def test_non_shared_host_is_staged_and_uploaded(self, patched_setup):
        patched_setup["worker"].alloc_child_tensor.side_effect = [
            _FakeBuffer(0xA000, 64),
            _FakeBuffer(0xB000, 64),
        ]
        rt = DistributedWorker(_compiled_2cards())
        host = torch.zeros(2, 4, 4, dtype=torch.float32)  # NOT shared

        stacked = rt.alloc_stacked_tensor(host)
        assert stacked.worker_ids == (0, 1)
        assert patched_setup["worker"].copy_to.call_count == 2
        rt.close()


class TestCopyStackedFrom:
    """``copy_stacked_from`` reads each resident shard back into host[i] (D2H)."""

    def _make_stacked(self, patched_setup, worker_ids=None):
        patched_setup["worker"].alloc_child_tensor.side_effect = [
            _FakeBuffer(0xA000, 64),
            _FakeBuffer(0xB000, 64),
        ]
        rt = DistributedWorker(_compiled_2cards())
        host = torch.zeros(2, 4, 4, dtype=torch.float32).share_memory_()
        stacked = rt.alloc_stacked_tensor(host, worker_ids=worker_ids)
        patched_setup["worker"].copy_from.reset_mock()
        patched_setup["worker"].release_buffer.reset_mock()
        return rt, stacked

    def test_reads_each_shard_back(self, patched_setup):
        rt, stacked = self._make_stacked(patched_setup)  # worker_ids == (0, 1)
        out = torch.zeros(2, 4, 4, dtype=torch.float32).share_memory_()

        rt.copy_stacked_from(stacked, out)

        worker = patched_setup["worker"]
        calls = worker.copy_from.call_args_list
        assert [call.args[1] for call in calls] == [stacked.shards[0].buffer, stacked.shards[1].buffer]
        assert all(isinstance(call.args[0], _FakeBuffer) for call in calls)
        assert worker.release_buffer.call_count == 2
        assert worker.copy_from.call_count == 2
        rt.close()

    def test_permuted_worker_ids(self, patched_setup):
        rt, stacked = self._make_stacked(patched_setup, worker_ids=[1, 0])
        out = torch.zeros(2, 4, 4, dtype=torch.float32).share_memory_()

        rt.copy_stacked_from(stacked, out)

        worker = patched_setup["worker"]
        # shard 0 resides on worker 1, shard 1 on worker 0.
        calls = worker.copy_from.call_args_list
        assert [call.args[1] for call in calls] == [stacked.shards[0].buffer, stacked.shards[1].buffer]
        rt.close()

    def test_shape_mismatch_rejected(self, patched_setup):
        rt, stacked = self._make_stacked(patched_setup)
        out = torch.zeros(3, 4, 4, dtype=torch.float32).share_memory_()
        with pytest.raises(ValueError, match="does not match stacked full_shape"):
            rt.copy_stacked_from(stacked, out)
        rt.close()

    def test_dtype_mismatch_rejected(self, patched_setup):
        rt, stacked = self._make_stacked(patched_setup)
        out = torch.zeros(2, 4, 4, dtype=torch.float16).share_memory_()
        with pytest.raises(ValueError, match="does not match stacked dtype"):
            rt.copy_stacked_from(stacked, out)
        rt.close()

    def test_non_shared_post_prepare_host_is_accepted(self, patched_setup):
        rt, stacked = self._make_stacked(patched_setup)
        out = torch.zeros(2, 4, 4, dtype=torch.float32)  # NOT shared
        rt.copy_stacked_from(stacked, out)
        assert patched_setup["worker"].copy_from.call_count == 2
        rt.close()

    def test_non_contiguous_host_rejected(self, patched_setup):
        rt, stacked = self._make_stacked(patched_setup)
        # Shared but transposed -> non-contiguous; still rejected.
        out = torch.zeros(2, 4, 4, dtype=torch.float32).share_memory_().transpose(1, 2)
        assert not out.is_contiguous()
        with pytest.raises(ValueError, match="CPU, contiguous"):
            rt.copy_stacked_from(stacked, out)
        rt.close()

    def test_wrong_type_rejected(self, patched_setup):
        rt, _stacked = self._make_stacked(patched_setup)
        out = torch.zeros(2, 4, 4, dtype=torch.float32).share_memory_()
        with pytest.raises(TypeError, match="expects a StackedDeviceTensor"):
            rt.copy_stacked_from(object(), out)  # type: ignore[arg-type]  # runtime guard under test
        rt.close()

    def test_after_close_raises(self, patched_setup):
        rt, stacked = self._make_stacked(patched_setup)
        out = torch.zeros(2, 4, 4, dtype=torch.float32).share_memory_()
        rt.close()
        with pytest.raises(RuntimeError, match="called after close"):
            rt.copy_stacked_from(stacked, out)


class TestLifecycle:
    def test_close_idempotent_and_closes_worker(self, patched_setup):
        compiled = _fake_compiled([_param("a", [16, 16])], [])
        rt = DistributedWorker(compiled)
        rt.close()
        rt.close()  # second close is a no-op
        assert patched_setup["worker"].close.call_count == 1

    def test_close_retries_worker_cleanup_after_failure(self, patched_setup):
        compiled = _fake_compiled([_param("weight", [16, 16])], [])
        weight = torch.zeros(16, 16, dtype=torch.float32)
        rt = DistributedWorker(compiled, inherited_host_tensors=[weight])
        worker = patched_setup["worker"]
        worker.close.side_effect = [RuntimeError("worker close failed"), None]

        with pytest.raises(RuntimeError, match="worker close failed"):
            rt.close()

        assert rt._closed is True
        assert rt._close_complete is False
        assert rt._inherited_host_tensors == ()
        with pytest.raises(RuntimeError, match="after close"):
            rt(DeviceTensor(0x1000, (16, 16), torch.float32))

        rt.close()
        assert rt._close_complete is True
        rt.close()  # cleanup completed, so further calls are no-ops
        assert worker.close.call_count == 2

    def test_concurrent_close_runs_teardown_once(self, patched_setup):
        compiled = _fake_compiled([_param("a", [16, 16])], [])
        rt = DistributedWorker(compiled)
        worker = patched_setup["worker"]
        close_started = threading.Event()
        allow_close = threading.Event()

        def worker_close() -> None:
            close_started.set()
            assert allow_close.wait(timeout=2)

        worker.close.side_effect = worker_close
        first = threading.Thread(target=rt.close)
        second = threading.Thread(target=rt.close)
        first.start()
        assert close_started.wait(timeout=2)
        second.start()
        second.join(timeout=2)
        assert not second.is_alive()
        worker.close.assert_called_once_with()

        allow_close.set()
        first.join(timeout=2)
        assert not first.is_alive()
        assert rt._close_complete is True

    def test_context_manager_closes(self, patched_setup):
        compiled = _fake_compiled([_param("a", [16, 16])], [])
        with DistributedWorker(compiled) as rt:
            assert rt is not None
        assert patched_setup["worker"].close.call_count == 1

    def test_call_after_close_raises(self, patched_setup):
        compiled = _fake_compiled([_param("a", [16, 16])], [])
        rt = DistributedWorker(compiled)
        rt.close()
        with pytest.raises(RuntimeError, match="after close"):
            rt(DeviceTensor(0x1000, (16, 16), torch.float32))


class TestCallbacks:
    def test_callback_reaches_register(self, patched_setup):
        m = patched_setup
        placeholder = object()

        def real(args):
            return None

        m["load_subs"].return_value = {"sample_and_prepare": placeholder}
        compiled = _fake_compiled([_param("a", [8, 8])], [])

        rt = DistributedWorker(compiled, callbacks={"sample_and_prepare": real})

        # _register_callables(w, sub_worker_fns, chip_callables): arg[1] is the bound set.
        passed = m["register"].call_args.args[1]
        assert passed == {"sample_and_prepare": real}
        rt.close()

    def test_no_callback_passes_loaded_unchanged(self, patched_setup):
        m = patched_setup
        loaded = {"sample_and_prepare": object()}
        m["load_subs"].return_value = loaded
        compiled = _fake_compiled([_param("a", [8, 8])], [])

        rt = DistributedWorker(compiled)

        assert m["register"].call_args.args[1] == loaded
        rt.close()

    def test_callback_unknown_name_raises(self, patched_setup):
        m = patched_setup
        m["load_subs"].return_value = {"sample_and_prepare": object()}
        compiled = _fake_compiled([_param("a", [8, 8])], [])

        with pytest.raises(ValueError, match="not sub-workers"):
            DistributedWorker(compiled, callbacks={"typo": lambda args: None})

    def test_missing_required_callback_raises(self, patched_setup):
        m = patched_setup
        m["load_subs"].return_value = {"sample": object()}
        m["load_required"].return_value = {"sample"}
        compiled = _fake_compiled([_param("a", [8, 8])], [])

        with pytest.raises(ValueError, match="runtime-bound callbacks"):
            DistributedWorker(compiled)  # abstract SubWorker not supplied

    def test_deprecated_alias_warns_and_binds(self, patched_setup):
        m = patched_setup

        def real(args):
            return None

        m["load_subs"].return_value = {"sample_and_prepare": object()}
        compiled = _fake_compiled([_param("a", [8, 8])], [])

        with pytest.warns(DeprecationWarning, match="sub_worker_overrides is deprecated"):
            rt = DistributedWorker(compiled, sub_worker_overrides={"sample_and_prepare": real})

        assert m["register"].call_args.args[1] == {"sample_and_prepare": real}
        rt.close()


class TestBindSubWorkers:
    def test_none_callbacks_returns_equal_set(self):
        from pypto.runtime.distributed_runner import _bind_sub_workers  # noqa: PLC0415

        loaded = {"a": object()}
        assert _bind_sub_workers(loaded, None, set()) == loaded
        assert _bind_sub_workers(loaded, {}, set()) == loaded

    def test_valid_callback_replaces(self):
        from pypto.runtime.distributed_runner import _bind_sub_workers  # noqa: PLC0415

        placeholder, other = object(), object()

        def real(args):
            return None

        loaded = {"a": placeholder, "b": other}
        bound = _bind_sub_workers(loaded, {"a": real}, set())
        assert bound == {"a": real, "b": other}

    def test_unknown_name_raises_listing_available(self):
        from pypto.runtime.distributed_runner import _bind_sub_workers  # noqa: PLC0415

        with pytest.raises(ValueError, match=r"not sub-workers.*Available sub-workers"):
            _bind_sub_workers({"a": object()}, {"b": lambda args: None}, set())

    def test_missing_required_raises(self):
        from pypto.runtime.distributed_runner import _bind_sub_workers  # noqa: PLC0415

        with pytest.raises(ValueError, match="runtime-bound callbacks"):
            _bind_sub_workers({"sample": object()}, None, {"sample"})

    def test_bad_arity_callback_rejected(self):
        from pypto.runtime.distributed_runner import _bind_sub_workers  # noqa: PLC0415

        with pytest.raises(TypeError, match="single positional"):
            _bind_sub_workers({"a": object()}, {"a": lambda: None}, set())


class TestOneShotRegression:
    """The one-shot execute_distributed path still works after helper extraction."""

    def test_one_shot_setup_dispatch_close(self, patched_setup):
        from pypto.runtime.distributed_runner import execute_distributed  # noqa: PLC0415

        compiled = _fake_compiled([_param("a", [8, 8]), _param("b", [8, 8])], [])
        a = torch.zeros(8, 8, dtype=torch.float32)
        b = torch.zeros(8, 8, dtype=torch.float32)

        execute_distributed(compiled, [a, b])

        patched_setup["assemble"].assert_called_once()
        patched_setup["construct"].assert_called_once()
        patched_setup["worker"].init.assert_called_once()
        patched_setup["dispatch"].assert_called_once()
        patched_setup["worker"].close.assert_called_once()

    def test_one_shot_rejects_resident_tensor_before_setup(self, patched_setup):
        from pypto.runtime.distributed_runner import execute_distributed  # noqa: PLC0415

        compiled = _fake_compiled([_param("a", [8, 8])], [])
        with pytest.raises(TypeError, match=r"same prepared DistributedWorker"):
            execute_distributed(compiled, [DeviceTensor(0x1000, (8, 8), torch.float32)])
        patched_setup["assemble"].assert_not_called()

    def test_one_shot_enables_sdma_when_a_chip_requires_it(self, patched_setup):
        from pypto.runtime.distributed_runner import execute_distributed  # noqa: PLC0415

        patched_setup["assemble"].return_value = ({"chip_orch": object()}, "rt_name", True)
        compiled = _fake_compiled([_param("a", [8, 8])], [])

        execute_distributed(compiled, [torch.zeros(8, 8, dtype=torch.float32)])

        assert patched_setup["construct"].call_args.kwargs["enable_sdma"] is True

    def test_one_shot_retries_incomplete_worker_cleanup(self, patched_setup):
        from pypto.runtime.distributed_runner import execute_distributed  # noqa: PLC0415

        worker = patched_setup["worker"]
        worker.close.side_effect = [RuntimeError("cleanup pending"), None]
        compiled = _fake_compiled([_param("a", [8, 8])], [])

        execute_distributed(compiled, [torch.zeros(8, 8, dtype=torch.float32)])

        assert worker.close.call_count == 2


class TestWorkerConstruction:
    def test_forwards_enable_sdma_to_simpler_worker(self, monkeypatch):
        worker_cls = MagicMock(name="simpler.Worker")
        monkeypatch.setitem(sys.modules, "simpler.worker", SimpleNamespace(Worker=worker_cls))
        dc = DistributedConfig(device_ids=[0, 1])

        _construct_worker(dc, "a2a3", "tensormap_and_ringbuffer", 3, enable_sdma=True)

        worker_cls.assert_called_once_with(
            level=3,
            device_ids=[0, 1],
            num_sub_workers=3,
            platform="a2a3",
            runtime="tensormap_and_ringbuffer",
            enable_sdma=True,
        )

    def test_forwards_startup_timeout_to_simpler_worker(self, monkeypatch):
        worker_cls = MagicMock(name="simpler.Worker")
        monkeypatch.setitem(sys.modules, "simpler.worker", SimpleNamespace(Worker=worker_cls))
        dc = DistributedConfig(device_ids=[0, 1])

        _construct_worker(
            dc,
            "a2a3",
            "tensormap_and_ringbuffer",
            3,
            startup_timeout_s=1800.0,
        )

        assert worker_cls.call_args.kwargs["startup_timeout_s"] == 1800.0

    @pytest.mark.parametrize(
        "startup_timeout_s",
        [0.0, -1.0, float("inf"), float("-inf"), float("nan")],
    )
    def test_rejects_invalid_startup_timeout_before_worker_construction(self, monkeypatch, startup_timeout_s):
        worker_cls = MagicMock(name="simpler.Worker")
        monkeypatch.setitem(sys.modules, "simpler.worker", SimpleNamespace(Worker=worker_cls))
        dc = DistributedConfig(device_ids=[0, 1])

        with pytest.raises(ValueError, match="positive finite"):
            _construct_worker(
                dc,
                "a2a3",
                "tensormap_and_ringbuffer",
                3,
                startup_timeout_s=startup_timeout_s,
            )

        worker_cls.assert_not_called()

    def test_distributed_worker_forwards_startup_timeout(self, patched_setup):
        compiled = _fake_compiled([_param("a", [8, 8])], [])

        rt = DistributedWorker(compiled, startup_timeout_s=1800.0)

        assert patched_setup["construct"].call_args.kwargs["startup_timeout_s"] == 1800.0
        rt.close()

    def test_failure_preserves_primary_error_when_cleanup_retry_fails(self, patched_setup, caplog):
        worker = patched_setup["worker"]
        worker.init.side_effect = RuntimeError("init failed")
        worker.close.side_effect = [
            RuntimeError("cleanup pending"),
            RuntimeError("cleanup still pending"),
        ]
        compiled = _fake_compiled([_param("a", [8, 8])], [])

        with pytest.raises(RuntimeError, match="init failed"):
            DistributedWorker(compiled)

        assert worker.close.call_count == 2
        assert "Worker cleanup was interrupted or still failed after one retry" in caplog.text

    def test_interrupted_failure_still_retries_cleanup_and_preserves_primary(self, patched_setup, caplog):
        worker = patched_setup["worker"]
        worker.init.side_effect = KeyboardInterrupt("init interrupted")
        worker.close.side_effect = [KeyboardInterrupt("cleanup interrupted"), None]
        compiled = _fake_compiled([_param("a", [8, 8])], [])

        with pytest.raises(KeyboardInterrupt, match="init interrupted"):
            DistributedWorker(compiled)

        assert worker.close.call_count == 2
        assert "Worker cleanup was interrupted or still failed after one retry" in caplog.text


class TestExplicitDispatchAPI:
    """The new ``run`` / ``register`` surface that mirrors ChipWorker.

    DistributedWorker.run() and ``__call__`` are blocking submit/result
    compositions. register() returns a :class:`RegistrationHandle` whose call
    delegates to run().
    """

    def test_run_delegates_to_call(self, patched_setup):
        from pypto.runtime import RegistrationHandle  # noqa: PLC0415

        compiled = _fake_compiled([_param("a", [4]), _param("b", [4])], [])
        rt = DistributedWorker(compiled)

        a = torch.zeros(4).share_memory_()
        b = torch.zeros(4).share_memory_()
        rt.run(compiled, a, b)
        patched_setup["submit_dispatch"].assert_called_once()

        # register() returns a usable handle.
        rt2 = DistributedWorker(compiled)
        h = rt2.register(compiled)
        assert isinstance(h, RegistrationHandle)
        assert h.compiled is compiled
        rt.close()
        rt2.close()

    def test_run_rejects_unregistered_compiled(self, patched_setup):
        compiled_a = _fake_compiled([_param("a", [4])], [])
        compiled_b = _fake_compiled([_param("a", [4])], [])
        rt = DistributedWorker(compiled_a)
        a = torch.zeros(4).share_memory_()
        with pytest.raises(ValueError, match="registered when this worker"):
            rt.run(compiled_b, a)
        rt.close()

    def test_register_rejects_unregistered_compiled(self, patched_setup):
        compiled_a = _fake_compiled([_param("a", [4])], [])
        compiled_b = _fake_compiled([_param("a", [4])], [])
        rt = DistributedWorker(compiled_a)
        with pytest.raises(ValueError, match="registered when this worker"):
            rt.register(compiled_b)
        rt.close()

    def test_register_rejects_after_close(self, patched_setup):
        """register() after close() must raise; mirrors ChipWorker behaviour."""
        compiled = _fake_compiled([_param("a", [4])], [])
        rt = DistributedWorker(compiled)
        rt.close()
        with pytest.raises(RuntimeError, match="register"):
            rt.register(compiled)

    def test_handle_call_dispatches(self, patched_setup):
        compiled = _fake_compiled([_param("a", [4]), _param("b", [4])], [])
        rt = DistributedWorker(compiled)
        a = torch.zeros(4).share_memory_()
        b = torch.zeros(4).share_memory_()

        h = rt.register(compiled)
        patched_setup["submit_dispatch"].reset_mock()
        h(a, b)
        patched_setup["submit_dispatch"].assert_called_once()
        rt.close()

    def test_close_marks_handle_closed(self, patched_setup):
        compiled = _fake_compiled([_param("a", [4])], [])
        rt = DistributedWorker(compiled)
        h = rt.register(compiled)
        assert h.closed is False
        rt.close()
        assert h.closed is True

    def test_close_auto_frees_owned_device_tensors(self, patched_setup):
        """alloc_tensor on DistributedWorker is also tracked through the ABC."""
        compiled = _fake_compiled([_param("a", [4])], [])
        rt = DistributedWorker(compiled)

        # alloc_tensor goes through Worker ABC -> records in _owned_tensors.
        host = torch.zeros(4, dtype=torch.float32).share_memory_()
        t = rt.alloc_tensor((4,), torch.float32, init=host)
        assert (0, t.data_ptr) in rt._owned_tensors

        # Spy on Simpler's public free so we can assert close drove auto-free.
        worker = patched_setup["worker"]
        worker.free.reset_mock()
        rt.close()
        assert worker.free.called


class TestLoadOrchEntry:
    """Entry resolution in ``_load_orch_entry`` (issue #1678).

    The dispatch entry is the unique module-level function tagged with the
    ``_pypto_distributed_entry`` marker — resolution must not depend on the
    function's Python name nor fall back to scanning callables by name.
    """

    @staticmethod
    def _write_orch(tmp_path, src: str):
        orch_dir = tmp_path / "orchestration"
        orch_dir.mkdir()
        (orch_dir / "host_orch.py").write_text(src)
        return tmp_path

    def test_resolves_marked_function_not_imported_class(self, tmp_path):
        """Resolution follows the marker, never an alphabetically-earlier import
        such as ``CommBufferSpec`` (the original failure mode of issue #1678)."""
        from pypto.runtime.distributed_runner import _load_orch_entry  # noqa: PLC0415

        root = self._write_orch(
            tmp_path,
            "class CommBufferSpec:\n"
            "    def __init__(self, **kw):\n"
            "        raise AssertionError('wrong callable resolved')\n\n\n"
            "def moe_ep_l3(orch, _args, config, **kw):\n"
            "    return 'ok'\n\n\n"
            "moe_ep_l3._pypto_distributed_entry = True\n",
        )
        entry_fn, alloc_fn = _load_orch_entry(root)
        assert entry_fn.__name__ == "moe_ep_l3"
        assert alloc_fn is None

    def test_returns_alloc_intermediates_when_present(self, tmp_path):
        from pypto.runtime.distributed_runner import _load_orch_entry  # noqa: PLC0415

        root = self._write_orch(
            tmp_path,
            "def host_orch(orch, _args, config, **kw):\n"
            "    return 'ok'\n\n\n"
            "host_orch._pypto_distributed_entry = True\n\n\n"
            "def _alloc_intermediates(tensors):\n"
            "    return None\n",
        )
        entry_fn, alloc_fn = _load_orch_entry(root)
        assert entry_fn.__name__ == "host_orch"
        assert alloc_fn is not None and alloc_fn.__name__ == "_alloc_intermediates"

    def test_alloc_intermediates_accepts_both_signatures(self, tmp_path):
        """The runtime must call a one-arg allocator without ``world_size``.

        Codegen emits ``_alloc_intermediates(tensors, world_size=1)`` so it can size the
        per-rank comm ordering tokens, but a ``build_output/`` produced by an older pypto
        is still replayable via ``from_dir``, and callers may inject their own allocator.
        Both shapes must work.
        """
        from pypto.runtime.distributed_runner import (  # noqa: PLC0415
            _call_alloc_intermediates,
        )

        seen: dict[str, object] = {}

        def old_style(tensors):
            seen["old"] = True

        def new_style(tensors, world_size=1):
            seen["new"] = world_size

        _call_alloc_intermediates(old_style, {}, 4)
        _call_alloc_intermediates(new_style, {}, 4)
        assert seen == {"old": True, "new": 4}

        # A TypeError raised *inside* the allocator must propagate, not be mistaken for a
        # signature mismatch and retried with fewer arguments.
        def boom(tensors, world_size=1):
            raise TypeError("inner failure")

        with pytest.raises(TypeError, match="inner failure"):
            _call_alloc_intermediates(boom, {}, 4)

    def test_no_marker_raises(self, tmp_path):
        from pypto.runtime.distributed_runner import _load_orch_entry  # noqa: PLC0415

        root = self._write_orch(
            tmp_path,
            "def moe_ep_l3(orch, _args, config, **kw):\n    return 'ok'\n",
        )
        with pytest.raises(RuntimeError, match="exactly one entry function"):
            _load_orch_entry(root)

    def test_multiple_markers_raise(self, tmp_path):
        from pypto.runtime.distributed_runner import _load_orch_entry  # noqa: PLC0415

        root = self._write_orch(
            tmp_path,
            "def a(orch, _args, config, **kw):\n    return 'a'\n\n\n"
            "def b(orch, _args, config, **kw):\n    return 'b'\n\n\n"
            "a._pypto_distributed_entry = True\n"
            "b._pypto_distributed_entry = True\n",
        )
        with pytest.raises(RuntimeError, match="exactly one entry function"):
            _load_orch_entry(root)


class TestMultiProgram:
    """Multiple compatible programs share one L3 worker (issue #1698).

    Each program registers its own callables/entry/state; dispatch selects the
    program via ``run(compiled, ...)``. The shared worker is constructed and
    init()'d exactly once across all programs.
    """

    def test_prepares_multiple_programs_on_one_worker(self, patched_setup):
        m = patched_setup
        prog_a = _fake_compiled([_param("a", [4])], [])
        prog_b = _fake_compiled([_param("b", [8])], [])

        rt = DistributedWorker([prog_a, prog_b])

        # One worker, init()'d once; per-program setup ran twice.
        m["construct"].assert_called_once()
        m["worker"].init.assert_called_once()
        assert m["assemble"].call_count == 2
        assert m["load_entry"].call_count == 2
        assert m["register"].call_count == 2
        # Both programs are dispatchable; the first is primary.
        assert set(rt._states) == {prog_a, prog_b}
        assert rt._compiled is prog_a
        rt.close()

    def test_run_selects_program_state(self, patched_setup):
        m = patched_setup
        # Distinct entry_fns per program so we can prove dispatch picks the
        # selected program's state, not the primary's.
        entry_a, entry_b = MagicMock(name="entry_a"), MagicMock(name="entry_b")
        m["load_entry"].side_effect = [(entry_a, None), (entry_b, None)]
        prog_a = _fake_compiled([_param("a", [4])], [])
        prog_b = _fake_compiled([_param("b", [8])], [])
        rt = DistributedWorker([prog_a, prog_b])

        a = torch.zeros(4).share_memory_()
        b = torch.zeros(8).share_memory_()

        rt.run(prog_b, b)
        assert m["submit_dispatch"].call_args.args[1] is entry_b
        rt.run(prog_a, a)
        assert m["submit_dispatch"].call_args.args[1] is entry_a
        rt.close()

    def test_num_sub_workers_is_max_across_programs(self, patched_setup):
        m = patched_setup
        m["load_subs"].side_effect = [{"s0": object()}, {"s0": object(), "s1": object()}]
        prog_a = _fake_compiled([_param("a", [4])], [])
        prog_b = _fake_compiled([_param("b", [8])], [])

        rt = DistributedWorker([prog_a, prog_b])

        # _construct_worker(dc, platform, runtime_name, num_sub) — num_sub is the
        # max sub-worker count across all programs (2 here).
        assert m["construct"].call_args.args[3] == 2
        rt.close()

    def test_enables_sdma_when_any_program_requires_it(self, patched_setup):
        m = patched_setup
        m["assemble"].side_effect = [
            ({"chip_a": object()}, "rt_name", False),
            ({"chip_b": object()}, "rt_name", True),
        ]
        prog_a = _fake_compiled([_param("a", [4])], [])
        prog_b = _fake_compiled([_param("b", [8])], [])

        rt = DistributedWorker([prog_a, prog_b])

        assert m["construct"].call_args.kwargs["enable_sdma"] is True
        rt.close()

    def test_single_program_preserves_default_sdma_capability(self, patched_setup):
        rt = DistributedWorker(_fake_compiled([_param("a", [4])], []))

        assert patched_setup["construct"].call_args.kwargs["enable_sdma"] is False
        rt.close()

    def test_single_program_list_keeps_call_shortcut(self, patched_setup):
        # A one-element list is what ``compiled.prepare()`` builds; the
        # ``rt(*args)`` shortcut must keep working for it.
        prog = _fake_compiled([_param("a", [4])], [])
        rt = DistributedWorker([prog])
        assert rt._multi_program is False
        rt(torch.zeros(4).share_memory_())
        patched_setup["submit_dispatch"].assert_called_once()
        rt.close()

    def test_call_raises_in_multi_program_mode(self, patched_setup):
        prog_a = _fake_compiled([_param("a", [4])], [])
        prog_b = _fake_compiled([_param("b", [8])], [])
        rt = DistributedWorker([prog_a, prog_b])
        with pytest.raises(TypeError, match="ambiguous"):
            rt(torch.zeros(4).share_memory_())
        rt.close()

    def test_shared_device_tensor_across_programs(self, patched_setup):
        m = patched_setup
        submitted_tensors: list[dict[str, Any]] = []

        def submit_dispatch(*args):
            submitted_tensors.append(dict(args[2]))
            return _ImmediateNativeHandle()

        m["submit_dispatch"].side_effect = submit_dispatch
        # Both programs take a same-shaped KV param; one resident DeviceTensor
        # is dispatched through both (the serving KV-cache sharing contract).
        prog_a = _fake_compiled([_param("kv", [16, 16])], [])
        prog_b = _fake_compiled([_param("kv", [16, 16])], [])
        rt = DistributedWorker([prog_a, prog_b])

        kv = _resident(rt, (16, 16))
        rt.run(prog_a, kv)
        rt.run(prog_b, kv)

        assert m["submit_dispatch"].call_count == 2
        for tensors in submitted_tensors:
            assert tensors["kv"] is kv  # same pointer in both tensor maps
        rt.close()

    def test_register_each_program_returns_handle(self, patched_setup):
        from pypto.runtime import RegistrationHandle  # noqa: PLC0415

        m = patched_setup
        entry_a, entry_b = MagicMock(name="entry_a"), MagicMock(name="entry_b")
        m["load_entry"].side_effect = [(entry_a, None), (entry_b, None)]
        prog_a = _fake_compiled([_param("a", [4])], [])
        prog_b = _fake_compiled([_param("b", [8])], [])
        rt = DistributedWorker([prog_a, prog_b])

        h_a = rt.register(prog_a)
        h_b = rt.register(prog_b)
        assert isinstance(h_a, RegistrationHandle) and isinstance(h_b, RegistrationHandle)
        assert h_a.compiled is prog_a
        assert h_b.compiled is prog_b

        # Each handle dispatches its own program's state.
        h_a(torch.zeros(4).share_memory_())
        assert m["submit_dispatch"].call_args.args[1] is entry_a
        h_b(torch.zeros(8).share_memory_())
        assert m["submit_dispatch"].call_args.args[1] is entry_b

        # close() marks every program's handle closed and tears down the one worker.
        rt.close()
        assert h_a.closed is True
        assert h_b.closed is True
        assert m["worker"].close.call_count == 1

    def test_callbacks_apply_per_program(self, patched_setup):
        m = patched_setup

        # prog_a declares sub-worker 'sample'; prog_b declares 'route'. A callback
        # for each binds only to the program that declares it — heterogeneous
        # sub-worker sets across programs must not raise.
        def cb_sample(args):
            return None

        def cb_route(args):
            return None

        m["load_subs"].side_effect = [{"sample": object()}, {"route": object()}]
        prog_a = _fake_compiled([_param("a", [4])], [])
        prog_b = _fake_compiled([_param("b", [8])], [])

        rt = DistributedWorker([prog_a, prog_b], callbacks={"sample": cb_sample, "route": cb_route})

        bound_sets = [call.args[1] for call in m["register"].call_args_list]
        assert {"sample": cb_sample} in bound_sets
        assert {"route": cb_route} in bound_sets
        rt.close()

    def test_callback_matching_no_program_raises(self, patched_setup):
        m = patched_setup
        m["load_subs"].side_effect = [{"sample": object()}, {"route": object()}]
        prog_a = _fake_compiled([_param("a", [4])], [])
        prog_b = _fake_compiled([_param("b", [8])], [])
        with pytest.raises(ValueError, match="not sub-workers of any prepared program"):
            DistributedWorker([prog_a, prog_b], callbacks={"typo": lambda args: None})

    def test_prepare_extra_compiled_forwards_program_list(self):
        from pypto.ir.distributed_compiled_program import DistributedCompiledProgram  # noqa: PLC0415

        primary = _fake_compiled([_param("a", [4])], [])
        extra = _fake_compiled([_param("b", [8])], [])
        with patch("pypto.runtime.distributed_runner.DistributedWorker") as fake_worker:
            DistributedCompiledProgram.prepare(primary, extra_compiled=[extra])
        # prepare() delegates to DistributedWorker([primary, *extra_compiled], ...).
        assert fake_worker.call_args.args[0] == [primary, extra]

    def test_prepare_forwards_persistent_flag(self):
        from pypto.ir.distributed_compiled_program import DistributedCompiledProgram  # noqa: PLC0415

        primary = _fake_compiled([_param("a", [4])], [])
        with patch("pypto.runtime.distributed_runner.DistributedWorker") as fake_worker:
            DistributedCompiledProgram.prepare(primary, persistent=True)
        assert fake_worker.call_args.kwargs["persistent"] is True
        assert fake_worker.call_args.kwargs["reset_persistent_windows"] is None

    def test_prepare_forwards_startup_timeout(self):
        from pypto.ir.distributed_compiled_program import DistributedCompiledProgram  # noqa: PLC0415

        primary = _fake_compiled([_param("a", [4])], [])
        with patch("pypto.runtime.distributed_runner.DistributedWorker") as fake_worker:
            DistributedCompiledProgram.prepare(primary, startup_timeout_s=1800.0)
        assert fake_worker.call_args.kwargs["startup_timeout_s"] == 1800.0

    def test_empty_sequence_raises(self, patched_setup):
        with pytest.raises(ValueError, match="at least one compiled program"):
            DistributedWorker([])

    def test_rejects_mismatched_platform(self, patched_setup):
        prog_a = _fake_compiled([_param("a", [4])], [])
        prog_b = _fake_compiled([_param("b", [8])], [])
        prog_b.platform = "different_platform"
        with pytest.raises(ValueError, match="same platform"):
            DistributedWorker([prog_a, prog_b])

    def test_rejects_mismatched_device_ids(self, patched_setup):
        prog_a = _fake_compiled([_param("a", [4])], [])
        prog_b = _fake_compiled([_param("b", [8])], [])
        prog_b._distributed_config = DistributedConfig(device_ids=[0, 1])
        with pytest.raises(ValueError, match="same device_ids"):
            DistributedWorker([prog_a, prog_b])

    def test_rejects_mismatched_runtime(self, patched_setup):
        m = patched_setup
        m["assemble"].side_effect = [
            ({"chip_orch": object()}, "rt_name", False),
            ({"chip_orch": object()}, "other_rt", False),
        ]
        prog_a = _fake_compiled([_param("a", [4])], [])
        prog_b = _fake_compiled([_param("b", [8])], [])
        with pytest.raises(ValueError, match="same runtime"):
            DistributedWorker([prog_a, prog_b])


class TestAssembleChipCallables:
    """``_assemble_chip_callables`` is driven by the on-disk ``next_levels/``
    layout (no live IR), so it works for both freshly-compiled programs and ones
    reconstructed via ``from_dir`` (the L3 runtime_dir replay path, #1689)."""

    @staticmethod
    def _build(tmp_path, chip_names, *, stray=False) -> Any:
        nl = tmp_path / "next_levels"
        for name in chip_names:
            (nl / name).mkdir(parents=True, exist_ok=True)
            (nl / name / "kernel_config.py").write_text("KERNELS = []\nORCHESTRATION = {}\n")
        if stray:  # a dir without kernel_config.py must be skipped, not assembled
            (nl / "_not_a_chip").mkdir(parents=True, exist_ok=True)
        return SimpleNamespace(output_dir=tmp_path, platform="a2a3sim")

    @staticmethod
    def _stub_device_runner(monkeypatch, ca) -> None:
        """Inject a stub ``device_runner`` so ``_assemble_chip_callables`` can be
        exercised without importing the real module (which pulls in the simpler
        toolchain via ``kernel_compiler`` and is absent in the unit-test env)."""
        monkeypatch.setitem(
            sys.modules, "pypto.runtime.device_runner", SimpleNamespace(compile_and_assemble=ca)
        )

    def test_picks_up_chip_dirs_with_kernel_config(self, tmp_path, monkeypatch):
        compiled = self._build(tmp_path, ["chip_a", "chip_b"], stray=True)
        ca = MagicMock(return_value=(MagicMock(name="ChipCallable"), "tensormap_and_ringbuffer", {}))
        self._stub_device_runner(monkeypatch, ca)
        chip_callables, runtime_name, enable_sdma = _assemble_chip_callables(compiled)

        assert set(chip_callables) == {"chip_a", "chip_b"}  # stray dir skipped
        assert runtime_name == "tensormap_and_ringbuffer"
        assert enable_sdma is False
        called_dirs = {call.args[0] for call in ca.call_args_list}
        assert called_dirs == {tmp_path / "next_levels" / "chip_a", tmp_path / "next_levels" / "chip_b"}
        assert all(call.args[1] == "a2a3sim" for call in ca.call_args_list)

    def test_aggregates_enable_sdma_across_chip_configs(self, tmp_path, monkeypatch):
        compiled = self._build(tmp_path, ["chip_a", "chip_b"])
        ca = MagicMock(
            side_effect=[
                (MagicMock(name="ChipCallableA"), "tensormap_and_ringbuffer", {}),
                (
                    MagicMock(name="ChipCallableB"),
                    "tensormap_and_ringbuffer",
                    {"enable_sdma": True},
                ),
            ]
        )
        self._stub_device_runner(monkeypatch, ca)

        _, _, enable_sdma = _assemble_chip_callables(compiled)

        assert enable_sdma is True

    def test_raises_on_inconsistent_runtime(self, tmp_path, monkeypatch):
        compiled = self._build(tmp_path, ["chip_a", "chip_b"])
        ca = MagicMock(
            side_effect=[
                (MagicMock(name="ChipCallable"), "rt_one", {}),
                (MagicMock(name="ChipCallable"), "rt_two", {}),
            ]
        )
        self._stub_device_runner(monkeypatch, ca)
        with pytest.raises(RuntimeError, match="Inconsistent runtime"):
            _assemble_chip_callables(compiled)

    def test_raises_when_no_chip_dirs(self, tmp_path):
        # No next_levels/, so the helpful error must surface without importing the
        # device_runner toolchain (the import is deferred until a chip is found).
        compiled: Any = SimpleNamespace(output_dir=tmp_path, platform="a2a3sim")
        with pytest.raises(RuntimeError, match="No chip-level tasks found"):
            _assemble_chip_callables(compiled)


class _SpyDfxConfig:
    """Minimal stand-in for ``CallConfig`` exposing a mutable ``output_prefix``."""

    def __init__(self, output_prefix: str = "") -> None:
        self.output_prefix = output_prefix


class _RecordingOrch:
    """Records the ``output_prefix`` observed at each ``submit_next_level``.

    Captures the prefix *at submit time* (not after) so tests can prove
    ``_submit_chip`` applied the per-dispatch suffix before the task was queued.
    """

    def __init__(self, chip_count: int | None = None) -> None:
        self.calls: list[tuple[Any, int, str]] = []
        # ``_submit_chip`` reads/writes this per-card dispatch counter on the
        # orch; declare it so the attribute is known to the type checker.
        self._dfx_dispatch_idx: dict[str, int] = {}
        # ``_dispatch.orch_fn`` stamps the placement state on the real
        # Orchestrator; mirror it here. Leaving ``chip_count`` unset models a
        # caller that bypassed ``orch_fn``.
        if chip_count is not None:
            self._pypto_chip_count: int = chip_count
        self._pypto_commless_seq: int = 0

    def submit_next_level(self, callable_id: Any, task_args: Any, config: Any, *, worker: int) -> str:
        self.calls.append((callable_id, worker, config.output_prefix))
        return "submitted"


class TestSubmitChip:
    """``_submit_chip`` namespaces per-dispatch DFX ``output_prefix`` then restores it."""

    def test_suffixes_prefix_at_submit_and_restores(self):
        orch = _RecordingOrch()
        cfg = _SpyDfxConfig(output_prefix="/work/dfx_outputs")
        ret = _submit_chip(orch, "chip_a", "ta", cfg, 3)
        # Card + the card's 0th dispatch was visible to the runtime at submit
        # time...
        assert orch.calls == [("chip_a", 3, "/work/dfx_outputs/rank3/d0")]
        # ...and the shared config is restored afterward.
        assert cfg.output_prefix == "/work/dfx_outputs"
        assert ret == "submitted"

    def test_distinct_ranks_get_distinct_dirs(self):
        orch = _RecordingOrch()
        cfg = _SpyDfxConfig(output_prefix="/work/dfx_outputs")
        for r in (0, 1, 2):
            _submit_chip(orch, "chip", "ta", cfg, r)
        # Each card's first dispatch is ``d0``.
        assert [c[2] for c in orch.calls] == [
            "/work/dfx_outputs/rank0/d0",
            "/work/dfx_outputs/rank1/d0",
            "/work/dfx_outputs/rank2/d0",
        ]
        assert cfg.output_prefix == "/work/dfx_outputs"

    def test_multiple_dispatches_same_card_get_distinct_dirs(self):
        # The bug this fix targets: several dispatches to ONE card must not
        # share a dir (the runtime rewrites fixed-name artifacts per run, so a
        # shared dir means all-but-the-last are clobbered). Each gets ``d{k}``.
        orch = _RecordingOrch()
        cfg = _SpyDfxConfig(output_prefix="/work/dfx_outputs")
        _submit_chip(orch, "chip_a", "ta", cfg, 0)
        _submit_chip(orch, "chip_b", "ta", cfg, 0)  # different program, same card
        _submit_chip(orch, "chip_a", "ta", cfg, 0)  # repeat dispatch, same card
        assert [c[2] for c in orch.calls] == [
            "/work/dfx_outputs/rank0/d0",
            "/work/dfx_outputs/rank0/d1",
            "/work/dfx_outputs/rank0/d2",
        ]
        assert cfg.output_prefix == "/work/dfx_outputs"

    def test_counter_resets_when_orch_dispatch_idx_cleared(self):
        # ``orch_fn`` clears ``_dfx_dispatch_idx`` at the top of every run, so a
        # given card's dispatch numbering matches across the swimlane two-pass.
        orch = _RecordingOrch()
        cfg = _SpyDfxConfig(output_prefix="/work/dfx_outputs")
        _submit_chip(orch, "chip", "ta", cfg, 0)  # pass 1: d0
        orch._dfx_dispatch_idx = {}  # what orch_fn does between passes
        _submit_chip(orch, "chip", "ta", cfg, 0)  # pass 2: d0 again
        assert [c[2] for c in orch.calls] == [
            "/work/dfx_outputs/rank0/d0",
            "/work/dfx_outputs/rank0/d0",
        ]

    def test_dfx_off_forwards_unchanged(self):
        orch = _RecordingOrch()
        cfg = _SpyDfxConfig(output_prefix="")
        _submit_chip(orch, "chip", "ta", cfg, 5)
        assert orch.calls == [("chip", 5, "")]
        assert cfg.output_prefix == ""

    def test_commless_dispatches_round_robin_over_chips(self):
        # A comm-less dispatch (``worker=None``) names no chip, but simpler
        # #1436 requires an exact target, so consecutive ones are handed out
        # round-robin over the program's chips — a host_orch with one comm-less
        # dispatch per chip still spreads across them.
        orch = _RecordingOrch(chip_count=2)
        cfg = _SpyDfxConfig(output_prefix="/work/dfx_outputs")
        for _ in range(3):
            _submit_chip(orch, "chip", "ta", cfg, None)
        assert [c[1] for c in orch.calls] == [0, 1, 0]
        # Each resolved chip gets its own dispatch counter.
        assert [c[2] for c in orch.calls] == [
            "/work/dfx_outputs/rank0/d0",
            "/work/dfx_outputs/rank1/d0",
            "/work/dfx_outputs/rank0/d1",
        ]
        assert cfg.output_prefix == "/work/dfx_outputs"

    def test_commless_dispatch_without_chip_count_falls_back_to_chip_zero(self):
        # A caller that bypassed ``orch_fn`` leaves no chip count on ``orch``;
        # chip 0 always exists, so it is the safe fallback.
        orch = _RecordingOrch()
        cfg = _SpyDfxConfig(output_prefix="")
        _submit_chip(orch, "chip", "ta", cfg, None)
        _submit_chip(orch, "chip", "ta", cfg, None)
        assert [c[1] for c in orch.calls] == [0, 0]

    def test_pinned_dispatch_keeps_its_rank(self):
        # A ``device=``-pinned dispatch is never re-placed, even when comm-less
        # dispatches are round-robining alongside it.
        orch = _RecordingOrch(chip_count=2)
        cfg = _SpyDfxConfig(output_prefix="")
        _submit_chip(orch, "chip", "ta", cfg, 1)
        _submit_chip(orch, "chip", "ta", cfg, None)
        _submit_chip(orch, "chip", "ta", cfg, 1)
        assert [c[1] for c in orch.calls] == [1, 0, 1]

    def test_records_each_dispatchs_l2_program(self, tmp_path):
        # Issue #2169: ``rank{w}/d{k}`` says where a dispatch ran, not what it
        # ran, and ``func_id`` only means something within one L2 program. The
        # marker written here is what lets the offline post-pass label a
        # dispatch's records with its own program's kernel names.
        chip_cids = {"lm_head": object(), "mtp_decode_layer": object()}
        orch = _RecordingOrch()
        _reset_dfx_dispatch_state(orch, chip_cids)
        cfg = _SpyDfxConfig(output_prefix=str(tmp_path))

        # One card, two different programs -> d0 and d1 name their own program.
        _submit_chip(orch, chip_cids["mtp_decode_layer"], "ta", cfg, 0)
        _submit_chip(orch, chip_cids["lm_head"], "ta", cfg, 0)

        assert json.loads((tmp_path / "rank0" / "d0" / "dispatch_program.json").read_text()) == {
            "program": "mtp_decode_layer"
        }
        assert json.loads((tmp_path / "rank0" / "d1" / "dispatch_program.json").read_text()) == {
            "program": "lm_head"
        }

    def test_no_marker_when_chip_names_unstamped(self, tmp_path):
        # A caller that bypassed ``_reset_dfx_dispatch_state`` leaves no name
        # table on ``orch``; the dispatch must still go through (the marker is a
        # diagnostic, never a precondition).
        orch = _RecordingOrch()
        cfg = _SpyDfxConfig(output_prefix=str(tmp_path))

        _submit_chip(orch, "chip_a", "ta", cfg, 0)

        assert orch.calls == [("chip_a", 0, f"{tmp_path}/rank0/d0")]
        assert not (tmp_path / "rank0" / "d0" / "dispatch_program.json").exists()


def _write_dfx_dispatch_dirs(dfx: Path, *rels: str) -> None:
    """Lay down ``<dfx>/<rel>/chip_swimlane_records.json`` for each dispatch dir.

    Shared by the cleaner and collector tests below so the on-disk DFX layout
    they both assume is spelled out once.
    """
    for rel in rels:
        (dfx / rel).mkdir(parents=True)
        (dfx / rel / "chip_swimlane_records.json").write_text("{}", encoding="utf-8")


def _write_chip_program(output_dir: Path, program: str, *kernel_names: str) -> None:
    """Lay down ``next_levels/<program>/kernel_config.py`` naming *kernel_names*.

    Every L2 program numbers its kernels from ``func_id`` 0 — that shared
    numbering is exactly what makes a name map merged across programs wrong
    (issue #2169), so each program written here starts at 0 on purpose.
    """
    chip_dir = output_dir / "next_levels" / program
    chip_dir.mkdir(parents=True)
    kernels = [{"func_id": i, "name": name} for i, name in enumerate(kernel_names)]
    (chip_dir / "kernel_config.py").write_text(f"KERNELS = {kernels!r}\n", encoding="utf-8")


def _mark_dispatch_program(disp_dir: Path, program: str) -> None:
    """Stamp the marker ``_submit_chip`` writes for a dispatch of *program*."""
    (disp_dir / "dispatch_program.json").write_text(json.dumps({"program": program}), encoding="utf-8")


@pytest.fixture
def fake_swimlane_converter(monkeypatch):
    """Register a fake ``simpler_setup.tools.swimlane_converter``.

    The real module ships with the optional ``simpler`` runtime package, which is
    not installed in CI. The fake reproduces the one function pypto calls,
    ``load_kernel_config``, with the real contract: import the ``kernel_config.py``
    and return its ``func_id`` (as ``str``) -> ``name`` mapping. Tests therefore
    still exercise the genuine on-disk layout.
    """
    pkg = ModuleType("simpler_setup")
    tools = ModuleType("simpler_setup.tools")
    mod = ModuleType("simpler_setup.tools.swimlane_converter")

    def load_kernel_config(config_path: str) -> dict[str, str]:
        spec = importlib.util.spec_from_file_location("kernel_config", config_path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return {str(k["func_id"]): k["name"] for k in module.KERNELS}

    mod.load_kernel_config = load_kernel_config  # pyright: ignore[reportAttributeAccessIssue]
    tools.swimlane_converter = mod  # pyright: ignore[reportAttributeAccessIssue]
    pkg.tools = tools  # pyright: ignore[reportAttributeAccessIssue]
    monkeypatch.setitem(sys.modules, "simpler_setup", pkg)
    monkeypatch.setitem(sys.modules, "simpler_setup.tools", tools)
    monkeypatch.setitem(sys.modules, "simpler_setup.tools.swimlane_converter", mod)
    return mod


class TestClearDfxDispatchDirs:
    """``_clear_dfx_dispatch_dirs`` drops stale ``rank*/d{k}`` dirs before a run."""

    def test_removes_only_dispatch_dirs(self, tmp_path):
        # A prior run left rank0/{d0,d1,d2} and rank1/d0; the current run will
        # only write d0, so the stale d1/d2 must be cleared. A sibling non-d{k}
        # dir (e.g. a future diagnostic) is preserved.
        dfx = tmp_path / "dfx_outputs"
        _write_dfx_dispatch_dirs(dfx, "rank0/d0", "rank0/d1", "rank0/d2", "rank1/d0", "rank0/keepme")

        _clear_dfx_dispatch_dirs(dfx)

        # All d{k} dirs gone, on every card...
        assert not (dfx / "rank0" / "d0").exists()
        assert not (dfx / "rank0" / "d1").exists()
        assert not (dfx / "rank0" / "d2").exists()
        assert not (dfx / "rank1" / "d0").exists()
        # ...but the non-dispatch dir and the rank dirs themselves remain.
        assert (dfx / "rank0" / "keepme").is_dir()
        assert (dfx / "rank0").is_dir()

    def test_missing_base_is_noop(self, tmp_path):
        # No dfx_outputs yet (first dispatch) -> nothing to clear, no error.
        _clear_dfx_dispatch_dirs(tmp_path / "dfx_outputs")


class TestPrepareReusedDepGenDirs:
    """The prepared two-dispatch path preserves graphs and drops stale timing."""

    def test_preserves_deps_and_removes_timing_outputs(self, tmp_path):
        dfx = tmp_path / "dfx_outputs"
        dispatches = [dfx / "rank0" / "d0", dfx / "rank1" / "d0"]
        for disp in dispatches:
            disp.mkdir(parents=True)
            (disp / "deps.json").write_text("{}", encoding="utf-8")
            (disp / "name_map.json").write_text("{}", encoding="utf-8")
            (disp / _CHIP_SWIMLANE_RECORDS_NAME).write_text("old", encoding="utf-8")
            (disp / "merged_swimlane_old.json").write_text("old", encoding="utf-8")
            (disp / "critical_path_report.md").write_text("old", encoding="utf-8")

        found = _prepare_reused_dep_gen_dirs(dfx)

        assert found == dispatches
        for disp in dispatches:
            assert (disp / "deps.json").is_file()
            assert (disp / "name_map.json").is_file()
            assert not (disp / _CHIP_SWIMLANE_RECORDS_NAME).exists()
            assert not (disp / "merged_swimlane_old.json").exists()
            assert not (disp / "critical_path_report.md").exists()

    def test_requires_an_existing_dep_capture(self, tmp_path):
        with pytest.raises(RuntimeError, match="run the same dispatch once"):
            _prepare_reused_dep_gen_dirs(tmp_path / "dfx_outputs")

    def test_rejects_partial_dep_capture(self, tmp_path):
        dfx = tmp_path / "dfx_outputs"
        (dfx / "rank0" / "d0").mkdir(parents=True)
        with pytest.raises(RuntimeError, match="without deps.json"):
            _prepare_reused_dep_gen_dirs(dfx)


class TestCollectL3Swimlane:
    """``_collect_l3_swimlane`` converts every ``rank*/d{k}`` dispatch's records."""

    @staticmethod
    def _spy_generate_swimlane(monkeypatch) -> list[SimpleNamespace]:
        """Record each converter invocation (dispatch dir, ``-k`` dir, name map)."""
        import pypto.runtime.runner as _runner  # noqa: PLC0415

        seen: list[SimpleNamespace] = []

        def _fake(work_dir, out_dir, records, func_names=None):  # noqa: ANN001
            seen.append(
                SimpleNamespace(work_dir=work_dir, out_dir=out_dir, records=records, func_names=func_names)
            )

        monkeypatch.setattr(_runner, "_generate_swimlane", _fake)
        return seen

    def test_collects_every_cards_dispatch_dirs(self, tmp_path, monkeypatch):
        # Globbing ``rank*`` (rather than iterating a rank count) is what lets a
        # run whose cards are not known up front — e.g. a comm-less program
        # whose dispatches were placed round-robin — get converted at all.
        seen = self._spy_generate_swimlane(monkeypatch)
        dfx = tmp_path / "dfx_outputs"
        # ``rank0/keepme`` carries a records file like a real dispatch dir, so
        # only the ``d[0-9]*`` filter can exclude it — that makes the assertion
        # below a genuine discriminator for the glob rather than for the
        # ``records.exists()`` guard.
        _write_dfx_dispatch_dirs(dfx, "rank0/d0", "rank0/d1", "rank1/d0", "rank0/keepme")
        # A dispatch dir with no records (DFX wrote nothing) is skipped.
        (dfx / "rank1" / "d1").mkdir(parents=True)

        _collect_l3_swimlane(tmp_path, "a2a3")

        assert sorted(str(s.out_dir.relative_to(dfx)) for s in seen) == [
            "rank0/d0",
            "rank0/d1",
            "rank1/d0",
        ]

    def test_simulator_platform_skips_conversion(self, tmp_path, monkeypatch):
        # Onboard-only: the simulator emits records but not the task metadata
        # the converter joins against, so the raw records are kept as-is.
        seen = self._spy_generate_swimlane(monkeypatch)
        _write_dfx_dispatch_dirs(tmp_path / "dfx_outputs", "rank0/d0")

        _collect_l3_swimlane(tmp_path, "a2a3sim")

        assert seen == []

    def test_missing_dfx_base_is_noop(self, tmp_path, monkeypatch):
        # DFX was off (or nothing was written) -> nothing to convert, no error.
        seen = self._spy_generate_swimlane(monkeypatch)

        _collect_l3_swimlane(tmp_path, "a2a3")

        assert seen == []

    def test_name_map_is_scoped_to_each_dispatchs_own_program(
        self, tmp_path, monkeypatch, fake_swimlane_converter
    ):
        # Regression for issue #2169. Two L2 programs both number their kernels
        # from func_id 0, so a name map merged across them relabels one
        # program's tasks with the other's names — silently and plausibly
        # (``lm_head_dispatch_wait``, a cross-card spin-wait, printed as
        # ``mtp_projection_norm``). Each dispatch must get its own program's map.
        seen = self._spy_generate_swimlane(monkeypatch)
        _write_chip_program(tmp_path, "lm_head", "lm_head_dispatch_push", "lm_head_dispatch_wait")
        _write_chip_program(tmp_path, "mtp_decode_layer", "mtp_projection_rms", "mtp_projection_norm")
        dfx = tmp_path / "dfx_outputs"
        _write_dfx_dispatch_dirs(dfx, "rank0/d0", "rank0/d1")
        _mark_dispatch_program(dfx / "rank0" / "d0", "mtp_decode_layer")
        _mark_dispatch_program(dfx / "rank0" / "d1", "lm_head")

        _collect_l3_swimlane(tmp_path, "a2a3")

        by_dir = {s.out_dir.name: s for s in seen}
        assert set(by_dir) == {"d0", "d1"}
        # Each dispatch's name map holds its own program's kernels...
        for disp, program, names in (
            ("d0", "mtp_decode_layer", ["mtp_projection_rms", "mtp_projection_norm"]),
            ("d1", "lm_head", ["lm_head_dispatch_push", "lm_head_dispatch_wait"]),
        ):
            name_map = json.loads((dfx / "rank0" / disp / "name_map.json").read_text())
            assert name_map["callable_id_to_name"] == {"0": names[0], "1": names[1]}
            assert by_dir[disp].func_names == dfx / "rank0" / disp / "name_map.json"
            # ...and the converter's ``-k`` fallback names the same program, so
            # the two label sources can never disagree.
            assert by_dir[disp].work_dir == tmp_path / "next_levels" / program

    def test_sole_program_names_an_unmarked_dispatch(self, tmp_path, monkeypatch, fake_swimlane_converter):
        # Only one L2 program in the build: there is no namespace to confuse, so
        # a dispatch without a marker (e.g. artifacts from an older run) is still
        # labelled rather than degraded to anonymous tasks.
        seen = self._spy_generate_swimlane(monkeypatch)
        _write_chip_program(tmp_path, "only_chip", "rms", "matmul")
        dfx = tmp_path / "dfx_outputs"
        _write_dfx_dispatch_dirs(dfx, "rank0/d0")

        _collect_l3_swimlane(tmp_path, "a2a3")

        name_map = json.loads((dfx / "rank0" / "d0" / "name_map.json").read_text())
        assert name_map["callable_id_to_name"] == {"0": "rms", "1": "matmul"}
        assert seen[0].work_dir == tmp_path / "next_levels" / "only_chip"

    def test_unresolvable_dispatch_converts_without_names(
        self, tmp_path, monkeypatch, fake_swimlane_converter, capsys
    ):
        # Several programs and no marker: the program is genuinely unknown. The
        # records still convert, but with anonymous labels — a wrong name is
        # worse than no name, since it reads as a real measurement.
        seen = self._spy_generate_swimlane(monkeypatch)
        _write_chip_program(tmp_path, "lm_head", "lm_head_dispatch_push")
        _write_chip_program(tmp_path, "mtp_decode_layer", "mtp_projection_rms")
        dfx = tmp_path / "dfx_outputs"
        _write_dfx_dispatch_dirs(dfx, "rank0/d0")

        _collect_l3_swimlane(tmp_path, "a2a3")

        assert len(seen) == 1
        assert seen[0].func_names is None
        assert not (dfx / "rank0" / "d0" / "name_map.json").exists()
        # ``work_dir`` holds no kernel_config.py, so no other program's table is
        # handed to the converter's ``-k`` fallback either.
        assert not (seen[0].work_dir / "kernel_config.py").exists()
        assert "No L2 program recorded for rank0/d0" in capsys.readouterr().out

    def test_unresolvable_dispatch_drops_a_stale_name_map(
        self, tmp_path, monkeypatch, fake_swimlane_converter
    ):
        # With no map passed, the converter auto-discovers a sibling
        # ``name_map*.json`` — so a map left by an earlier run would quietly
        # resurrect the mislabelling this fix removes.
        self._spy_generate_swimlane(monkeypatch)
        _write_chip_program(tmp_path, "lm_head", "lm_head_dispatch_push")
        _write_chip_program(tmp_path, "mtp_decode_layer", "mtp_projection_rms")
        dfx = tmp_path / "dfx_outputs"
        _write_dfx_dispatch_dirs(dfx, "rank0/d0")
        stale = dfx / "rank0" / "d0" / "name_map.json"
        stale.write_text('{"callable_id_to_name": {"0": "mtp_projection_rms"}}', encoding="utf-8")

        _collect_l3_swimlane(tmp_path, "a2a3")

        assert not stale.exists()

    def test_resolved_program_without_a_table_drops_a_stale_name_map(
        self, tmp_path, monkeypatch, fake_swimlane_converter
    ):
        # The program resolves, but its ``kernel_config.py`` names no kernels, so
        # no map is written for this run. A previous run's map must not survive to
        # be picked up in its place — this dispatch renders anonymously.
        self._spy_generate_swimlane(monkeypatch)
        _write_chip_program(tmp_path, "lm_head")  # KERNELS = []
        dfx = tmp_path / "dfx_outputs"
        _write_dfx_dispatch_dirs(dfx, "rank0/d0")
        _mark_dispatch_program(dfx / "rank0" / "d0", "lm_head")
        stale = dfx / "rank0" / "d0" / "name_map.json"
        stale.write_text('{"callable_id_to_name": {"0": "mtp_projection_rms"}}', encoding="utf-8")

        _collect_l3_swimlane(tmp_path, "a2a3")

        assert not stale.exists()

    def test_stray_subdir_does_not_hide_the_sole_program(
        self, tmp_path, monkeypatch, fake_swimlane_converter
    ):
        # ``next_levels/`` may hold a subdir that is not an L2 program (no
        # kernel_config.py). Counting it would make the build look multi-program
        # and needlessly drop the unmarked dispatch to anonymous labels.
        seen = self._spy_generate_swimlane(monkeypatch)
        _write_chip_program(tmp_path, "only_chip", "rms")
        (tmp_path / "next_levels" / "scratch").mkdir()
        dfx = tmp_path / "dfx_outputs"
        _write_dfx_dispatch_dirs(dfx, "rank0/d0")

        _collect_l3_swimlane(tmp_path, "a2a3")

        assert seen[0].work_dir == tmp_path / "next_levels" / "only_chip"
        assert json.loads((dfx / "rank0" / "d0" / "name_map.json").read_text())["callable_id_to_name"] == {
            "0": "rms"
        }


class _BoolStrictCallConfig:
    """Fake ``CallConfig`` whose ``enable_dep_gen`` mirrors simpler's pybind setter.

    The real ``CallConfig.enable_dep_gen`` pybind overload accepts only ``bool``
    and raises ``TypeError`` on an ``int`` — exactly the crash issue #1952
    reproduces when the int ``enable_chip_swimlane`` collection level (0-4) leaks through
    the ``and``/``or`` chain unwrapped. ``bool`` is a subclass of ``int``, so
    ``isinstance(value, bool)`` matches the pybind behavior (rejects ``1``/``0``).
    """

    def __init__(self) -> None:
        self.aicpu_thread_num = 0
        self.enable_dump_args = 0
        self.enable_pmu = 0
        self.enable_scope_stats = False
        self.enable_chip_swimlane: Any = 0
        self.output_prefix = ""
        self.runtime_env = SimpleNamespace(ring_task_window=0, ring_heap=0, ring_dep_pool=0)
        self._enable_dep_gen = False

    @property
    def enable_dep_gen(self) -> bool:
        return self._enable_dep_gen

    @enable_dep_gen.setter
    def enable_dep_gen(self, value: object) -> None:
        if not isinstance(value, bool):
            raise TypeError(
                f"incompatible function arguments: enable_dep_gen expects bool, got {type(value).__name__}"
            )
        self._enable_dep_gen = value


@pytest.fixture
def fake_simpler_task_interface(monkeypatch):
    """Register a fake ``simpler.task_interface`` exposing a bool-strict ``CallConfig``.

    Lets ``_make_call_config`` run without the real (optional) ``simpler`` runtime
    package while still enforcing the pybind ``bool``-only contract on
    ``enable_dep_gen``.
    """
    pkg = ModuleType("simpler")
    mod = ModuleType("simpler.task_interface")
    mod.CallConfig = _BoolStrictCallConfig  # pyright: ignore[reportAttributeAccessIssue]
    pkg.task_interface = mod  # pyright: ignore[reportAttributeAccessIssue]
    monkeypatch.setitem(sys.modules, "simpler", pkg)
    monkeypatch.setitem(sys.modules, "simpler.task_interface", mod)
    return mod


class TestMakeCallConfigDepGenType:
    """``_make_call_config`` must assign a ``bool`` to ``enable_dep_gen``.

    Regression for issue #1952: ``enable_chip_swimlane`` is a collection level
    (0-4), so the
    ``dfx.enable_dep_gen or (co_enable_swimlane_dep_gen and dfx.enable_chip_swimlane)``
    chain can yield an int, which the ``bool``-only pybind setter rejects.
    """

    def test_int_swimlane_flag_yields_bool_dep_gen(self, tmp_path, fake_simpler_task_interface):
        # ``--enable-chip-swimlane 1`` reaches RunConfig as the int ``1``; the
        # co-enable path must still hand ``enable_dep_gen`` a genuine ``bool``.
        run_config = RunConfig(enable_chip_swimlane=1)
        cfg = _make_call_config(DistributedConfig(), run_config, dfx_base=tmp_path / "dfx")
        assert cfg.enable_dep_gen is True
        assert cfg.enable_chip_swimlane == 1

    def test_int_zero_swimlane_yields_bool_false_dep_gen(self, tmp_path, fake_simpler_task_interface):
        # Another DFX flag opens the block while swimlane is the int ``0``; the
        # ``and``/``or`` chain would otherwise assign int ``0`` and still crash.
        run_config = RunConfig(enable_dump_args=1, enable_chip_swimlane=0)
        cfg = _make_call_config(DistributedConfig(), run_config, dfx_base=tmp_path / "dfx")
        assert cfg.enable_dep_gen is False

    def test_clean_timing_suppresses_implicit_dep_gen(self, tmp_path, fake_simpler_task_interface):
        run_config = RunConfig(enable_chip_swimlane=1)
        cfg = _make_call_config(
            DistributedConfig(),
            run_config,
            dfx_base=tmp_path / "dfx",
            co_enable_swimlane_dep_gen=False,
        )
        assert cfg.enable_dep_gen is False

    def test_explicit_dep_gen_still_wins_when_co_enable_is_off(self, tmp_path, fake_simpler_task_interface):
        run_config = RunConfig(
            enable_chip_swimlane=1,  # AICore-timing level, not just "on"
            enable_dep_gen=True,
        )
        cfg = _make_call_config(
            DistributedConfig(),
            run_config,
            dfx_base=tmp_path / "dfx",
            co_enable_swimlane_dep_gen=False,
        )
        assert cfg.enable_dep_gen is True


class _PersistentRunResources:
    """Small model of Simpler's per-run domain journal."""

    def __init__(self) -> None:
        self.live_domains: dict[str, Any] = {}
        self.pending_release_domains: list[Any] = []
        self.retired = False
        self.domain_lock = threading.Lock()
        self.requires_ordered_cleanup = False


class _PersistentDomainHandle:
    def __init__(
        self,
        name: str,
        workers: list[int],
        window_size: int,
        allocation_index: int,
        worker: Any,
        owner: _PersistentRunResources,
        buffers: list[Any],
    ) -> None:
        self.name = name
        self.workers = tuple(workers)
        self.contexts = {}
        for worker_id in workers:
            local_window_base = 0x10000000 + allocation_index * 0x100000 + worker_id * 0x10000
            offset = 0
            named_buffers = {}
            for spec in buffers:
                named_buffers[spec.name] = _FakeBuffer(
                    local_window_base + offset,
                    int(spec.nbytes),
                    owner_worker_id=worker_id,
                )
                offset += int(spec.nbytes)
            self.contexts[worker_id] = SimpleNamespace(
                local_window_base=local_window_base,
                actual_window_size=window_size,
                buffers=named_buffers,
            )
        self.worker = worker
        self.owner = owner
        self.release_count = 0
        self.close_sweep_count = 0
        self.backend_release_count = 0
        self.released = False
        self.freed = False
        self.release_error: BaseException | None = None
        self.free_on_release = True

    def __getitem__(self, worker_id: int):
        return self.contexts[worker_id]

    def release(self) -> None:
        if self.released:
            return
        self.released = True
        self.release_count += 1
        with self.owner.domain_lock:
            if not self.owner.retired:
                self.owner.pending_release_domains.append(self)
                self.owner.live_domains.pop(self.name, None)
                if self.worker._live_domains.get(self.name) is self:
                    self.worker._live_domains.pop(self.name)
                return
        self._free_from_global_registry(from_close=False)

    def close_sweep(self) -> None:
        """Model Worker.close()'s direct global live-domain sweep."""
        self.close_sweep_count += 1
        self.released = True
        self._free_from_global_registry(from_close=True)

    def _free_from_global_registry(self, *, from_close: bool) -> None:
        if self.freed:
            return
        if not from_close and not self.free_on_release:
            return
        if self.backend_release_count == 0:
            self.backend_release_count = 1
        if self.release_error is not None:
            raise self.release_error
        self.freed = True
        if self.worker._live_domains.get(self.name) is self:
            self.worker._live_domains.pop(self.name)


class _PersistentOrch:
    def __init__(self, worker: Any) -> None:
        self.worker = worker
        self.allocate_calls: list[dict[str, Any]] = []
        self.copy_calls: list[tuple[_FakeBuffer, _FakeBuffer, bytes]] = []
        self.handles: list[_PersistentDomainHandle] = []
        self.resources: list[_PersistentRunResources] = []
        self.worker.close.side_effect = self._close_live_domains

    def _begin_run(self) -> _PersistentRunResources:
        resources = _PersistentRunResources()
        self.resources.append(resources)
        self.worker._building_run_resources = resources
        return resources

    def _retire_run(self, resources: _PersistentRunResources) -> None:
        assert self.worker._building_run_resources is resources
        self.worker._building_run_resources = None
        with resources.domain_lock:
            resources.retired = True

    def run(self, fn, *, before_retire=None, on_error=None):
        resources = self._begin_run()
        try:
            result = fn(self, None, None)
        except BaseException:
            if on_error is not None:
                on_error()
            raise
        else:
            if before_retire is not None:
                before_retire()
            return result
        finally:
            self._retire_run(resources)

    def run_with_abandoned_finalization(self, fn):
        """Leave the owner unretired, as an ambiguous finalizer boundary does."""
        resources = self._begin_run()
        try:
            fn(self, None, None)
        finally:
            assert self.worker._building_run_resources is resources
            self.worker._building_run_resources = None
        raise RuntimeError("run finalization abandoned")

    def allocate_domain(self, **kwargs):
        self.allocate_calls.append(kwargs)
        resources = self.worker._building_run_resources
        assert isinstance(resources, _PersistentRunResources)
        handle = _PersistentDomainHandle(
            kwargs["name"],
            list(kwargs["workers"]),
            int(kwargs["window_size"]),
            len(self.handles),
            self.worker,
            resources,
            list(kwargs["buffers"]),
        )
        self.handles.append(handle)
        self.worker._live_domains[handle.name] = handle
        resources.live_domains[handle.name] = handle
        resources.requires_ordered_cleanup = True
        return handle

    def copy_to(self, dst: _FakeBuffer, src: _FakeBuffer) -> None:
        payload = ctypes.string_at(int(src.base), int(src.nbytes))
        self.copy_calls.append((dst, src, payload))

    def _close_live_domains(self) -> None:
        first_error: BaseException | None = None
        for handle in list(self.worker._live_domains.values())[::-1]:
            try:
                handle.close_sweep()
            except BaseException as exc:  # noqa: BLE001 - model best-effort close
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error


def _persistent_entry(
    window_size: int,
    seen_handles: list[Any],
    *,
    buffer_nbytes: tuple[int, ...] | None = None,
    buffer_names: tuple[str, ...] | None = None,
    signature_tokens: list[Any] | None = None,
):
    sizes = (window_size,) if buffer_nbytes is None else buffer_nbytes
    names = tuple(f"buffer_{index}" for index in range(len(sizes))) if buffer_names is None else buffer_names
    assert len(names) == len(sizes)

    def entry(
        orch,
        _args,
        config,
        *,
        tensors,
        callables,
        sub_ids,
        _keep,
        world_size,
        _domain_provider=None,
    ):
        del orch, _args, config, tensors, callables, sub_ids, _keep
        assert _domain_provider is not None
        if signature_tokens is not None:
            signature_tokens.append(getattr(_domain_provider, "_pypto_task_args_signature_token", None))
        with _domain_provider(
            name="comm_d0",
            workers=[*range(world_size)],
            window_size=window_size,
            buffers=[
                SimpleNamespace(name=name, dtype="opaque", count=size, nbytes=size)
                for name, size in zip(names, sizes, strict=True)
            ],
        ) as domain:
            seen_handles.append(domain)

    return entry


_K8_LEGACY_CONTROL_NAMES = (
    "dense_attn_signal_stack_buf",
    "dense_mlp_signal_stack_buf",
    "moe_attn_signal_stack_buf",
    "moe_meta_arrived_stack_buf",
    "moe_data_arrived_stack_buf",
    "moe_sh_signal_stack_buf",
    "moe_combine_arrived_stack_buf",
)
_K8_LEGACY_DATA_NAMES = (
    "dense_attn_tmp_stack_buf",
    "dense_mlp_tmp_stack_buf",
    "moe_attn_tmp_stack_buf",
    "moe_recv_meta_stack_buf",
    "moe_recv_x_stack_buf",
    "moe_recv_aux_stack_buf",
    "moe_recv_route_stack_buf",
    "moe_sh_tmp_stack_buf",
    "moe_routed_y_buf_stack_buf",
)
_K8_LEGACY_CONTROL_NBYTES = (1536, 1536, 21504, 512, 512, 21504, 512)
_K8_LEGACY_DATA_NBYTES = (393216, 393216, 5505024, 1280, 18874368, 147456, 147456, 5505024, 1048576)
_K8_LOCAL_OWNER_CONTROL_NAMES = (
    "dense_attn_signal_stack_buf",
    "dense_mlp_signal_stack_buf",
    "moe_attn_signal_stack_buf",
    "moe_sh_signal_stack_buf",
)
_K8_LOCAL_OWNER_DATA_NAMES = (
    "dense_attn_tmp_stack_buf",
    "dense_mlp_tmp_stack_buf",
    "moe_attn_tmp_stack_buf",
    "moe_sh_tmp_stack_buf",
)
_K8_LOCAL_OWNER_CONTROL_NBYTES = (1536, 1536, 21504, 21504)
_K8_LOCAL_OWNER_DATA_NBYTES = (393216, 393216, 5505024, 5505024)


def _device_memset_persistent_runtime(
    patched_setup: dict[str, Any],
    *,
    buffer_names: tuple[str, ...],
    buffer_nbytes: tuple[int, ...],
) -> tuple[_PersistentOrch, DistributedWorker, DeviceTensor]:
    """Construct a reused persistent domain backed by device memset."""
    patched_setup["worker"]._live_domains = {}
    patched_setup["worker"].device_memset_available = True
    orch = _PersistentOrch(patched_setup["worker"])
    patched_setup["worker"].submit.side_effect = lambda fn: (
        orch.run(fn),
        _ImmediateNativeHandle(),
    )[1]
    patched_setup["load_entry"].return_value = (
        _persistent_entry(
            sum(buffer_nbytes),
            [],
            buffer_nbytes=buffer_nbytes,
            buffer_names=buffer_names,
        ),
        None,
    )
    compiled = _fake_compiled([_param("a", [16, 16])], [])
    compiled._distributed_config = DistributedConfig(device_ids=[0, 1])
    rt = DistributedWorker(compiled, persistent=True)
    return orch, rt, _resident(rt, (16, 16))


class TestPersistentDistributedWorker:
    def test_window_reset_requires_persistent_mode(self):
        compiled = _fake_compiled([_param("a", [16, 16])], [])
        with pytest.raises(ValueError, match="requires persistent=True"):
            DistributedWorker(compiled, reset_persistent_windows=True)

    def test_rejects_artifact_without_domain_provider_hook(self, patched_setup):
        compiled = _fake_compiled([_param("a", [16, 16])], [])
        with pytest.raises(ValueError, match="requires regenerated host orchestration"):
            DistributedWorker(compiled, persistent=True)

    @pytest.mark.parametrize("attribute", ["_live_domains", "_building_run_resources"])
    def test_rejects_missing_persistent_runtime_hooks_before_init(self, patched_setup, attribute):
        m = patched_setup
        if attribute == "_live_domains":
            m["worker"]._live_domains = None
        else:
            del m["worker"]._building_run_resources
        m["load_entry"].return_value = (_persistent_entry(64, []), None)
        compiled = _fake_compiled([_param("a", [16, 16])], [])

        with pytest.raises(RuntimeError, match=attribute):
            DistributedWorker(compiled, persistent=True)

        m["worker"].init.assert_not_called()
        m["worker"].close.assert_called_once_with()

    def test_request_run_fences_reuse_and_zero_domain_by_default(self, patched_setup):
        m = patched_setup
        m["worker"]._live_domains = {}
        created_zero_buffers: list[_FakeBuffer] = []
        released_zero_buffers: list[_FakeBuffer] = []

        def create_dirty_host_buffer(nbytes: int) -> _FakeBuffer:
            buffer = _FakeBuffer(0, nbytes, host=True)
            ctypes.memset(int(buffer.base), 0xA5, nbytes)
            created_zero_buffers.append(buffer)
            return buffer

        def release_host_buffer(buffer: _FakeBuffer) -> None:
            # Simpler holds a non-reentrant submit lock during the callback, so
            # release is only safe after its run journal has been retired.
            assert m["worker"]._building_run_resources is None
            released_zero_buffers.append(buffer)

        m["worker"].create_buffer.side_effect = create_dirty_host_buffer
        m["worker"].release_buffer.side_effect = release_host_buffer
        orch = _PersistentOrch(m["worker"])
        submit_threads: list[int] = []

        def worker_submit(fn):
            submit_threads.append(threading.get_ident())
            orch.run(fn)
            return _ImmediateNativeHandle()

        m["worker"].submit.side_effect = worker_submit
        seen_handles: list[Any] = []
        window_size = (1 << 20) + 17
        m["load_entry"].return_value = (
            _persistent_entry(window_size, seen_handles, buffer_nbytes=(1 << 20, 17)),
            None,
        )
        compiled = _fake_compiled([_param("a", [16, 16])], [])
        compiled._distributed_config = DistributedConfig(device_ids=[0, 1])

        rt = DistributedWorker(compiled, persistent=True)
        arg = _resident(rt, (16, 16))
        rt(arg)
        handle = orch.handles[0]
        owner = orch.resources[0]
        assert owner.live_domains == {}
        assert owner.retired
        assert owner.requires_ordered_cleanup
        assert m["worker"]._live_domains == {"p0:comm_d0": handle}
        assert not handle.released
        assert not handle.freed
        rt(arg)
        rt.close()

        assert m["worker"].submit.call_count == 2
        assert submit_threads == [threading.get_ident(), threading.get_ident()]
        assert [call["name"] for call in orch.allocate_calls] == ["p0:comm_d0"]
        assert len(seen_handles) == 2
        assert seen_handles[0] is seen_handles[1]
        # The first request receives the freshly-zeroed allocation. The second
        # resets each of its two named buffers on both workers through Buffer
        # handles, reusing one runtime-owned host buffer per distinct size.
        assert [(dst.owner_worker_id, dst.nbytes) for dst, _src, _payload in orch.copy_calls] == [
            (0, 1 << 20),
            (1, 1 << 20),
            (0, 17),
            (1, 17),
        ]
        assert all(payload == bytes(dst.nbytes) for dst, _src, payload in orch.copy_calls)
        assert [src for _dst, src, _payload in orch.copy_calls] == [
            created_zero_buffers[0],
            created_zero_buffers[0],
            created_zero_buffers[1],
            created_zero_buffers[1],
        ]
        assert released_zero_buffers == created_zero_buffers
        assert [call.args[0] for call in m["worker"].release_buffer.call_args_list] == created_zero_buffers
        # A retained domain survives both request run-fences and is released
        # once when the persistent worker closes.
        assert handle.release_count == 1
        assert handle.close_sweep_count == 0
        assert handle.backend_release_count == 1
        assert handle.freed
        assert m["worker"]._live_domains == {}

    def test_persistent_provider_reuses_and_invalidates_prepared_signature_token(self, patched_setup):
        m = patched_setup
        m["worker"]._live_domains = {}
        orch = _PersistentOrch(m["worker"])
        m["worker"].submit.side_effect = lambda fn: (
            orch.run(fn),
            _ImmediateNativeHandle(),
        )[1]
        signature_tokens: list[Any] = []
        m["load_entry"].return_value = (
            _persistent_entry(64, [], signature_tokens=signature_tokens),
            None,
        )
        compiled = _fake_compiled([_param("a", [16, 16])], [])
        rt = DistributedWorker(
            compiled,
            persistent=True,
            reset_persistent_windows=False,
        )
        arg = _resident(rt, (16, 16))

        rt(arg)
        rt(arg)
        assert signature_tokens[0] is not None
        assert signature_tokens[1] == signature_tokens[0]

        arg.buffer.identity.generation += 1
        rt(arg)
        assert signature_tokens[2] != signature_tokens[1]
        rt.close()

    def test_persistent_free_waits_until_signature_token_is_consumed(self, patched_setup):
        m = patched_setup
        m["worker"]._live_domains = {}
        orch = _PersistentOrch(m["worker"])
        submit_entered = threading.Event()
        continue_submit = threading.Event()
        signature_tokens: list[Any] = []

        def worker_submit(fn):
            submit_entered.set()
            assert continue_submit.wait(timeout=5)
            orch.run(fn)
            return _ImmediateNativeHandle()

        m["worker"].submit.side_effect = worker_submit
        m["load_entry"].return_value = (
            _persistent_entry(64, [], signature_tokens=signature_tokens),
            None,
        )
        compiled = _fake_compiled([_param("a", [16, 16])], [])
        rt = DistributedWorker(
            compiled,
            persistent=True,
            reset_persistent_windows=False,
        )
        observed_lock = _ObservedLock()
        rt._dispatch_submit_mu = observed_lock
        arg = _resident(rt, (16, 16))

        def free_arg():
            rt.free_tensor(arg)

        def free_buffer(buffer):
            assert signature_tokens
            buffer.closed = True

        m["worker"].free.side_effect = free_buffer
        with ThreadPoolExecutor(max_workers=2) as pool:
            submit_future = pool.submit(rt, arg)
            assert submit_entered.wait(timeout=5)
            free_future = pool.submit(free_arg)
            assert observed_lock.blocked.wait(timeout=5)
            m["worker"].free.assert_not_called()
            continue_submit.set()
            submit_future.result(timeout=5)
            free_future.result(timeout=5)

        assert signature_tokens[0] is not None
        m["worker"].free.assert_called_once_with(arg.buffer)
        rt.close()

    def test_persistent_signature_token_advances_after_unsupported_args(self, patched_setup):
        m = patched_setup
        m["worker"]._live_domains = {}
        orch = _PersistentOrch(m["worker"])
        m["worker"].submit.side_effect = lambda fn: (
            orch.run(fn),
            _ImmediateNativeHandle(),
        )[1]
        signature_tokens: list[Any] = []
        m["load_entry"].return_value = (
            _persistent_entry(64, [], signature_tokens=signature_tokens),
            None,
        )
        scalar = _ParamInfo(
            name="seq_len",
            direction=ParamDirection.In,
            shape=None,
            dtype=DataType.FP32,
        )
        compiled = _fake_compiled([scalar], [])
        rt = DistributedWorker(
            compiled,
            persistent=True,
            reset_persistent_windows=False,
        )

        rt(7)
        rt([7])
        rt(7)

        assert signature_tokens[0] is not None
        assert signature_tokens[1] is None
        assert signature_tokens[2] is not None
        assert signature_tokens[2] != signature_tokens[0]
        rt.close()

    def test_device_memset_reset_skips_unused_host_zero_buffers(self, patched_setup):
        m = patched_setup
        m["worker"]._live_domains = {}
        m["worker"].device_memset_available = True
        orch = _PersistentOrch(m["worker"])
        m["worker"].submit.side_effect = lambda fn: (
            orch.run(fn),
            _ImmediateNativeHandle(),
        )[1]
        m["load_entry"].return_value = (_persistent_entry(64, []), None)
        compiled = _fake_compiled([_param("a", [16, 16])], [])
        compiled._distributed_config = DistributedConfig(device_ids=[0, 1])
        rt = DistributedWorker(compiled, persistent=True)
        arg = _resident(rt, (16, 16))

        rt(arg)
        handle = orch.handles[0]
        m["worker"].create_buffer.reset_mock()
        m["worker"].release_buffer.reset_mock()
        m["worker"].memset_all.reset_mock()

        rt(arg)

        m["worker"].create_buffer.assert_not_called()
        m["worker"].release_buffer.assert_not_called()
        m["worker"].memset_all.assert_called_once_with(
            {
                0: (handle[0].local_window_base, 64),
                1: (handle[1].local_window_base, 64),
            }
        )
        assert orch.copy_calls == []
        rt.close()

    @pytest.mark.parametrize(
        (
            "control_names",
            "data_names",
            "control_nbytes",
            "data_nbytes",
            "expected_control_bytes",
            "expected_window_bytes",
        ),
        [
            pytest.param(
                _K8_LEGACY_CONTROL_NAMES,
                _K8_LEGACY_DATA_NAMES,
                _K8_LEGACY_CONTROL_NBYTES,
                _K8_LEGACY_DATA_NBYTES,
                47616,
                32063232,
                id="legacy-ep",
            ),
            pytest.param(
                _K8_LEGACY_CONTROL_NAMES,
                _K8_LEGACY_DATA_NAMES,
                (4096, 4096, 4096, 8192, 8192, 8192, 10752),
                (1,) * len(_K8_LEGACY_DATA_NAMES),
                47616,
                47625,
                id="legacy-flexible-capacity",
            ),
            pytest.param(
                _K8_LOCAL_OWNER_CONTROL_NAMES,
                _K8_LOCAL_OWNER_DATA_NAMES,
                _K8_LOCAL_OWNER_CONTROL_NBYTES,
                _K8_LOCAL_OWNER_DATA_NBYTES,
                46080,
                11842560,
                id="local-owner",
            ),
        ],
    )
    def test_device_memset_k8_prefix_traces_only_reused_domain(
        self,
        patched_setup,
        monkeypatch,
        tmp_path,
        control_names,
        data_names,
        control_nbytes,
        data_nbytes,
        expected_control_bytes,
        expected_window_bytes,
    ):
        m = patched_setup
        buffer_nbytes = control_nbytes + data_nbytes
        assert sum(control_nbytes) == expected_control_bytes
        assert sum(buffer_nbytes) == expected_window_bytes
        buffer_names = tuple(f"{name}__ssa_v0" for name in control_names + data_names)
        trace_path = tmp_path / "persistent_reset.jsonl"
        monkeypatch.setenv("PYPTO_PERSISTENT_RESET_TRACE", str(trace_path))
        orch, rt, arg = _device_memset_persistent_runtime(
            m,
            buffer_names=buffer_names,
            buffer_nbytes=buffer_nbytes,
        )

        rt(arg)
        handle = orch.handles[0]
        m["worker"].memset_all.assert_not_called()
        m["worker"].create_buffer.assert_not_called()
        m["worker"].release_buffer.assert_not_called()
        assert not trace_path.exists()

        rt(arg)

        worker_ranges = {
            0: (handle[0].local_window_base, expected_control_bytes),
            1: (handle[1].local_window_base, expected_control_bytes),
        }
        m["worker"].memset_all.assert_called_once_with(worker_ranges)
        m["worker"].create_buffer.assert_not_called()
        m["worker"].release_buffer.assert_not_called()
        assert orch.copy_calls == []
        records = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
        assert len(records) == 1
        record = records[0]
        assert record["schema"] == "pypto.persistent-reset-trace.v1"
        assert record["seq"] == 0
        assert record["device_memset_available"] is True
        assert record["domain_count"] == 1
        domain = record["domains"][0]
        assert domain["requested_window_size"] == expected_window_bytes
        assert domain["actual_window_sizes"] == {
            "0": expected_window_bytes,
            "1": expected_window_bytes,
        }
        assert domain["k8_prefix_applied"] is True
        assert domain["k8_control_bytes"] == expected_control_bytes
        assert domain["k8_control_range_count"] == 1
        assert domain["k8_control_ranges"] == [[handle[0].local_window_base, expected_control_bytes]]
        assert domain["k8_full_window_bytes"] == expected_window_bytes
        assert domain["memset_all_us"] >= 0
        assert record["reset_body_us"] >= domain["memset_all_us"]
        rt.close()

    def test_device_memset_k8_prefix_rejects_control_after_data(self, patched_setup):
        m = patched_setup
        names = _K8_LOCAL_OWNER_CONTROL_NAMES + _K8_LOCAL_OWNER_DATA_NAMES
        nbytes = _K8_LOCAL_OWNER_CONTROL_NBYTES + _K8_LOCAL_OWNER_DATA_NBYTES
        reordered_names = (names[4],) + names[:4] + names[5:]
        reordered_nbytes = (nbytes[4],) + nbytes[:4] + nbytes[5:]
        orch, rt, arg = _device_memset_persistent_runtime(
            m,
            buffer_names=tuple(f"{name}__ssa_v0" for name in reordered_names),
            buffer_nbytes=reordered_nbytes,
        )

        rt(arg)
        assert len(orch.handles) == 1
        with pytest.raises(RuntimeError, match="control buffer .* appears after"):
            rt(arg)

        m["worker"].memset_all.assert_not_called()
        rt.close()

    @pytest.mark.parametrize(
        "swap_indices",
        [
            pytest.param((0, 1), id="equal-sized-controls"),
            pytest.param((4, 5), id="equal-sized-data"),
        ],
    )
    def test_device_memset_k8_local_owner_prefix_rejects_order_permutation(
        self,
        patched_setup,
        swap_indices,
    ):
        m = patched_setup
        names = list(_K8_LOCAL_OWNER_CONTROL_NAMES + _K8_LOCAL_OWNER_DATA_NAMES)
        nbytes = _K8_LOCAL_OWNER_CONTROL_NBYTES + _K8_LOCAL_OWNER_DATA_NBYTES
        first, second = swap_indices
        names[first], names[second] = names[second], names[first]
        orch, rt, arg = _device_memset_persistent_runtime(
            m,
            buffer_names=tuple(f"{name}__ssa_v0" for name in names),
            buffer_nbytes=nbytes,
        )

        rt(arg)
        assert len(orch.handles) == 1
        with pytest.raises(RuntimeError, match="ordered buffer fingerprint changed"):
            rt(arg)

        m["worker"].memset_all.assert_not_called()
        rt.close()

    @pytest.mark.parametrize(
        ("size_deltas", "error_match"),
        [
            pytest.param(((0, 1),), "control bytes 46081 != pinned 46080", id="control-prefix"),
            pytest.param(((4, 1),), "pinned full window 11842560", id="full-window"),
            pytest.param(
                ((0, 1), (1, -1)),
                "ordered buffer fingerprint changed",
                id="control-redistribution",
            ),
            pytest.param(
                ((4, 1), (5, -1)),
                "ordered buffer fingerprint changed",
                id="data-redistribution",
            ),
        ],
    )
    def test_device_memset_k8_local_owner_prefix_rejects_size_mismatch(
        self,
        patched_setup,
        size_deltas,
        error_match,
    ):
        m = patched_setup
        names = _K8_LOCAL_OWNER_CONTROL_NAMES + _K8_LOCAL_OWNER_DATA_NAMES
        nbytes = list(_K8_LOCAL_OWNER_CONTROL_NBYTES + _K8_LOCAL_OWNER_DATA_NBYTES)
        for size_index, delta in size_deltas:
            nbytes[size_index] += delta
        orch, rt, arg = _device_memset_persistent_runtime(
            m,
            buffer_names=tuple(f"{name}__ssa_v0" for name in names),
            buffer_nbytes=tuple(nbytes),
        )

        rt(arg)
        assert len(orch.handles) == 1
        with pytest.raises(RuntimeError, match=error_match):
            rt(arg)

        m["worker"].memset_all.assert_not_called()
        rt.close()

    def test_device_memset_unknown_layout_falls_back_to_full_window(
        self,
        patched_setup,
        monkeypatch,
        tmp_path,
    ):
        m = patched_setup
        window_nbytes = 128
        trace_path = tmp_path / "persistent_reset.jsonl"
        monkeypatch.setenv("PYPTO_PERSISTENT_RESET_TRACE", str(trace_path))
        orch, rt, arg = _device_memset_persistent_runtime(
            m,
            buffer_names=(
                "dense_attn_signal_stack_buf__ssa_v0",
                "unknown_data_buf__ssa_v0",
            ),
            buffer_nbytes=(16, 112),
        )

        rt(arg)
        handle = orch.handles[0]
        rt(arg)

        m["worker"].memset_all.assert_called_once_with(
            {
                0: (handle[0].local_window_base, window_nbytes),
                1: (handle[1].local_window_base, window_nbytes),
            }
        )
        record = json.loads(trace_path.read_text(encoding="utf-8"))
        domain = record["domains"][0]
        assert domain["k8_prefix_applied"] is False
        assert domain["k8_control_bytes"] is None
        assert domain["k8_control_ranges"] is None
        assert domain["k8_full_window_bytes"] is None
        rt.close()

    def test_host_reset_zero_buffer_lives_until_native_completion(self, patched_setup):
        m = patched_setup
        m["worker"]._live_domains = {}
        orch = _PersistentOrch(m["worker"])
        native = _ControlledNativeHandle()
        native_handles = [_ImmediateNativeHandle(), native]

        def worker_submit(fn):
            orch.run(fn)
            return native_handles.pop(0)

        m["worker"].submit.side_effect = worker_submit
        m["load_entry"].return_value = (_persistent_entry(64, []), None)
        compiled = _fake_compiled([_param("a", [16, 16])], [])
        rt = DistributedWorker(compiled, persistent=True)
        arg = _resident(rt, (16, 16))
        rt(arg)
        created: list[_FakeBuffer] = []
        released: list[_FakeBuffer] = []

        def create_dirty_host_buffer(nbytes: int) -> _FakeBuffer:
            buffer = _FakeBuffer(0, nbytes, host=True)
            ctypes.memset(int(buffer.base), 0xA5, nbytes)
            created.append(buffer)
            return buffer

        def release_after_completion(buffer: _FakeBuffer) -> None:
            assert m["worker"]._building_run_resources is None
            released.append(buffer)

        m["worker"].create_buffer.side_effect = create_dirty_host_buffer
        m["worker"].release_buffer.side_effect = release_after_completion

        handle = rt.submit(compiled, arg)

        assert len(created) == 1
        assert released == []
        assert handle.done is False
        native.complete()
        handle.result()
        assert released == created
        m["worker"].release_buffer.assert_called_once_with(created[0])
        rt.close()

    def test_warm_domain_serializes_submit_until_prior_native_completion(self, patched_setup):
        m = patched_setup
        m["worker"]._live_domains = {}
        orch = _PersistentOrch(m["worker"])
        first_native = _ControlledNativeHandle()
        second_native = _ControlledNativeHandle()
        natives = [_ImmediateNativeHandle(), first_native, second_native]

        def worker_submit(fn):
            orch.run(fn)
            return natives.pop(0)

        m["worker"].submit.side_effect = worker_submit
        seen_handles: list[Any] = []
        m["load_entry"].return_value = (_persistent_entry(64, seen_handles), None)
        compiled = _fake_compiled([_param("a", [16, 16])], [])
        rt = DistributedWorker(compiled, persistent=True)
        arg = _resident(rt, (16, 16))

        rt(arg)
        first = rt.submit(compiled, arg)
        domain = orch.handles[0]
        reset_count_before_second = len(orch.copy_calls)
        second_result: list[Any] = []
        second_error: list[BaseException] = []

        def submit_second() -> None:
            try:
                second_result.append(rt.submit(compiled, arg))
            except BaseException as exc:  # noqa: BLE001 - asserted below
                second_error.append(exc)

        caller = threading.Thread(target=submit_second)
        caller.start()
        assert first_native.result_started.wait(timeout=2)
        # The second submit is blocked in the persistent admission drain. It
        # must not invoke generated orchestration or reset the retained domain
        # while the first native request can still be using that window.
        assert caller.is_alive()
        assert not second_result
        assert not second_error
        assert len(rt._active_dispatch_handles) == 1
        assert len(orch.copy_calls) == reset_count_before_second
        assert seen_handles == [domain, domain]

        first_native.complete()
        caller.join(timeout=2)
        assert not caller.is_alive()
        assert not second_error
        assert len(second_result) == 1
        second = second_result[0]

        # The second request is admitted only after the first handle's result
        # cleanup retires its frame, so its reset and domain reuse are ordered.
        assert len(rt._active_dispatch_handles) == 1
        assert len(orch.copy_calls) > reset_count_before_second
        assert len(orch.allocate_calls) == 1
        assert seen_handles == [domain, domain, domain]
        assert second.done is False

        second_native.complete()
        first.result()
        second.result()
        rt.close()

        assert domain.release_count == 1
        assert domain.backend_release_count == 1
        assert domain.freed

    def test_reused_domain_skips_window_reset_when_disabled(self, patched_setup):
        m = patched_setup
        m["worker"]._live_domains = {}
        orch = _PersistentOrch(m["worker"])
        m["worker"].submit.side_effect = lambda fn: (orch.run(fn), _ImmediateNativeHandle())[1]
        seen_handles: list[Any] = []
        m["load_entry"].return_value = (_persistent_entry(64, seen_handles), None)
        compiled = _fake_compiled([_param("a", [16, 16])], [])
        compiled._distributed_config = DistributedConfig(device_ids=[0, 1])

        rt = DistributedWorker(compiled, persistent=True, reset_persistent_windows=False)
        arg = _resident(rt, (16, 16))
        rt(arg)
        rt(arg)
        rt.close()

        assert len(orch.allocate_calls) == 1
        assert seen_handles[0] is seen_handles[1]
        assert orch.copy_calls == []
        assert m["worker"].submit.call_count == 2

    def test_reset_rejects_unnamed_window_slack(self, patched_setup):
        m = patched_setup
        m["worker"]._live_domains = {}
        orch = _PersistentOrch(m["worker"])
        m["worker"].submit.side_effect = lambda fn: (orch.run(fn), _ImmediateNativeHandle())[1]
        m["load_entry"].return_value = (
            _persistent_entry(64, [], buffer_nbytes=(4,)),
            None,
        )
        compiled = _fake_compiled([_param("a", [16, 16])], [])
        rt = DistributedWorker(compiled, persistent=True)
        arg = _resident(rt, (16, 16))
        rt(arg)
        m["worker"].create_buffer.reset_mock()

        with pytest.raises(RuntimeError, match="named buffers to cover its window"):
            rt(arg)

        m["worker"].create_buffer.assert_not_called()
        assert m["worker"].submit.call_count == 1
        rt.close()

    def test_reset_buffer_is_released_after_failed_run(self, patched_setup):
        m = patched_setup
        m["worker"]._live_domains = {}
        orch = _PersistentOrch(m["worker"])
        m["worker"].submit.side_effect = lambda fn: (orch.run(fn), _ImmediateNativeHandle())[1]
        m["load_entry"].return_value = (_persistent_entry(64, []), None)
        compiled = _fake_compiled([_param("a", [16, 16])], [])
        rt = DistributedWorker(compiled, persistent=True)
        arg = _resident(rt, (16, 16))
        rt(arg)

        created: list[_FakeBuffer] = []
        released: list[_FakeBuffer] = []

        def create_dirty_host_buffer(nbytes: int) -> _FakeBuffer:
            buffer = _FakeBuffer(0, nbytes, host=True)
            ctypes.memset(int(buffer.base), 0xA5, nbytes)
            created.append(buffer)
            return buffer

        def release_after_run(buffer: _FakeBuffer) -> None:
            assert m["worker"]._building_run_resources is None
            released.append(buffer)

        m["worker"].create_buffer.side_effect = create_dirty_host_buffer
        m["worker"].release_buffer.side_effect = release_after_run
        original_copy_to = orch.copy_to

        def fail_copy(dst: _FakeBuffer, src: _FakeBuffer) -> None:
            original_copy_to(dst, src)
            raise RuntimeError("persistent reset copy failed")

        orch.copy_to = fail_copy
        with pytest.raises(RuntimeError, match="persistent reset copy failed"):
            rt(arg)

        assert len(created) == 1
        assert released == created
        assert orch.copy_calls[-1][2] == bytes(64)
        rt.close()

    def test_task_args_stay_alive_through_request_drain(self, patched_setup):
        m = patched_setup
        m["worker"]._live_domains = {}
        orch = _PersistentOrch(m["worker"])
        task_args_ref = None

        class TaskArgsSentinel:
            pass

        def entry(
            orch,
            _args,
            config,
            *,
            tensors,
            callables,
            sub_ids,
            _keep,
            world_size,
            _domain_provider=None,
        ):
            del orch, _args, config, tensors, callables, sub_ids, world_size, _domain_provider
            nonlocal task_args_ref
            task_args = TaskArgsSentinel()
            task_args_ref = weakref.ref(task_args)
            _keep.append(task_args)

        def assert_task_args_alive() -> None:
            assert task_args_ref is not None
            assert task_args_ref() is not None

        native = _ControlledNativeHandle(on_result=assert_task_args_alive)

        def worker_submit(fn):
            orch.run(fn, before_retire=assert_task_args_alive)
            native.complete()
            return native

        m["worker"].submit.side_effect = worker_submit
        m["load_entry"].return_value = (entry, None)
        compiled = _fake_compiled([_param("a", [16, 16])], [])

        rt = DistributedWorker(compiled, persistent=True)
        rt(_resident(rt, (16, 16)))

        assert task_args_ref is not None
        # Once the caller waits the handle, its bounded frame releases the
        # request keepalive instead of retaining it for the worker lifetime.
        assert task_args_ref() is None
        rt.close()

    def test_multi_program_domains_are_isolated_and_reused(self, patched_setup):
        m = patched_setup
        m["worker"]._live_domains = {}
        orch = _PersistentOrch(m["worker"])
        m["worker"].submit.side_effect = lambda fn: (orch.run(fn), _ImmediateNativeHandle())[1]
        seen_a: list[Any] = []
        seen_b: list[Any] = []
        m["load_entry"].side_effect = [
            (_persistent_entry(64, seen_a), None),
            (_persistent_entry(128, seen_b), None),
        ]
        compiled_a = _fake_compiled([_param("a", [16, 16])], [])
        compiled_b = _fake_compiled([_param("b", [16, 16])], [])
        compiled_a._distributed_config = DistributedConfig(device_ids=[0, 1])
        compiled_b._distributed_config = DistributedConfig(device_ids=[0, 1])
        rt = DistributedWorker(
            [compiled_a, compiled_b],
            persistent=True,
            reset_persistent_windows=True,
        )
        arg = _resident(rt, (16, 16))
        rt.run(compiled_a, arg)
        rt.run(compiled_b, arg)
        rt.run(compiled_a, arg)
        rt.close()

        assert m["worker"].submit.call_count == 3
        assert [call["name"] for call in orch.allocate_calls] == ["p0:comm_d0", "p1:comm_d0"]
        assert seen_a[0] is seen_a[1]
        assert seen_a[0] is not seen_b[0]
        # Only program A is reused; program B's first use needs no reset.
        assert [(dst.owner_worker_id, dst.nbytes) for dst, _src, _payload in orch.copy_calls] == [
            (0, 64),
            (1, 64),
        ]
        # Isolation does not change final ownership: each retained domain is
        # released exactly once when the shared persistent worker closes.
        assert [handle.release_count for handle in orch.handles] == [1, 1]
        assert [handle.close_sweep_count for handle in orch.handles] == [0, 0]
        assert [handle.backend_release_count for handle in orch.handles] == [1, 1]

    def test_domain_release_error_reaches_close(self, patched_setup):
        m = patched_setup
        m["worker"]._live_domains = {}
        orch = _PersistentOrch(m["worker"])
        m["worker"].submit.side_effect = lambda fn: (orch.run(fn), _ImmediateNativeHandle())[1]
        m["load_entry"].return_value = (_persistent_entry(64, []), None)
        compiled = _fake_compiled([_param("a", [16, 16])], [])
        rt = DistributedWorker(compiled, persistent=True)
        rt(_resident(rt, (16, 16)))
        handle = orch.handles[0]
        handle.release_error = RuntimeError("persistent domain release failed")

        with pytest.raises(RuntimeError, match="persistent domain release failed"):
            rt.close()

        assert handle.release_count == 1
        assert handle.close_sweep_count == 1
        # Worker.close() observes the cached backend failure instead of
        # issuing the collective release a second time.
        assert handle.backend_release_count == 1
        assert not handle.freed
        assert m["worker"]._live_domains == {"p0:comm_d0": handle}
        m["worker"].close.assert_called_once_with()

    def test_unfreed_domain_release_reaches_close(self, patched_setup):
        m = patched_setup
        orch = _PersistentOrch(m["worker"])
        m["worker"].submit.side_effect = lambda fn: (orch.run(fn), _ImmediateNativeHandle())[1]
        m["load_entry"].return_value = (_persistent_entry(64, []), None)
        compiled = _fake_compiled([_param("a", [16, 16])], [])
        rt = DistributedWorker(compiled, persistent=True)
        rt(_resident(rt, (16, 16)))
        handle = orch.handles[0]
        handle.free_on_release = False

        with pytest.raises(RuntimeError, match="did not free.*p0:comm_d0"):
            rt.close()

        assert handle.release_count == 1
        assert handle.close_sweep_count == 1
        assert handle.backend_release_count == 1
        assert handle.freed
        assert m["worker"]._live_domains == {}
        m["worker"].close.assert_called_once_with()

    def test_dispatch_error_defers_domain_to_worker_close(self, patched_setup):
        m = patched_setup
        m["worker"]._live_domains = {}
        orch = _PersistentOrch(m["worker"])
        m["worker"].submit.side_effect = lambda fn: (orch.run(fn), _ImmediateNativeHandle())[1]

        def failing_entry(
            orch,
            _args,
            config,
            *,
            tensors,
            callables,
            sub_ids,
            _keep,
            world_size,
            _domain_provider=None,
        ):
            del orch, _args, config, tensors, callables, sub_ids, _keep
            assert _domain_provider is not None
            with _domain_provider(
                name="comm_d0",
                workers=[*range(world_size)],
                window_size=64,
                buffers=[SimpleNamespace(name="signal", dtype="opaque", count=4, nbytes=4)],
            ):
                raise RuntimeError("persistent dispatch failed")

        m["load_entry"].return_value = (failing_entry, None)
        compiled = _fake_compiled([_param("a", [16, 16])], [])
        rt = DistributedWorker(compiled, persistent=True)
        arg = _resident(rt, (16, 16))

        with pytest.raises(RuntimeError, match="persistent dispatch failed"):
            rt(arg)
        handle = orch.handles[0]
        owner = orch.resources[0]
        assert owner.retired
        assert owner.live_domains == {}
        assert handle.release_count == 0
        assert handle.close_sweep_count == 0
        assert m["worker"]._live_domains == {"p0:comm_d0": handle}

        rt.close()

        # A failed request never calls handle.release(): even a conservatively
        # unretired owner must stay globally reachable for whole-tree teardown.
        assert handle.release_count == 0
        assert handle.close_sweep_count == 1
        assert handle.backend_release_count == 1
        assert handle.freed
        assert m["worker"]._live_domains == {}
        assert m["worker"].submit.call_count == 1

    def test_abandoned_run_keeps_domain_reachable_for_worker_close(self, patched_setup):
        m = patched_setup
        orch = _PersistentOrch(m["worker"])
        m["worker"].submit.side_effect = orch.run_with_abandoned_finalization
        m["load_entry"].return_value = (_persistent_entry(64, []), None)
        compiled = _fake_compiled([_param("a", [16, 16])], [])
        rt = DistributedWorker(compiled, persistent=True)
        arg = _resident(rt, (16, 16))

        with pytest.raises(RuntimeError, match="run finalization abandoned"):
            rt(arg)

        handle = orch.handles[0]
        owner = orch.resources[0]
        assert not owner.retired
        assert owner.live_domains == {}
        assert owner.pending_release_domains == []
        assert handle.release_count == 0
        assert m["worker"]._live_domains == {"p0:comm_d0": handle}

        rt.close()

        assert handle.release_count == 0
        assert handle.close_sweep_count == 1
        assert handle.backend_release_count == 1
        assert handle.freed
        assert m["worker"]._live_domains == {}

    def test_native_failure_waits_for_handle_finalization(self, patched_setup):
        m = patched_setup
        m["worker"]._live_domains = {}
        native = _ControlledNativeHandle()

        def worker_submit(fn):
            del fn
            return native

        m["worker"].submit.side_effect = worker_submit

        def persistent_entry(
            orch,
            _args,
            config,
            *,
            tensors,
            callables,
            sub_ids,
            _keep,
            world_size,
            _domain_provider=None,
        ):
            del orch, _args, config, tensors, callables, sub_ids, _keep, world_size, _domain_provider

        m["load_entry"].return_value = (persistent_entry, None)
        compiled = _fake_compiled([_param("a", [16, 16])], [])
        rt = DistributedWorker(compiled, persistent=True)
        arg = _resident(rt, (16, 16))
        caller_done = threading.Event()
        errors: list[BaseException] = []

        def call_worker() -> None:
            try:
                rt(arg)
            except BaseException as exc:  # noqa: BLE001 - asserted below
                errors.append(exc)
            finally:
                caller_done.set()

        caller = threading.Thread(target=call_worker)
        caller.start()
        assert native.result_started.wait(timeout=2)
        # A native failure is not published while its handle is still running.
        assert not caller_done.is_set()

        native.complete(RuntimeError("persistent dispatch failed before cleanup"))
        caller.join(timeout=2)
        assert not caller.is_alive()
        assert caller_done.is_set()
        assert len(errors) == 1
        assert isinstance(errors[0], RuntimeError)
        assert str(errors[0]) == "persistent dispatch failed before cleanup"
        rt.close()


class TestNamedInheritedHostRanges:
    """The zero-copy H2D/D2H path and, more importantly, where it must NOT engage.

    Staging a copy through shm costs a full host-side memcpy of the payload, which a
    fork-inherited MAP_SHARED range does not need. The boundary matters more than the
    optimisation: a MAP_PRIVATE range is inherited too, but copy-on-write freezes the
    child's view at fork, so naming one would upload stale bytes the moment the parent
    writes. These tests pin that boundary.
    """

    @staticmethod
    def _dev(rt):
        return _resident(rt, (4, 4))

    def test_copy_to_names_a_contained_shared_range(self, patched_setup):
        host = torch.zeros(4, 4, dtype=torch.float32).share_memory_()
        rt = DistributedWorker(_fake_compiled([_param("a", [4, 4])], []), inherited_host_tensors=[host])
        dev = self._dev(rt)
        nbytes = host.numel() * host.element_size()
        patched_setup["worker"].create_buffer.reset_mock()

        rt.copy_to(dev.data_ptr, host.data_ptr(), nbytes)

        named = patched_setup["worker"].copy_to.call_args.args[1]
        assert isinstance(named, _NamedHostRange)
        assert (named.base, named.nbytes) == (host.data_ptr(), nbytes)
        assert (named.backend_kind, named.access) == ("FORK_SHM", "READWRITE")
        # The point of naming it: no staging buffer is created at all.
        patched_setup["worker"].create_buffer.assert_not_called()
        rt.close()

    def test_copy_to_stages_a_private_inherited_range(self, patched_setup):
        # MAP_PRIVATE: the child's pages are frozen at fork, so a named range would upload
        # whatever was there before a later parent write. Staging reads the parent's current
        # contents instead, which is why it stays the correct path here.
        host = torch.zeros(4, 4, dtype=torch.float32)
        rt = DistributedWorker(_fake_compiled([_param("a", [4, 4])], []), inherited_host_tensors=[host])
        dev = self._dev(rt)
        nbytes = host.numel() * host.element_size()
        host.fill_(1.0)

        rt.copy_to(dev.data_ptr, host.data_ptr(), nbytes)

        staged = patched_setup["worker"].copy_to.call_args.args[1]
        assert not isinstance(staged, _NamedHostRange)
        assert ctypes.string_at(staged.base, nbytes) == ctypes.string_at(host.data_ptr(), nbytes)
        rt.close()

    def test_copy_to_stages_a_partially_overlapping_range(self, patched_setup):
        host = torch.zeros(8, dtype=torch.float32).share_memory_()
        rt = DistributedWorker(_fake_compiled([_param("a", [4, 4])], []), inherited_host_tensors=[host])
        dev = self._dev(rt)
        nbytes = host.numel() * host.element_size()
        patched_setup["worker"].create_buffer.reset_mock()

        # Starts inside the span but runs past its end: naming it would hand the child a
        # Buffer longer than the mapping it vouches for.
        rt.copy_to(dev.data_ptr, host.data_ptr() + 4, nbytes)

        assert not isinstance(patched_setup["worker"].copy_to.call_args.args[1], _NamedHostRange)
        patched_setup["worker"].create_buffer.assert_called_once()
        rt.close()

    def test_copy_from_names_a_shared_range(self, patched_setup):
        host = torch.zeros(4, 4, dtype=torch.float32).share_memory_()
        rt = DistributedWorker(_fake_compiled([_param("a", [4, 4])], []), inherited_host_tensors=[host])
        dev = self._dev(rt)
        nbytes = host.numel() * host.element_size()
        patched_setup["worker"].create_buffer.reset_mock()

        rt.copy_from(host.data_ptr(), dev.data_ptr, nbytes)

        named = patched_setup["worker"].copy_from.call_args.args[0]
        assert isinstance(named, _NamedHostRange)
        assert named.backend_kind == "FORK_SHM"
        patched_setup["worker"].create_buffer.assert_not_called()
        rt.close()

    def test_releasing_inherited_refs_sends_later_copies_back_to_staging(self, patched_setup):
        host = torch.zeros(4, 4, dtype=torch.float32).share_memory_()
        rt = DistributedWorker(_fake_compiled([_param("a", [4, 4])], []), inherited_host_tensors=[host])
        dev = self._dev(rt)
        nbytes = host.numel() * host.element_size()

        rt.release_inherited_host_tensor_refs()
        rt.copy_to(dev.data_ptr, host.data_ptr(), nbytes)

        # The parent has dropped its references, so nobody vouches for the mapping anymore.
        assert not isinstance(patched_setup["worker"].copy_to.call_args.args[1], _NamedHostRange)
        rt.close()

    def test_one_range_keeps_one_identity_across_copies(self, patched_setup):
        """Re-copying a range must reuse its identity, not mint a new one.

        Two reasons, both in the consumer: ``ImportRegistry.materialize`` refuses a second
        descriptor for an identity it already handed out, so a fresh id per copy would make the
        second copy of a range fail outright; and a consumer only drops an entry when the owner
        releases the Buffer, which the named path never does — so per-copy identities would
        leave one permanent ``ImportedBuffer`` per copy in every chip child. A per-step D2H
        read-back would grow that registry for the life of the process.
        """
        host = torch.zeros(4, 4, dtype=torch.float32).share_memory_()
        rt = DistributedWorker(_fake_compiled([_param("a", [4, 4])], []), inherited_host_tensors=[host])
        dev = self._dev(rt)
        nbytes = host.numel() * host.element_size()

        rt.copy_to(dev.data_ptr, host.data_ptr(), nbytes)
        first = patched_setup["worker"].copy_to.call_args.args[1]
        rt.copy_to(dev.data_ptr, host.data_ptr(), nbytes)
        second = patched_setup["worker"].copy_to.call_args.args[1]

        assert (first.owner, first.buffer_id) == (second.owner, second.buffer_id)
        rt.close()

    def test_distinct_sub_ranges_get_distinct_identities(self, patched_setup):
        """Identity is per range, so a sharded upload's halves must not collide on one name."""
        host = torch.zeros(4, 4, dtype=torch.float32).share_memory_()
        rt = DistributedWorker(_fake_compiled([_param("a", [4, 4])], []), inherited_host_tensors=[host])
        dev = self._dev(rt)
        half = host.numel() * host.element_size() // 2

        rt.copy_to(dev.data_ptr, host.data_ptr(), half)
        first = patched_setup["worker"].copy_to.call_args.args[1]
        rt.copy_to(dev.data_ptr, host.data_ptr() + half, half)
        second = patched_setup["worker"].copy_to.call_args.args[1]

        assert first.owner == second.owner
        assert first.buffer_id != second.buffer_id
        rt.close()

    def test_concurrent_copies_of_distinct_ranges_never_share_an_identity(self, patched_setup):
        """The configuration `alloc_stacked_tensor` actually creates: one thread per chip.

        Identity minting is not atomic on its own — `+=` is load/add/store and the first-time
        owner mint is check-then-act — and two ranges landing on one identity is a hard failure
        in the consumer, not a silent one. A single-threaded test cannot observe that, so this
        drives the path concurrently and asserts the mapping is still one-to-one.
        """
        host = torch.zeros(64, 4, dtype=torch.float32).share_memory_()
        rt = DistributedWorker(_fake_compiled([_param("a", [4, 4])], []), inherited_host_tensors=[host])
        row = 4 * host.element_size()
        rows = 64
        seen: list[tuple[int, tuple[Any, int]]] = []
        lock = threading.Lock()

        def _copy(index: int) -> None:
            offset = host.data_ptr() + index * row
            named = rt._named_host_buffer(offset, row)
            with lock:
                seen.append((offset, (named.owner, named.buffer_id)))

        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(_copy, range(rows)))

        identities = [identity for _, identity in seen]
        assert len(seen) == rows
        # One identity per range, and no range sharing another's: both directions matter, since
        # a duplicate id makes the child refuse the import and a missing one breaks reuse.
        assert len(set(identities)) == rows
        assert len({offset for offset, _ in seen}) == rows
        rt.close()

    def test_identity_cache_is_dropped_with_the_ranges_it_names(self, patched_setup):
        """After the parent releases its references, nothing may keep naming those ranges."""
        host = torch.zeros(4, 4, dtype=torch.float32).share_memory_()
        rt = DistributedWorker(_fake_compiled([_param("a", [4, 4])], []), inherited_host_tensors=[host])
        dev = self._dev(rt)
        nbytes = host.numel() * host.element_size()
        rt.copy_to(dev.data_ptr, host.data_ptr(), nbytes)
        assert rt._named_identities

        rt.release_inherited_host_tensor_refs()

        assert not rt._named_identities
        rt.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
