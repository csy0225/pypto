# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Verify the pypto-owned ``make_tensor_arg`` used by generated distributed
orchestration code.

It must:
- derive an address-free wire ``Tensor`` from a worker-resident
  :class:`DeviceTensor`'s retained ``Buffer``;
- pass an already-built ``Tensor`` through unchanged;
- delegate a host ``torch.Tensor`` to simpler's worker-aware wire helper.
"""

from types import MappingProxyType, SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from pypto.runtime import DeviceTensor

# ``task_interface`` eagerly imports the optional ``simpler`` runtime package;
# keep pure descriptor/cache tests active and skip only tests that need Simpler.
try:
    import simpler  # noqa: F401  # pyright: ignore[reportMissingImports]
except ImportError:
    _has_simpler = False
else:
    _has_simpler = True

requires_simpler = pytest.mark.skipif(not _has_simpler, reason="make_tensor_arg requires the simpler package")


@requires_simpler
def test_device_tensor_derives_wire_tensor_from_retained_buffer():
    captured: dict = {"tensor_calls": []}

    class FakeBuffer:
        base = 0xABCD

        def tensor(self, *, shapes, dtype):
            captured["tensor_calls"].append({"shapes": tuple(shapes), "dtype": dtype})
            return MagicMock(name="wire_tensor")

    buffer = FakeBuffer()
    dt = DeviceTensor(buffer.base, (8, 16), torch.float16, buffer=buffer)
    worker = MagicMock(name="worker")
    owner = MagicMock(name="pypto_owner")

    from pypto.runtime.tensor_arg import bind_tensor_arg_owner, make_tensor_arg  # noqa: PLC0415

    bind_tensor_arg_owner(worker, owner)

    with patch(
        "pypto.runtime.task_interface.torch_dtype_to_datatype",
        side_effect=lambda d: f"<dtype:{d}>",
    ):
        make_tensor_arg(worker, dt)

    owner._require_owned_resident_tensor.assert_called_once_with(dt, "Tensor argument")
    assert len(captured["tensor_calls"]) == 1
    call = captured["tensor_calls"][0]
    assert call["shapes"] == (8, 16)
    assert call["dtype"] == "<dtype:torch.float16>"


@requires_simpler
def test_retained_buffer_is_rejected_without_pypto_owner_binding():
    class FakeBuffer:
        base = 0xABCD

        def tensor(self, *, shapes, dtype):
            raise AssertionError("an unowned Buffer must be rejected before tensor()")

    from pypto.runtime.tensor_arg import make_tensor_arg  # noqa: PLC0415

    buffer = FakeBuffer()
    dt = DeviceTensor(buffer.base, (4,), torch.float32, buffer=buffer)
    with pytest.raises(TypeError, match="one-shot or raw simpler Worker"):
        make_tensor_arg(MagicMock(name="unbound_worker"), dt)


@requires_simpler
def test_owner_liveness_failure_prevents_wire_tensor_creation():
    class FakeBuffer:
        base = 0xABCD

        def tensor(self, *, shapes, dtype):
            raise AssertionError("a stale Buffer must be rejected before tensor()")

    from pypto.runtime.tensor_arg import bind_tensor_arg_owner, make_tensor_arg  # noqa: PLC0415

    worker = MagicMock(name="worker")
    owner = MagicMock(name="pypto_owner")
    owner._require_owned_resident_tensor.side_effect = ValueError("not a live allocation")
    bind_tensor_arg_owner(worker, owner)
    buffer = FakeBuffer()
    dt = DeviceTensor(buffer.base, (4,), torch.float32, buffer=buffer)

    with pytest.raises(ValueError, match="not a live allocation"):
        make_tensor_arg(worker, dt)


@requires_simpler
def test_stale_raw_backend_is_rejected_before_owner_validation():
    class FakeBuffer:
        base = 0xABCD

        def tensor(self, *, shapes, dtype):
            raise AssertionError("a stale backend must be rejected before tensor()")

    from pypto.runtime.tensor_arg import bind_tensor_arg_owner, make_tensor_arg  # noqa: PLC0415

    old_worker = MagicMock(name="old_worker")
    owner = MagicMock(name="pypto_owner")
    bind_tensor_arg_owner(old_worker, owner)
    owner._tensor_arg_worker = MagicMock(name="replacement_worker")
    buffer = FakeBuffer()
    dt = DeviceTensor(buffer.base, (4,), torch.float32, buffer=buffer)

    with pytest.raises(ValueError, match="stale simpler Worker backend"):
        make_tensor_arg(old_worker, dt)
    owner._require_owned_resident_tensor.assert_not_called()


@requires_simpler
def test_raw_pointer_device_tensor_is_rejected_for_wire_dispatch():
    from pypto.runtime.tensor_arg import make_tensor_arg  # noqa: PLC0415

    with pytest.raises(TypeError, match="raw-pointer DeviceTensor"):
        make_tensor_arg(MagicMock(name="worker"), DeviceTensor(0x1000, (4,), torch.float32))


@requires_simpler
def test_wire_tensor_passes_through():
    from pypto.runtime.tensor_arg import make_tensor_arg  # noqa: PLC0415

    class FakeWireTensor:
        pass

    wire = FakeWireTensor()
    with patch("pypto.runtime.task_interface.Tensor", FakeWireTensor):
        assert make_tensor_arg(MagicMock(name="worker"), wire) is wire


@requires_simpler
def test_host_tensor_delegates_to_simpler():
    host = torch.zeros(4, 4, dtype=torch.float32)
    sentinel = MagicMock(name="Tensor(host)")
    worker = MagicMock(name="worker")

    with patch("simpler_setup.torch_interop.make_tensor_arg", return_value=sentinel) as impl:
        from pypto.runtime.tensor_arg import make_tensor_arg  # noqa: PLC0415

        result = make_tensor_arg(worker, host)

    impl.assert_called_once_with(worker, host)
    assert result is sentinel


def test_task_args_signature_ignores_in_place_payload_mutation():
    from pypto.runtime.tensor_arg import _task_args_signature  # noqa: PLC0415

    host = torch.zeros(4, 4, dtype=torch.float32).share_memory_()
    before = _task_args_signature((host,))
    host.add_(1)
    assert _task_args_signature((host,)) == before


def test_task_args_signature_tracks_tensor_descriptor_replacement():
    from pypto.runtime.tensor_arg import _task_args_signature  # noqa: PLC0415

    host = torch.zeros(4, 4, dtype=torch.float32).share_memory_()
    before = _task_args_signature((host,))
    replacement = host.view(2, 8)
    assert _task_args_signature((replacement,)) != before


def test_task_args_signature_tracks_same_object_stride_change():
    from pypto.runtime.tensor_arg import _task_args_signature  # noqa: PLC0415

    host = torch.zeros(4, 4, dtype=torch.float32).share_memory_()
    before = _task_args_signature((host,))
    host.transpose_(0, 1)
    assert _task_args_signature((host,)) != before


@requires_simpler
def test_task_args_signature_tracks_device_and_stacked_buffer_lifetime():
    from pypto.runtime import StackedDeviceTensor  # noqa: PLC0415
    from pypto.runtime.tensor_arg import _task_args_signature  # noqa: PLC0415

    class FakeBuffer:
        def __init__(self, base: int, buffer_id: int) -> None:
            self.base = base
            self.closed = False
            self.identity = SimpleNamespace(
                owner_instance_id=b"owner-id",
                buffer_id=buffer_id,
                generation=1,
            )
            self.address_space = 1
            self.access = 2
            self.backend_kind = 4
            self.nbytes = 64
            self.owner_worker_path_id = 7
            self.body = base.to_bytes(8, "little")

        def tensor(self, *, shapes, dtype):
            return shapes, dtype

    first_buffer = FakeBuffer(0x1000, 1)
    second_buffer = FakeBuffer(0x2000, 2)
    first = DeviceTensor(first_buffer.base, (4,), torch.float32, buffer=first_buffer)
    second = DeviceTensor(second_buffer.base, (4,), torch.float32, buffer=second_buffer)
    stacked = StackedDeviceTensor((first, second), (2, 4), (0, 1))

    before = _task_args_signature((first, stacked))
    assert _task_args_signature((first, stacked)) == before
    first_buffer.closed = True
    assert _task_args_signature((first, stacked)) != before


@requires_simpler
def test_task_args_signature_tracks_simpler_tensor_descriptor():
    from pypto.runtime.tensor_arg import _task_args_signature  # noqa: PLC0415
    from simpler.buffer import (  # noqa: PLC0415  # pyright: ignore[reportMissingImports]
        AccessMode,
        AddressSpace,
        BackendKind,
        BufferDescriptor,
        CanonicalIdentity,
        DataType,
        Tensor,
    )

    identity = CanonicalIdentity(b"owner-id", 11, 1)
    descriptor = BufferDescriptor(
        identity,
        AddressSpace.HOST,
        AccessMode.READWRITE,
        BackendKind.FORK_SHM,
        64,
        (0x1000).to_bytes(8, "little"),
    )
    first = Tensor(descriptor, 0, (4,), (1,), DataType.FLOAT32)
    replacement = Tensor(descriptor, 0, (4,), (1,), DataType.FLOAT32)
    changed_identity = CanonicalIdentity(b"owner-id", 12, 1)
    changed_descriptor = BufferDescriptor(
        changed_identity,
        AddressSpace.HOST,
        AccessMode.READWRITE,
        BackendKind.FORK_SHM,
        64,
        (0x1000).to_bytes(8, "little"),
    )
    changed = Tensor(changed_descriptor, 0, (4,), (1,), DataType.FLOAT32)

    before = _task_args_signature((first,))
    assert _task_args_signature((first,)) == before
    assert _task_args_signature((replacement,)) == before
    assert _task_args_signature((changed,)) != before


def test_task_args_signature_disables_unknown_and_cyclic_values():
    from pypto.runtime.tensor_arg import _task_args_signature  # noqa: PLC0415

    assert _task_args_signature((object(),)) is None
    cyclic: list[object] = []
    cyclic.append(cyclic)
    assert _task_args_signature((cyclic,)) is None


def test_task_args_signature_for_cache_memoizes_validated_token():
    from pypto.runtime import tensor_arg  # noqa: PLC0415

    def provider(**_kwargs):
        return None

    setattr(provider, "_pypto_task_args_signature_token", ("program", 0, 7))
    orch = SimpleNamespace()
    cache = tensor_arg._task_args_cache_for_orch(orch, provider)
    host = torch.zeros(4, 4, dtype=torch.float32).share_memory_()

    with patch.object(tensor_arg, "_task_args_signature", wraps=tensor_arg._task_args_signature) as signature:
        first = tensor_arg._task_args_signature_for_cache(cache, (host,))
        second = tensor_arg._task_args_signature_for_cache(cache, (host,))

    assert first == second
    assert signature.call_count == 1


def test_task_args_signature_for_cache_recomputes_when_validated_token_changes():
    from pypto.runtime import tensor_arg  # noqa: PLC0415

    def provider(**_kwargs):
        return None

    orch = SimpleNamespace()
    host = torch.zeros(4, 4, dtype=torch.float32).share_memory_()
    with patch.object(tensor_arg, "_task_args_signature", wraps=tensor_arg._task_args_signature) as signature:
        setattr(provider, "_pypto_task_args_signature_token", ("program", 0, 1))
        cache = tensor_arg._task_args_cache_for_orch(orch, provider)
        tensor_arg._task_args_signature_for_cache(cache, (host,))
        setattr(provider, "_pypto_task_args_signature_token", ("program", 0, 2))
        cache = tensor_arg._task_args_cache_for_orch(orch, provider)
        tensor_arg._task_args_signature_for_cache(cache, (host,))

    assert signature.call_count == 2


def test_task_args_signature_for_cache_without_token_recomputes_and_never_memoizes_none():
    from pypto.runtime import tensor_arg  # noqa: PLC0415

    def provider(**_kwargs):
        return None

    orch = SimpleNamespace()
    values = ([1],)
    with patch.object(tensor_arg, "_task_args_signature", wraps=tensor_arg._task_args_signature) as signature:
        cache = tensor_arg._task_args_cache_for_orch(orch, provider)
        first = tensor_arg._task_args_signature_for_cache(cache, values)
        values[0].append(2)
        cache = tensor_arg._task_args_cache_for_orch(orch, provider)
        second = tensor_arg._task_args_signature_for_cache(cache, values)

    assert first != second
    assert signature.call_count == 2
    assert cache is not None
    memo = cache.get(tensor_arg._PYPTO_TASK_ARGS_SIGNATURE_MEMO_KEY, {})
    assert None not in memo


def test_task_args_signature_memo_is_bounded_and_refreshes_lru_hits():
    from pypto.runtime import tensor_arg  # noqa: PLC0415

    def provider(**_kwargs):
        return None

    orch = SimpleNamespace()
    host = torch.zeros(4, 4, dtype=torch.float32).share_memory_()
    limit = tensor_arg._PYPTO_TASK_ARGS_SIGNATURE_MEMO_MAX_ENTRIES
    tokens = [("program", 0, generation) for generation in range(limit + 2)]

    with patch.object(tensor_arg, "_task_args_signature", wraps=tensor_arg._task_args_signature) as signature:
        for token in tokens[: limit + 1]:
            setattr(provider, "_pypto_task_args_signature_token", token)
            cache = tensor_arg._task_args_cache_for_orch(orch, provider)
            tensor_arg._task_args_signature_for_cache(cache, (host,))

        assert cache is not None
        memo = cache[tensor_arg._PYPTO_TASK_ARGS_SIGNATURE_MEMO_KEY]
        assert len(memo) == limit
        assert tokens[0] not in memo

        setattr(provider, "_pypto_task_args_signature_token", tokens[1])
        cache = tensor_arg._task_args_cache_for_orch(orch, provider)
        tensor_arg._task_args_signature_for_cache(cache, (host,))
        setattr(provider, "_pypto_task_args_signature_token", tokens[-1])
        cache = tensor_arg._task_args_cache_for_orch(orch, provider)
        tensor_arg._task_args_signature_for_cache(cache, (host,))

    memo = cache[tensor_arg._PYPTO_TASK_ARGS_SIGNATURE_MEMO_KEY]
    assert len(memo) == limit
    assert tokens[1] in memo
    assert tokens[2] not in memo
    assert tokens[-1] in memo
    assert signature.call_count == limit + 2


def test_task_args_cache_helpers_fail_open_on_broken_metadata():
    from pypto.runtime import tensor_arg  # noqa: PLC0415

    def provider(**_kwargs):
        return None

    class ReadOnlyOrch:
        @property
        def _pypto_task_args_cache_v1(self):
            return None

    class BrokenOrch:
        @property
        def _pypto_task_args_cache_v1(self):
            raise RuntimeError("cache unavailable")

    class BrokenProvider:
        @property
        def _pypto_task_args_signature_token(self):
            raise RuntimeError("token unavailable")

    class BrokenCache:
        def get(self, _key):
            raise RuntimeError("metadata unavailable")

    assert (
        tensor_arg._task_args_cache_for_orch(
            SimpleNamespace(_pypto_task_args_cache_v1=MappingProxyType({})), provider
        )
        is None
    )
    assert tensor_arg._task_args_cache_for_orch(ReadOnlyOrch(), provider) is None
    assert tensor_arg._task_args_cache_for_orch(BrokenOrch(), provider) is None
    assert tensor_arg._task_args_cache_for_orch(SimpleNamespace(), BrokenProvider()) is None

    values = ([1],)
    expected = tensor_arg._task_args_signature(values)
    with patch.object(tensor_arg, "_task_args_signature", wraps=tensor_arg._task_args_signature) as signature:
        assert tensor_arg._task_args_signature_for_cache(BrokenCache(), values) == expected
    assert signature.call_count == 1


def test_task_args_cache_cold_insert_uses_entry_count_without_iteration():
    from pypto.runtime import tensor_arg  # noqa: PLC0415

    class NoIterDict(dict):
        def __iter__(self):
            raise AssertionError("cold TaskArgs insertion must not scan the cache")

    slot = ("slot", 0)
    cache = NoIterDict({tensor_arg._PYPTO_TASK_ARGS_CACHE_ENTRY_COUNT_KEY: 0})
    tensor_arg._task_args_cache_store(cache, slot, ("sig", 0), "ta0")
    tensor_arg._task_args_cache_store(cache, slot, ("sig", 1), "ta1")

    assert cache[slot] == (("sig", 1), "ta1")
    assert cache[tensor_arg._PYPTO_TASK_ARGS_CACHE_ENTRY_COUNT_KEY] == 1


def test_task_args_cache_eviction_preserves_signature_metadata():
    from pypto.runtime import tensor_arg  # noqa: PLC0415

    def provider(**_kwargs):
        return None

    token = ("program", 0, 1)
    setattr(provider, "_pypto_task_args_signature_token", token)
    orch = SimpleNamespace()
    cache = tensor_arg._task_args_cache_for_orch(orch, provider)
    assert cache is not None
    host = torch.zeros(4, 4, dtype=torch.float32).share_memory_()
    tensor_arg._task_args_signature_for_cache(cache, (host,))

    with patch.object(tensor_arg, "_PYPTO_TASK_ARGS_CACHE_MAX_ENTRIES", 2):
        tensor_arg._task_args_cache_store(cache, ("slot", 0), ("sig", 0), "ta0")
        tensor_arg._task_args_cache_store(cache, ("slot", 1), ("sig", 1), "ta1")
        tensor_arg._task_args_cache_store(cache, ("slot", 2), ("sig", 2), "ta2")

    assert cache[tensor_arg._PYPTO_TASK_ARGS_SIGNATURE_TOKEN_KEY] == token
    assert token in cache[tensor_arg._PYPTO_TASK_ARGS_SIGNATURE_MEMO_KEY]
    slots = [key for key in cache if isinstance(key, tuple) and key[:1] == ("slot",)]
    assert slots == [("slot", 1), ("slot", 2)]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
