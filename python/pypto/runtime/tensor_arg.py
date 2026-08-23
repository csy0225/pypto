# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""``make_tensor_arg`` used by generated distributed orchestration code.

The generated ``orchestration/host_orch.py`` builds simpler ``TaskArgs`` by
calling ``make_tensor_arg(orch._worker, tensors["<name>"])`` for every tensor
parameter. Current simpler deliberately separates this address-free wire
conversion from the direct-chip ``make_chip_tensor_arg`` helper.
This pypto-owned wrapper widens that conversion to also accept worker-resident
:class:`~pypto.runtime.DeviceTensor` handles (and already-built simpler
``Tensor`` values), so distributed programs can be invoked with
pre-uploaded device buffers — mirroring the L2 path in
:func:`pypto.runtime.runner.execute_compiled`.

Host ``torch.Tensor`` arguments are delegated to
``simpler_setup.torch_interop.make_tensor_arg(worker, tensor)``. Never route
this path through :mod:`pypto.runtime.task_interface`: its compatibility
``make_tensor_arg`` alias is chip-only and produces a ``ChipTensor``.
"""

import math
import weakref
from functools import cache
from typing import Any

import torch

_PYPTO_OWNER_REF_ATTR = "_pypto_tensor_owner_ref"
_UNSUPPORTED = object()
_PYPTO_TASK_ARGS_CACHE_MAX_ENTRIES = 4096


def bind_tensor_arg_owner(worker: Any, owner: Any) -> None:
    """Bind a raw simpler Worker to its weak PyPTO owner.

    Generated orchestration receives the raw simpler Worker (``orch._worker``),
    while DeviceTensor liveness is tracked by the public PyPTO Worker.  This
    weak backlink lets wire conversion consult the authoritative PyPTO Buffer
    registry without introducing a reference cycle or changing generated code.
    """
    setattr(worker, _PYPTO_OWNER_REF_ATTR, weakref.ref(owner))
    owner._tensor_arg_worker = worker


def _require_device_tensor_owner(worker: Any, arg: Any) -> None:
    """Require *arg* to be a live DeviceTensor owned by *worker*'s PyPTO wrapper."""
    worker_dict = getattr(worker, "__dict__", None)
    owner_ref = worker_dict.get(_PYPTO_OWNER_REF_ATTR) if isinstance(worker_dict, dict) else None
    if not isinstance(owner_ref, weakref.ReferenceType):
        raise TypeError(
            "DeviceTensor dispatch requires the owning PyPTO Worker; a one-shot or raw simpler "
            "Worker cannot prove that the retained Buffer is live. Dispatch through the same "
            "PyPTO Worker that allocated the tensor."
        )
    owner = owner_ref()
    if owner is None:
        raise ValueError("DeviceTensor's owning PyPTO Worker no longer exists.")
    if getattr(owner, "_tensor_arg_worker", None) is not worker:
        raise ValueError(
            "DeviceTensor dispatch received a stale simpler Worker backend; use the owning PyPTO "
            "Worker's current initialized backend."
        )
    owner._require_owned_resident_tensor(arg, "Tensor argument")


def _enum_value(value: Any) -> int | str:
    """Return a stable scalar for a Python enum or an already numeric field."""
    raw = getattr(value, "value", value)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return str(raw)


def _buffer_signature(buffer: Any) -> tuple[Any, ...] | object:
    """Describe a simpler BufferDescriptor without relying on wrapper identity."""
    identity = getattr(buffer, "identity", None)
    if identity is None:
        return _UNSUPPORTED
    try:
        identity_sig = (
            bytes(identity.owner_instance_id),
            int(identity.buffer_id),
            int(identity.generation),
        )
        return (
            identity_sig,
            _enum_value(buffer.address_space),
            _enum_value(buffer.access),
            _enum_value(buffer.backend_kind),
            int(buffer.nbytes),
            int(buffer.owner_worker_path_id),
            bytes(buffer.body),
        )
    except (AttributeError, TypeError, ValueError):
        return _UNSUPPORTED


def _task_arg_identity(value: Any, seen: set[int]) -> tuple[Any, ...] | object:
    """Build a descriptor-only signature; payload contents are intentionally omitted."""
    if isinstance(value, torch.Tensor):
        try:
            storage = value.untyped_storage()
            storage_nbytes = storage.nbytes()
            return (
                "torch",
                value.device.type,
                value.device.index,
                int(storage.data_ptr()),
                int(storage_nbytes),
                int(value.data_ptr()),
                tuple(int(dim) for dim in value.shape),
                tuple(int(stride) for stride in value.stride()),
                int(value.storage_offset()),
                str(value.dtype),
                bool(value.is_shared()),
            )
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return _UNSUPPORTED

    if value is None or isinstance(value, (bool, int, str, bytes)):
        return (type(value).__module__, type(value).__qualname__, value)
    if isinstance(value, float):
        return (
            type(value).__module__,
            type(value).__qualname__,
            float(value).hex() if not math.isnan(value) else "nan",
        )

    if isinstance(value, (list, tuple)):
        marker = id(value)
        if marker in seen:
            return _UNSUPPORTED
        seen.add(marker)
        try:
            children = tuple(_task_arg_identity(item, seen) for item in value)
            if any(child is _UNSUPPORTED for child in children):
                return _UNSUPPORTED
            return (type(value).__name__, children)
        finally:
            seen.remove(marker)

    # Avoid importing the optional runtime stack for arbitrary user objects.
    # Known wire-wrapper classes live under these modules; everything else is
    # unsupported and keeps caching fail-open.
    value_module = type(value).__module__
    if not (
        value_module.startswith("pypto.runtime.")
        or value_module.startswith("simpler.")
        or value_module == "_task_interface"
    ):
        return _UNSUPPORTED

    task_interface, device_tensor, _torch_interop = _modules()
    stacked_cls = getattr(device_tensor, "StackedDeviceTensor", None)
    if stacked_cls is not None and isinstance(value, stacked_cls):
        marker = id(value)
        if marker in seen:
            return _UNSUPPORTED
        seen.add(marker)
        try:
            shards = tuple(_task_arg_identity(shard, seen) for shard in value.shards)
            if any(shard is _UNSUPPORTED for shard in shards):
                return _UNSUPPORTED
            return (
                "stacked",
                tuple(int(dim) for dim in value.full_shape),
                tuple(int(worker) for worker in value.worker_ids),
                str(value.dtype),
                shards,
            )
        except (AttributeError, TypeError, ValueError):
            return _UNSUPPORTED
        finally:
            seen.remove(marker)

    device_cls = getattr(device_tensor, "DeviceTensor", None)
    if device_cls is not None and isinstance(value, device_cls):
        try:
            buffer = value.buffer
            if buffer is None:
                return _UNSUPPORTED
            buffer_sig = _buffer_signature(buffer)
            if buffer_sig is _UNSUPPORTED:
                return _UNSUPPORTED
            return (
                "device",
                int(value.data_ptr),
                tuple(int(dim) for dim in value.shape),
                str(value.dtype),
                buffer_sig,
                bool(getattr(buffer, "closed", False)),
                int(buffer.base),
            )
        except (AttributeError, TypeError, ValueError):
            return _UNSUPPORTED

    if isinstance(value, task_interface.Tensor):
        try:
            buffer_sig = _buffer_signature(value.buffer)
            if buffer_sig is _UNSUPPORTED:
                return _UNSUPPORTED
            return (
                "simpler_tensor",
                buffer_sig,
                int(value.byte_offset),
                int(value.ndims),
                tuple(int(dim) for dim in value.shapes),
                tuple(int(stride) for stride in value.strides),
                int(value.dtype),
            )
        except (AttributeError, TypeError, ValueError):
            return _UNSUPPORTED

    return _UNSUPPORTED


def _task_args_signature(values: Any) -> tuple[Any, ...] | None:
    """Return a bounded, descriptor-only signature or ``None`` to disable caching."""
    try:
        items = tuple(values)
        signature = tuple(_task_arg_identity(value, set()) for value in items)
        if any(item is _UNSUPPORTED for item in signature):
            return None
        return signature
    except Exception:  # noqa: BLE001 - optimization must fail open
        return None


def _task_args_signature_for_cache(cache: Any, values: Any) -> tuple[Any, ...] | None:
    """Compute a signature only when a persistent cache is enabled."""
    if cache is None:
        return None
    return _task_args_signature(values)


def _task_args_cache_for_orch(orch: Any, provider: Any) -> dict[Any, tuple[Any, Any]] | None:
    """Return the persistent cache for a provider-backed orchestrator invocation."""
    if provider is None:
        return None
    cache = getattr(orch, "_pypto_task_args_cache_v1", None)
    if cache is None:
        cache = {}
        orch._pypto_task_args_cache_v1 = cache
    return cache


def _task_args_cache_store(
    cache: dict[Any, tuple[Any, Any]], slot: Any, signature: Any, task_args: Any
) -> None:
    """Store one TaskArgs entry while keeping the persistent cache hard-bounded."""
    try:
        if slot not in cache and len(cache) >= _PYPTO_TASK_ARGS_CACHE_MAX_ENTRIES:
            cache.pop(next(iter(cache)))
        cache[slot] = (signature, task_args)
    except Exception:  # noqa: BLE001 - cache failures must not block dispatch
        return


@cache
def _modules() -> tuple[Any, Any, Any]:
    """Import and cache runtime modules on first ``make_tensor_arg`` call.

    The imports stay inside this function so importing pypto never requires
    simpler (only available in the runtime environment). ``functools.cache``
    runs the body once — instead of on every call — which matters because the
    generated ``host_orch`` calls ``make_tensor_arg`` once per tensor per rank
    (~90 tensors × world_size), where per-call ``from ... import`` was pure
    overhead on the host dispatch loop.

    Only the *module objects* are cached; individual symbols are resolved via
    attribute access on every call. This keeps the helper responsive to test
    monkeypatches while still paying the import cost only once.

    Returns:
        ``(task_interface, device_tensor, torch_interop)`` module objects.
    """
    from simpler_setup import torch_interop  # pyright: ignore[reportMissingImports]  # noqa: PLC0415

    from . import device_tensor, task_interface  # noqa: PLC0415

    return task_interface, device_tensor, torch_interop


def make_tensor_arg(worker: Any, arg: Any) -> Any:
    """Convert an orchestration tensor argument into a simpler ``Tensor``.

    Args:
        worker: The raw simpler Worker performing dispatch. DeviceTensor
            conversion requires it to be bound to the same live PyPTO Worker
            that allocated the Buffer; host tensors use it directly for
            simpler's worker-aware conversion.
        arg: One of:
            - ``torch.Tensor``: a CPU-contiguous host tensor (delegated to
              simpler's worker-aware wire helper).
            - :class:`~pypto.runtime.DeviceTensor`: a worker-resident buffer;
              after owner/liveness validation, its retained ``Buffer``
              constructs an address-free wire ``Tensor`` (memory is
              caller-managed).
            - simpler ``Tensor``: returned as-is (already device-side).

    Returns:
        A simpler ``Tensor`` ready to add to ``TaskArgs``.
    """
    task_interface, device_tensor, torch_interop = _modules()

    if isinstance(arg, task_interface.Tensor):
        return arg
    if isinstance(arg, device_tensor.DeviceTensor):
        if arg.buffer is None:
            raise TypeError(
                "A raw-pointer DeviceTensor cannot cross the public Worker.run wire ABI; "
                "allocate it with Worker.alloc_tensor() so it retains its simpler Buffer."
            )
        _require_device_tensor_owner(worker, arg)
        try:
            dtype = task_interface.torch_dtype_to_datatype(arg.dtype)
        except KeyError as e:
            raise ValueError(f"Unsupported DeviceTensor dtype: {arg.dtype}") from e
        return arg.buffer.tensor(shapes=arg.shape, dtype=dtype)
    return torch_interop.make_tensor_arg(worker, arg)
