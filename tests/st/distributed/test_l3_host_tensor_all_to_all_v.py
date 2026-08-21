# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""L3 distributed ST: host-orchestrator ``pld.tensor.all_to_all_v`` builtin dispatch.

Validates the HOST-level variable-size all-to-all (MPI_Alltoallv pattern) lowers
through ``LowerHostTensorCollectives`` and produces correct rank-ordered
personalized exchange via the hand-written ``builtin.tensor.all_to_all_v``
kernel — the HOST-orchestrated counterpart of the InCore-only
``test_l3_tensor_all_to_all_v_intrinsic.py``.

The HOST lowering path detects ``pld.tensor.all_to_all_v`` in ``host_orch`` and
lowers it to ``builtin.tensor.all_to_all_v`` per chip. The exchange uses a
push-based TPUT pattern with FIVE window-bound resources:

  1. **Stage** (``stage_step``): each rank writes its per-destination chunks
     into ``input_buf`` — a window used ONLY as a TPUT source, never as an
     incoming-push destination (same discipline as symmetric all_to_all's HOST
     builtin).
  2. **Fill counts** (``fill_counts_step``): each rank writes its own
     per-destination send counts into ``counts_buf``. This staging step only
     exists because the HOST builtin narrows ``send_counts`` from the InCore
     composite's ``Tensor``-or-``DistributedTensor`` contract down to a strict
     window-bound ``DistributedTensor`` — ``EmitBuiltinWindowCollectiveDispatch``
     has no dispatch path for a plain ``Tensor`` arg. A real ergonomic cost of
     the narrowing, not a test artifact.
  3. **All-to-all-v** (``builtin.tensor.all_to_all_v``): the kernel pushes the
     full ``MAX_RECV``-row capacity block per destination into ``data_buf``,
     publishes the clamped ``min(send_counts[dest], MAX_RECV)`` into peer
     ``recv_counts[my_rank, 0]`` via TNOTIFY, and synchronises with one
     barrier.
  4. **Consume** (``consume_step``): each rank reads ``recv_counts`` to learn
     how many rows each source actually sent, then reads back only those valid
     rows from ``data_buf``.

``input_buf``, ``data_buf``, ``signal_buf``, ``counts_buf``, and ``recv_buf``
must all be allocated inside the SAME host_orch comm-domain scope —
``EmitBuiltinWindowCollectiveDispatch`` requires every window-bound arg of one
builtin call to share a comm-domain handle.

The HOST kernel derives ``MAX_RECV`` at entry from the runtime rank count
(``target.shape[0] / nranks``), so the per-destination block size is always
consistent with the devices actually running. The program is still built via
a factory function per parametrized rank count (signal/counts shapes are
per-rank-count), matching ``test_l3_tensor_all_to_all_v_intrinsic.py``'s
pattern.

ST coverage: P=2 and P=4 (skips when fewer devices are available).
"""

import sys

import pypto.language as pl
import pypto.language.distributed as pld
import pytest
import torch
from pypto import ir
from pypto.ir.distributed_compiled_program import DistributedConfig

SIZE = 64
MAX_RECV = 4


def _build_host_all_to_all_v_program(n_ranks: int, max_recv: int):
    """Build an N-rank HOST-orchestrated variable-size all-to-all program."""
    nr = n_ranks
    mr = max_recv
    total = nr * mr

    @pl.program
    class HostTensorAllToAllV:
        """N-rank HOST-orchestrated variable-size all-to-all program."""

        @pl.function(type=pl.FunctionType.InCore)
        def stage_step(
            self,
            inp: pl.Tensor[[total, SIZE], pl.FP32],
            stage: pl.Out[pld.DistributedTensor[[total, SIZE], pl.FP32]],
        ):
            for row in pl.range(total):
                chunk = pl.load(inp, [row, 0], [1, SIZE])
                stage = pl.store(chunk, [row, 0], stage)

        @pl.function(type=pl.FunctionType.Orchestration)
        def stage_orch(
            self,
            inp: pl.Tensor[[total, SIZE], pl.FP32],
            stage: pl.Out[pld.DistributedTensor[[total, SIZE], pl.FP32]],
        ):
            self.stage_step(inp, stage)

        @pl.function(type=pl.FunctionType.InCore)
        def fill_counts_step(
            self,
            counts_row: pl.Tensor[[nr, 1], pl.INT32],
            counts: pl.Out[pld.DistributedTensor[[nr, 1], pl.INT32]],
        ):
            # Scalar read/write — a [1,1] INT32 tile.load/store fails ptoas
            # 32-byte row alignment (4 bytes); same pitfall the InCore
            # all_to_all_v intrinsic ST avoids for recv_counts.
            for d in pl.range(nr):
                v = pl.read(counts_row, [d, 0])
                pl.write(counts, [d, 0], v)

        @pl.function(type=pl.FunctionType.Orchestration)
        def fill_counts_orch(
            self,
            counts_row: pl.Tensor[[nr, 1], pl.INT32],
            counts: pl.Out[pld.DistributedTensor[[nr, 1], pl.INT32]],
        ):
            self.fill_counts_step(counts_row, counts)

        @pl.function(type=pl.FunctionType.InCore)
        def consume_step(
            self,
            data: pld.DistributedTensor[[total, SIZE], pl.FP32],
            recv_counts: pld.DistributedTensor[[nr, 1], pl.INT32],
            out: pl.Out[pl.Tensor[[total, SIZE], pl.FP32]],
            recv_out: pl.Out[pl.Tensor[[nr, 1], pl.INT32]],
        ) -> tuple[pl.Tensor[[total, SIZE], pl.FP32], pl.Tensor[[nr, 1], pl.INT32]]:
            for src in pl.range(nr):
                n_rows_i32 = pl.read(recv_counts, [src, 0])
                pl.write(recv_out, [src, 0], n_rows_i32)
                n_rows = pl.cast(n_rows_i32, pl.INDEX)
                base = src * mr
                for r in pl.range(n_rows):
                    flat_row = base + r
                    chunk = pl.load(data, [flat_row, 0], [1, SIZE])
                    out = pl.store(chunk, [flat_row, 0], out)
            return out, recv_out

        @pl.function(type=pl.FunctionType.Orchestration)
        def consume_orch(
            self,
            data: pld.DistributedTensor[[total, SIZE], pl.FP32],
            recv_counts: pld.DistributedTensor[[nr, 1], pl.INT32],
            out: pl.Out[pl.Tensor[[total, SIZE], pl.FP32]],
            recv_out: pl.Out[pl.Tensor[[nr, 1], pl.INT32]],
        ) -> tuple[pl.Tensor[[total, SIZE], pl.FP32], pl.Tensor[[nr, 1], pl.INT32]]:
            return self.consume_step(data, recv_counts, out, recv_out)

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(
            self,
            inputs: pl.Tensor[[nr, total, SIZE], pl.FP32],
            send_counts: pl.Tensor[[nr, nr, 1], pl.INT32],
            outputs: pl.Out[pl.Tensor[[nr, total, SIZE], pl.FP32]],
            recv_outputs: pl.Out[pl.Tensor[[nr, nr, 1], pl.INT32]],
        ) -> tuple[pl.Tensor[[nr, total, SIZE], pl.FP32], pl.Tensor[[nr, nr, 1], pl.INT32]]:
            input_buf = pld.alloc_window_buffer(total * SIZE * pl.FP32.get_byte())
            data_buf = pld.alloc_window_buffer(total * SIZE * pl.FP32.get_byte())
            signal_buf = pld.alloc_window_buffer(nr * pl.INT32.get_byte())
            counts_buf = pld.alloc_window_buffer(nr * pl.INT32.get_byte())
            recv_buf = pld.alloc_window_buffer(nr * pl.INT32.get_byte())

            for r in pl.range(pld.world_size()):
                stage = pld.window(input_buf, [total, SIZE], dtype=pl.FP32)
                self.stage_orch(inputs[r], stage, device=r)

            for r in pl.range(pld.world_size()):
                counts = pld.window(counts_buf, [nr, 1], dtype=pl.INT32)
                self.fill_counts_orch(send_counts[r], counts, device=r)

            stage = pld.window(input_buf, [total, SIZE], dtype=pl.FP32)
            data = pld.window(data_buf, [total, SIZE], dtype=pl.FP32)
            signal = pld.window(signal_buf, [nr, 1], dtype=pl.INT32)
            counts = pld.window(counts_buf, [nr, 1], dtype=pl.INT32)
            recv = pld.window(recv_buf, [nr, 1], dtype=pl.INT32)
            data = pld.tensor.all_to_all_v(stage, data, signal, counts, recv)

            for r in pl.range(pld.world_size()):
                self.consume_orch(data, recv, outputs[r], recv_outputs[r], device=r)

            return outputs, recv_outputs

    return HostTensorAllToAllV


class TestL3HostTensorAllToAllV:
    """L3 distributed runtime: HOST-level variable-size all-to-all via builtin dispatch."""

    @pytest.mark.parametrize("n_ranks", [2, 4])
    def test_host_tensor_all_to_all_v(self, test_config, device_ids, n_ranks):
        """Compile and run host-level all_to_all_v for P in {2, 4}.

        Each rank sends ``n_ranks - dest`` rows to each destination (variable,
        runtime-dependent counts) — same golden pattern as the InCore-only ST
        (``test_l3_tensor_all_to_all_v_intrinsic.py``).
        """
        if len(device_ids) < n_ranks:
            pytest.skip(f"host all_to_all_v P={n_ranks} needs {n_ranks} devices, got {device_ids}")

        nr = n_ranks
        mr = MAX_RECV
        total = nr * mr

        program = _build_host_all_to_all_v_program(nr, mr)
        compiled = ir.compile(
            program,
            platform=test_config.platform,
            distributed_config=DistributedConfig(
                device_ids=device_ids[:nr],
                num_sub_workers=0,
            ),
        )

        variant_dir = compiled.output_dir / "next_levels" / "builtin.tensor.all_to_all_v__fp32"
        assert variant_dir.is_dir(), f"expected {variant_dir}"
        assert (variant_dir / "kernel_config.py").is_file()

        # Rank r sends to dest d: rows dest*mr+k for k=0..n_rows-1.
        # Value = r*1000 + d*100 + k*10 + j%10 (same formula as the InCore ST).
        # The TPUT transfers the full per-destination capacity block (max_recv
        # rows, derived at kernel entry from the runtime nranks); rows beyond
        # n_rows are sent as well — the receiver uses recv_counts to skip the
        # unwritten window holes.
        inputs = torch.zeros((nr, total, SIZE), dtype=torch.float32)
        send_counts = torch.zeros((nr, nr, 1), dtype=torch.int32)
        for r in range(nr):
            for d in range(nr):
                n_rows = nr - d  # variable send count, same golden formula as the InCore ST
                send_counts[r, d, 0] = n_rows
                base = d * mr
                for k in range(n_rows):
                    for j in range(SIZE):
                        inputs[r, base + k, j] = float(r * 1000 + d * 100 + k * 10 + j % 10)

        outputs = torch.zeros((nr, total, SIZE), dtype=torch.float32)
        recv_outputs = torch.zeros((nr, nr, 1), dtype=torch.int32)

        compiled(inputs, send_counts, outputs, recv_outputs)

        # Rank rank receives from src the chunk that src sent to dest=rank.
        for rank in range(nr):
            for src in range(nr):
                n_rows = int(send_counts[src, rank, 0].item())
                assert int(recv_outputs[rank, src, 0].item()) == n_rows, (
                    f"P={nr} rank={rank} src={src}: recv_counts="
                    f"{int(recv_outputs[rank, src, 0].item())} != expected send_counts={n_rows}"
                )
                base = src * mr
                for k in range(n_rows):
                    expected_row = inputs[src, rank * mr + k, :]
                    got_row = outputs[rank, base + k, :]
                    assert torch.allclose(got_row, expected_row, atol=1e-5), (
                        f"P={nr} rank={rank} src={src} row={k}: "
                        f"max diff = {(got_row - expected_row).abs().max().item()}"
                    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", *sys.argv[1:]])
