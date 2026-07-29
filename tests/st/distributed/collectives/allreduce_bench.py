#!/usr/bin/env python
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Standalone A/B microbenchmark for step3p5 TP all-reduce algorithm variants.

Isolates a single cross-rank all-reduce of a ``[BATCH, HIDDEN]`` BF16 tensor
(step3p5 hidden shape) at a configurable rank count, so the collective's cost
can be measured without the whole-network harness. All variants are
raw-primitive hand-written kernels (``pl.load``/``pl.store``/``pl.add`` +
``pld.tile.remote_load`` + ``pld.system.notify``/``wait``), FP32 accumulation
to mirror the model's precision handling.

Variants (``--mode``):
  * ``onephase`` — naive symmetric mesh (step3p5 today): each rank reads the
    FULL vector from every peer. Remote data/rank = (P-1)*N. 1 mesh barrier.
  * ``twophase`` — mesh reduce-scatter + all-gather: each rank reduces only its
    owned N/P chunk, then gathers the reduced chunks. Remote ~ 2N/P*(P-1).
    2 mesh barriers.
  * ``ring``     — NCCL-style chunked reduce-scatter + all-gather ring.
    Remote ~ 2(P-1)/P*N. 2(P-1) mesh barriers (one per round).

Each algorithm is its own ``@pl.program`` (the tracer walks all branches of an
in-body ``if`` and forbids arbitrary Python-function calls, so the algorithm
must be inlined per-class rather than selected at runtime).

Golden: ``output[r] == sum(inputs[*])`` on every rank. Inputs are small ints so
BF16 round-trips exactly and the golden sum is exact.

Usage (device host, e.g. 0162):
    python allreduce_bench.py -p a2a3   -d 0-7  --mode twophase --iters 50
    python allreduce_bench.py -p a2a3sim -d 0-1 --mode ring     # sim smoke
"""

# pyright: reportUndefinedVariable=false

import argparse
import sys
import time

import pypto.language as pl
import pypto.language.distributed as pld
import torch
from pypto import ir
from pypto.ir.distributed_compiled_program import DistributedConfig

BATCH = 16
HIDDEN = 4096
# Column chunk for the onephase FP32 accumulator: [BATCH, COL_CHUNK] FP32 must
# fit the ~188KB UB. HIDDEN//8 = 512 -> 16*512*4 = 32KB. Matches the model.
COL_CHUNK = HIDDEN // 8

_MODES = ("onephase", "onephase_par", "twophase", "twophase_par", "ring", "pld_mesh", "pld_ring")


def _build_onephase(n_ranks: int):
    sig_rows, sig_cols = n_ranks, 1

    @pl.program
    class AllReduceOnephase:
        @pl.function(type=pl.FunctionType.InCore)
        def reduce_step(
            self,
            inp: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
            out: pl.Out[pl.Tensor[[BATCH, HIDDEN], pl.BF16]],
            data: pl.InOut[pld.DistributedTensor[[BATCH, HIDDEN], pl.BF16]],
            signal: pl.InOut[pld.DistributedTensor[[sig_rows, sig_cols], pl.INT32]],
            my_rank: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[BATCH, HIDDEN], pl.BF16]:
            for k0 in pl.range(0, HIDDEN, COL_CHUNK):
                pl.store(pl.load(inp, [0, k0], [BATCH, COL_CHUNK]), [0, k0], data)
            for peer in pl.range(n_ranks):
                if peer != my_rank:
                    pld.system.notify(signal, peer=peer, offsets=[my_rank, 0],
                                      value=1, op=pld.NotifyOp.AtomicAdd)
            for src in pl.range(n_ranks):
                if src != my_rank:
                    pld.system.wait(signal=signal, offsets=[src, 0],
                                    expected=1, cmp=pld.WaitCmp.Ge)
            for k0 in pl.range(0, HIDDEN, COL_CHUNK):
                own = pl.load(data, [0, k0], [BATCH, COL_CHUNK])
                acc = pl.cast(own, target_type=pl.FP32)
                for peer in pl.range(n_ranks):
                    if peer != my_rank:
                        recv = pld.tile.remote_load(data, peer=peer, offsets=[0, k0],
                                                    shape=[BATCH, COL_CHUNK])
                        acc = pl.add(acc, pl.cast(recv, target_type=pl.FP32))
                pl.store(pl.cast(acc, target_type=pl.BF16), [0, k0], out)
            return out

        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(
            self,
            inp: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
            out: pl.Out[pl.Tensor[[BATCH, HIDDEN], pl.BF16]],
            data: pl.InOut[pld.DistributedTensor[[BATCH, HIDDEN], pl.BF16]],
            signal: pl.InOut[pld.DistributedTensor[[sig_rows, sig_cols], pl.INT32]],
            my_rank: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[BATCH, HIDDEN], pl.BF16]:
            return self.reduce_step(inp, out, data, signal, my_rank)

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(
            self,
            inputs: pl.Tensor[[n_ranks, BATCH, HIDDEN], pl.BF16],
            outputs: pl.Out[pl.Tensor[[n_ranks, BATCH, HIDDEN], pl.BF16]],
        ) -> pl.Tensor[[n_ranks, BATCH, HIDDEN], pl.BF16]:
            data_buf = pld.alloc_window_buffer(BATCH * HIDDEN * 2)  # BF16 = 2 bytes
            signal_buf = pld.alloc_window_buffer(sig_rows * sig_cols * 4)  # INT32 = 4 bytes
            for r in pl.range(pld.world_size()):
                data = pld.window(data_buf, [BATCH, HIDDEN], dtype=pl.BF16)
                signal = pld.window(signal_buf, [sig_rows, sig_cols], dtype=pl.INT32)
                self.chip_orch(inputs[r], outputs[r], data, signal, r, device=r)
            return outputs

    return AllReduceOnephase


def _build_twophase(n_ranks: int):
    sig_rows, sig_cols = 2, n_ranks
    chunk = HIDDEN // n_ranks

    @pl.program
    class AllReduceTwophase:
        @pl.function(type=pl.FunctionType.InCore)
        def reduce_step(
            self,
            inp: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
            out: pl.Out[pl.Tensor[[BATCH, HIDDEN], pl.BF16]],
            data: pl.InOut[pld.DistributedTensor[[BATCH, HIDDEN], pl.BF16]],
            signal: pl.InOut[pld.DistributedTensor[[sig_rows, sig_cols], pl.INT32]],
            my_rank: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[BATCH, HIDDEN], pl.BF16]:
            for k0 in pl.range(0, HIDDEN, COL_CHUNK):
                pl.store(pl.load(inp, [0, k0], [BATCH, COL_CHUNK]), [0, k0], data)
            # RS barrier (row 0).
            for peer in pl.range(n_ranks):
                if peer != my_rank:
                    pld.system.notify(signal, peer=peer, offsets=[0, my_rank],
                                      value=1, op=pld.NotifyOp.AtomicAdd)
            for src in pl.range(n_ranks):
                if src != my_rank:
                    pld.system.wait(signal=signal, offsets=[0, src],
                                    expected=1, cmp=pld.WaitCmp.Ge)
            # Reduce-scatter: reduce ONLY my owned chunk from all peers.
            base = my_rank * chunk
            own = pl.load(data, [0, base], [BATCH, chunk])
            acc = pl.cast(own, target_type=pl.FP32)
            for peer in pl.range(n_ranks):
                if peer != my_rank:
                    recv = pld.tile.remote_load(data, peer=peer, offsets=[0, base],
                                                shape=[BATCH, chunk])
                    acc = pl.add(acc, pl.cast(recv, target_type=pl.FP32))
            reduced = pl.cast(acc, target_type=pl.BF16)
            pl.store(reduced, [0, base], data)
            # AG barrier (row 1).
            for peer in pl.range(n_ranks):
                if peer != my_rank:
                    pld.system.notify(signal, peer=peer, offsets=[1, my_rank],
                                      value=1, op=pld.NotifyOp.AtomicAdd)
            for src in pl.range(n_ranks):
                if src != my_rank:
                    pld.system.wait(signal=signal, offsets=[1, src],
                                    expected=1, cmp=pld.WaitCmp.Ge)
            # All-gather: read every peer r's reduced chunk r into out.
            for r in pl.range(n_ranks):
                off = r * chunk
                if r != my_rank:
                    red = pld.tile.remote_load(data, peer=r, offsets=[0, off],
                                               shape=[BATCH, chunk])
                    pl.store(red, [0, off], out)
                else:
                    pl.store(reduced, [0, off], out)
            return out

        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(
            self,
            inp: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
            out: pl.Out[pl.Tensor[[BATCH, HIDDEN], pl.BF16]],
            data: pl.InOut[pld.DistributedTensor[[BATCH, HIDDEN], pl.BF16]],
            signal: pl.InOut[pld.DistributedTensor[[sig_rows, sig_cols], pl.INT32]],
            my_rank: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[BATCH, HIDDEN], pl.BF16]:
            return self.reduce_step(inp, out, data, signal, my_rank)

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(
            self,
            inputs: pl.Tensor[[n_ranks, BATCH, HIDDEN], pl.BF16],
            outputs: pl.Out[pl.Tensor[[n_ranks, BATCH, HIDDEN], pl.BF16]],
        ) -> pl.Tensor[[n_ranks, BATCH, HIDDEN], pl.BF16]:
            data_buf = pld.alloc_window_buffer(BATCH * HIDDEN * 2)  # BF16 = 2 bytes
            signal_buf = pld.alloc_window_buffer(sig_rows * sig_cols * 4)  # INT32 = 4 bytes
            for r in pl.range(pld.world_size()):
                data = pld.window(data_buf, [BATCH, HIDDEN], dtype=pl.BF16)
                signal = pld.window(signal_buf, [sig_rows, sig_cols], dtype=pl.INT32)
                self.chip_orch(inputs[r], outputs[r], data, signal, r, device=r)
            return outputs

    return AllReduceTwophase


def _build_ring(n_ranks: int):
    sig_rows, sig_cols = 2 * (n_ranks - 1), n_ranks
    chunk = HIDDEN // n_ranks

    @pl.program
    class AllReduceRing:
        @pl.function(type=pl.FunctionType.InCore)
        def reduce_step(
            self,
            inp: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
            out: pl.Out[pl.Tensor[[BATCH, HIDDEN], pl.BF16]],
            data: pl.InOut[pld.DistributedTensor[[BATCH, HIDDEN], pl.BF16]],
            signal: pl.InOut[pld.DistributedTensor[[sig_rows, sig_cols], pl.INT32]],
            my_rank: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[BATCH, HIDDEN], pl.BF16]:
            for k0 in pl.range(0, HIDDEN, COL_CHUNK):
                pl.store(pl.load(inp, [0, k0], [BATCH, COL_CHUNK]), [0, k0], data)
            left = (my_rank - 1 + n_ranks) % n_ranks
            # Reduce-scatter: (P-1) rounds, mesh barrier per round.
            for s in pl.range(n_ranks - 1):
                step = s + 1
                recv_add_idx = (my_rank - step - 1 + n_ranks) % n_ranks
                left_send_idx = (left - step + n_ranks) % n_ranks
                rs_round = s
                for peer in pl.range(n_ranks):
                    if peer != my_rank:
                        pld.system.notify(signal, peer=peer, offsets=[rs_round, my_rank],
                                          value=1, op=pld.NotifyOp.AtomicAdd)
                for peer in pl.range(n_ranks):
                    if peer != my_rank:
                        pld.system.wait(signal=signal, offsets=[rs_round, peer],
                                        expected=1, cmp=pld.WaitCmp.Ge)
                recv = pld.tile.remote_load(data, peer=left,
                                            offsets=[0, left_send_idx * chunk],
                                            shape=[BATCH, chunk])
                own = pl.load(data, [0, recv_add_idx * chunk], [BATCH, chunk])
                acc = pl.cast(own, target_type=pl.FP32)
                acc = pl.add(acc, pl.cast(recv, target_type=pl.FP32))
                pl.store(pl.cast(acc, target_type=pl.BF16),
                         [0, recv_add_idx * chunk], data)
            # All-gather: (P-1) rounds, mesh barrier per round.
            for s in pl.range(n_ranks - 1):
                step = s + 1
                recv_idx = (my_rank - step + n_ranks) % n_ranks
                left_send_idx = (left - step + 1 + n_ranks) % n_ranks
                ag_round = (n_ranks - 1) + s
                for peer in pl.range(n_ranks):
                    if peer != my_rank:
                        pld.system.notify(signal, peer=peer, offsets=[ag_round, my_rank],
                                          value=1, op=pld.NotifyOp.AtomicAdd)
                for peer in pl.range(n_ranks):
                    if peer != my_rank:
                        pld.system.wait(signal=signal, offsets=[ag_round, peer],
                                        expected=1, cmp=pld.WaitCmp.Ge)
                recv = pld.tile.remote_load(data, peer=left,
                                            offsets=[0, left_send_idx * chunk],
                                            shape=[BATCH, chunk])
                pl.store(recv, [0, recv_idx * chunk], data)
            for k0 in pl.range(0, HIDDEN, COL_CHUNK):
                pl.store(pl.load(data, [0, k0], [BATCH, COL_CHUNK]), [0, k0], out)
            return out

        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(
            self,
            inp: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
            out: pl.Out[pl.Tensor[[BATCH, HIDDEN], pl.BF16]],
            data: pl.InOut[pld.DistributedTensor[[BATCH, HIDDEN], pl.BF16]],
            signal: pl.InOut[pld.DistributedTensor[[sig_rows, sig_cols], pl.INT32]],
            my_rank: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[BATCH, HIDDEN], pl.BF16]:
            return self.reduce_step(inp, out, data, signal, my_rank)

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(
            self,
            inputs: pl.Tensor[[n_ranks, BATCH, HIDDEN], pl.BF16],
            outputs: pl.Out[pl.Tensor[[n_ranks, BATCH, HIDDEN], pl.BF16]],
        ) -> pl.Tensor[[n_ranks, BATCH, HIDDEN], pl.BF16]:
            data_buf = pld.alloc_window_buffer(BATCH * HIDDEN * 2)  # BF16 = 2 bytes
            signal_buf = pld.alloc_window_buffer(sig_rows * sig_cols * 4)  # INT32 = 4 bytes
            for r in pl.range(pld.world_size()):
                data = pld.window(data_buf, [BATCH, HIDDEN], dtype=pl.BF16)
                signal = pld.window(signal_buf, [sig_rows, sig_cols], dtype=pl.INT32)
                self.chip_orch(inputs[r], outputs[r], data, signal, r, device=r)
            return outputs

    return AllReduceRing


def _build_onephase_par(n_ranks: int):
    """onephase mesh, but parallelized: the 8 column-chunks reduce concurrently
    (pl.parallel over k0, each core does one chunk's serial 7-peer reduce) and
    the notify/wait barrier fan-out is pl.parallel.
    """
    sig_rows, sig_cols = n_ranks, 1

    @pl.program
    class AllReduceOnephasePar:
        @pl.function(type=pl.FunctionType.InCore)
        def reduce_step(
            self,
            inp: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
            out: pl.Out[pl.Tensor[[BATCH, HIDDEN], pl.BF16]],
            data: pl.InOut[pld.DistributedTensor[[BATCH, HIDDEN], pl.BF16]],
            signal: pl.InOut[pld.DistributedTensor[[sig_rows, sig_cols], pl.INT32]],
            my_rank: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[BATCH, HIDDEN], pl.BF16]:
            for k0 in pl.parallel(0, HIDDEN, COL_CHUNK):
                pl.store(pl.load(inp, [0, k0], [BATCH, COL_CHUNK]), [0, k0], data)
            for peer in pl.parallel(n_ranks):
                if peer != my_rank:
                    pld.system.notify(signal, peer=peer, offsets=[my_rank, 0],
                                      value=1, op=pld.NotifyOp.AtomicAdd)
            for src in pl.parallel(n_ranks):
                if src != my_rank:
                    pld.system.wait(signal=signal, offsets=[src, 0],
                                    expected=1, cmp=pld.WaitCmp.Ge)
            for k0 in pl.parallel(0, HIDDEN, COL_CHUNK):
                own = pl.load(data, [0, k0], [BATCH, COL_CHUNK])
                acc = pl.cast(own, target_type=pl.FP32)
                for peer in pl.range(n_ranks):
                    if peer != my_rank:
                        recv = pld.tile.remote_load(data, peer=peer, offsets=[0, k0],
                                                    shape=[BATCH, COL_CHUNK])
                        acc = pl.add(acc, pl.cast(recv, target_type=pl.FP32))
                pl.store(pl.cast(acc, target_type=pl.BF16), [0, k0], out)
            return out

        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(
            self,
            inp: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
            out: pl.Out[pl.Tensor[[BATCH, HIDDEN], pl.BF16]],
            data: pl.InOut[pld.DistributedTensor[[BATCH, HIDDEN], pl.BF16]],
            signal: pl.InOut[pld.DistributedTensor[[sig_rows, sig_cols], pl.INT32]],
            my_rank: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[BATCH, HIDDEN], pl.BF16]:
            return self.reduce_step(inp, out, data, signal, my_rank)

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(
            self,
            inputs: pl.Tensor[[n_ranks, BATCH, HIDDEN], pl.BF16],
            outputs: pl.Out[pl.Tensor[[n_ranks, BATCH, HIDDEN], pl.BF16]],
        ) -> pl.Tensor[[n_ranks, BATCH, HIDDEN], pl.BF16]:
            data_buf = pld.alloc_window_buffer(BATCH * HIDDEN * 2)
            signal_buf = pld.alloc_window_buffer(sig_rows * sig_cols * 4)
            for r in pl.range(pld.world_size()):
                data = pld.window(data_buf, [BATCH, HIDDEN], dtype=pl.BF16)
                signal = pld.window(signal_buf, [sig_rows, sig_cols], dtype=pl.INT32)
                self.chip_orch(inputs[r], outputs[r], data, signal, r, device=r)
            return outputs

    return AllReduceOnephasePar


def _build_twophase_par(n_ranks: int):
    """twophase, but the independent loops use pl.parallel (fan-out across cores).

    Parallelizes: stage-in, the RS/AG notify+wait barrier fan-outs, and the
    all-gather peer stores — all iteration-independent. The reduce-scatter
    accumulate stays pl.range (carried FP32 reduction).
    """
    sig_rows, sig_cols = 2, n_ranks
    chunk = HIDDEN // n_ranks

    @pl.program
    class AllReduceTwophasePar:
        @pl.function(type=pl.FunctionType.InCore)
        def reduce_step(
            self,
            inp: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
            out: pl.Out[pl.Tensor[[BATCH, HIDDEN], pl.BF16]],
            data: pl.InOut[pld.DistributedTensor[[BATCH, HIDDEN], pl.BF16]],
            signal: pl.InOut[pld.DistributedTensor[[sig_rows, sig_cols], pl.INT32]],
            my_rank: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[BATCH, HIDDEN], pl.BF16]:
            for k0 in pl.parallel(0, HIDDEN, COL_CHUNK):
                pl.store(pl.load(inp, [0, k0], [BATCH, COL_CHUNK]), [0, k0], data)
            # RS barrier (row 0) — parallel fan-out.
            for peer in pl.parallel(n_ranks):
                if peer != my_rank:
                    pld.system.notify(signal, peer=peer, offsets=[0, my_rank],
                                      value=1, op=pld.NotifyOp.AtomicAdd)
            for src in pl.parallel(n_ranks):
                if src != my_rank:
                    pld.system.wait(signal=signal, offsets=[0, src],
                                    expected=1, cmp=pld.WaitCmp.Ge)
            # Reduce-scatter accumulate — serial (carried reduction).
            base = my_rank * chunk
            own = pl.load(data, [0, base], [BATCH, chunk])
            acc = pl.cast(own, target_type=pl.FP32)
            for peer in pl.range(n_ranks):
                if peer != my_rank:
                    recv = pld.tile.remote_load(data, peer=peer, offsets=[0, base],
                                                shape=[BATCH, chunk])
                    acc = pl.add(acc, pl.cast(recv, target_type=pl.FP32))
            reduced = pl.cast(acc, target_type=pl.BF16)
            pl.store(reduced, [0, base], data)
            # AG barrier (row 1) — parallel fan-out.
            for peer in pl.parallel(n_ranks):
                if peer != my_rank:
                    pld.system.notify(signal, peer=peer, offsets=[1, my_rank],
                                      value=1, op=pld.NotifyOp.AtomicAdd)
            for src in pl.parallel(n_ranks):
                if src != my_rank:
                    pld.system.wait(signal=signal, offsets=[1, src],
                                    expected=1, cmp=pld.WaitCmp.Ge)
            # All-gather — parallel peer stores.
            for r in pl.parallel(n_ranks):
                off = r * chunk
                if r != my_rank:
                    red = pld.tile.remote_load(data, peer=r, offsets=[0, off],
                                               shape=[BATCH, chunk])
                    pl.store(red, [0, off], out)
                else:
                    pl.store(reduced, [0, off], out)
            return out

        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(
            self,
            inp: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
            out: pl.Out[pl.Tensor[[BATCH, HIDDEN], pl.BF16]],
            data: pl.InOut[pld.DistributedTensor[[BATCH, HIDDEN], pl.BF16]],
            signal: pl.InOut[pld.DistributedTensor[[sig_rows, sig_cols], pl.INT32]],
            my_rank: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[BATCH, HIDDEN], pl.BF16]:
            return self.reduce_step(inp, out, data, signal, my_rank)

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(
            self,
            inputs: pl.Tensor[[n_ranks, BATCH, HIDDEN], pl.BF16],
            outputs: pl.Out[pl.Tensor[[n_ranks, BATCH, HIDDEN], pl.BF16]],
        ) -> pl.Tensor[[n_ranks, BATCH, HIDDEN], pl.BF16]:
            data_buf = pld.alloc_window_buffer(BATCH * HIDDEN * 2)  # BF16 = 2 bytes
            signal_buf = pld.alloc_window_buffer(sig_rows * sig_cols * 4)  # INT32 = 4 bytes
            for r in pl.range(pld.world_size()):
                data = pld.window(data_buf, [BATCH, HIDDEN], dtype=pl.BF16)
                signal = pld.window(signal_buf, [sig_rows, sig_cols], dtype=pl.INT32)
                self.chip_orch(inputs[r], outputs[r], data, signal, r, device=r)
            return outputs

    return AllReduceTwophasePar


def _build_pld(n_ranks: int, ring: bool):
    """Composite intrinsic ``pld.tensor.allreduce(data, signal, mode=...)``.

    Compares the compiler-lowered collective against the hand-rolled raw
    variants. Signal shape follows the intrinsic contract: mesh ``[NR, 1]``;
    ring ``[2*(NR-1), NR]`` (one row per ring round).
    """
    if ring:
        sig_rows, sig_cols = 2 * (n_ranks - 1), n_ranks
        _mode = "ring"
    else:
        sig_rows, sig_cols = n_ranks, 1
        _mode = "mesh"

    @pl.program
    class AllReducePld:
        @pl.function(type=pl.FunctionType.InCore)
        def reduce_step(
            self,
            inp: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
            out: pl.Out[pl.Tensor[[BATCH, HIDDEN], pl.BF16]],
            data: pl.InOut[pld.DistributedTensor[[BATCH, HIDDEN], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[sig_rows, sig_cols], pl.INT32]],
            my_rank: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[BATCH, HIDDEN], pl.BF16]:
            # Stage BF16 input into an FP32 window: makes the intrinsic's tadd
            # FP32 (A2/A3 rejects BF16 tadd) and, for ring, keeps the per-chunk
            # FP32 accumulator within UB.
            for k0 in pl.range(0, HIDDEN, COL_CHUNK):
                t_in = pl.load(inp, [0, k0], [BATCH, COL_CHUNK])
                pl.store(pl.cast(t_in, target_type=pl.FP32), [0, k0], data)
            data = pld.tensor.allreduce(data, signal, op=pld.ReduceOp.Sum, mode=_mode)
            for k0 in pl.range(0, HIDDEN, COL_CHUNK):
                t_out = pl.load(data, [0, k0], [BATCH, COL_CHUNK])
                pl.store(pl.cast(t_out, target_type=pl.BF16), [0, k0], out)
            return out

        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(
            self,
            inp: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
            out: pl.Out[pl.Tensor[[BATCH, HIDDEN], pl.BF16]],
            data: pl.InOut[pld.DistributedTensor[[BATCH, HIDDEN], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[sig_rows, sig_cols], pl.INT32]],
            my_rank: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[BATCH, HIDDEN], pl.BF16]:
            return self.reduce_step(inp, out, data, signal, my_rank)

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(
            self,
            inputs: pl.Tensor[[n_ranks, BATCH, HIDDEN], pl.BF16],
            outputs: pl.Out[pl.Tensor[[n_ranks, BATCH, HIDDEN], pl.BF16]],
        ) -> pl.Tensor[[n_ranks, BATCH, HIDDEN], pl.BF16]:
            data_buf = pld.alloc_window_buffer(BATCH * HIDDEN * 4)  # FP32 = 4 bytes
            signal_buf = pld.alloc_window_buffer(sig_rows * sig_cols * 4)  # INT32 = 4 bytes
            for r in pl.range(pld.world_size()):
                data = pld.window(data_buf, [BATCH, HIDDEN], dtype=pl.FP32)
                signal = pld.window(signal_buf, [sig_rows, sig_cols], dtype=pl.INT32)
                self.chip_orch(inputs[r], outputs[r], data, signal, r, device=r)
            return outputs

    return AllReducePld


def _build(mode: str, n_ranks: int):
    if mode == "onephase":
        return _build_onephase(n_ranks)
    if mode == "onephase_par":
        return _build_onephase_par(n_ranks)
    if mode == "twophase":
        return _build_twophase(n_ranks)
    if mode == "twophase_par":
        return _build_twophase_par(n_ranks)
    if mode == "ring":
        return _build_ring(n_ranks)
    if mode in ("pld_mesh", "pld_ring"):
        return _build_pld(n_ranks, ring=(mode == "pld_ring"))
    raise ValueError(f"mode must be one of {_MODES}, got {mode!r}")


def _make_rank_inputs(n_ranks: int) -> torch.Tensor:
    rows = []
    base = torch.arange(HIDDEN, dtype=torch.float32) % 8  # column pattern 0..7
    for r in range(n_ranks):
        val = (base + r).reshape(1, HIDDEN).expand(BATCH, HIDDEN).contiguous()
        rows.append(val.to(torch.bfloat16))
    return torch.stack(rows)


def _expected_allreduce(inputs: torch.Tensor) -> torch.Tensor:
    reduced = inputs.to(torch.float32).sum(dim=0)
    return torch.stack([reduced] * inputs.shape[0]).to(torch.bfloat16)


def _parse_devices(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--platform", default="a2a3sim")
    ap.add_argument("-d", "--device", default="0-1")
    ap.add_argument("--mode", default="twophase", choices=_MODES)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    args = ap.parse_args()

    device_ids = _parse_devices(args.device)
    n_ranks = len(device_ids)
    program = _build(args.mode, n_ranks)
    compiled = ir.compile(
        program,
        platform=args.platform,
        distributed_config=DistributedConfig(device_ids=device_ids, num_sub_workers=0),
    )

    inputs = _make_rank_inputs(n_ranks)
    outputs = torch.zeros((n_ranks, BATCH, HIDDEN), dtype=torch.bfloat16)
    compiled(inputs, outputs)

    expected = _expected_allreduce(inputs)
    max_diff = (outputs.to(torch.float32) - expected.to(torch.float32)).abs().max().item()
    ok = torch.allclose(outputs.to(torch.float32), expected.to(torch.float32), atol=0.0)
    print(f"[{args.mode}] P={n_ranks} golden max_diff={max_diff} ok={ok}")
    if not ok:
        print(f"[{args.mode}] GOLDEN FAIL — aborting timing")
        return 1

    for _ in range(args.warmup):
        compiled(inputs, outputs)
    t0 = time.perf_counter()
    for _ in range(args.iters):
        compiled(inputs, outputs)
    dt = (time.perf_counter() - t0) / args.iters
    print(f"[{args.mode}] P={n_ranks} avg wall/iter={dt * 1e3:.3f} ms over {args.iters} iters")
    return 0


if __name__ == "__main__":
    sys.exit(main())
