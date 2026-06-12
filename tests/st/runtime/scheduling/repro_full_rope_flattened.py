from pathlib import Path
import shutil
import torch

import pypto.language as pl
from pypto import ir
from pypto.ir.distributed_compiled_program import DistributedConfig
from pypto.backend import BackendType, set_backend_type

BATCH = 16
HIDDEN = 4096
HEAD_DIM = 128
HIDDEN_Q = 1024
KV_HIDDEN = 128
HALF = 32
ROTARY_DIM = 64
Q_HEAD_BATCH = 8
Q_HEAD_PAD = 16
SEQ = 4096
BLOCK_SIZE = 16
MAX_BLOCKS_PER_SEQ = 32
LAYER_DYN = 45
NF = 12
LAYER_HIDDEN_ROWS = NF * HIDDEN
LAYER_QHIDDEN_ROWS = NF * HIDDEN_Q

@pl.program
class FullRopeFlattened:
    @pl.function(type=pl.FunctionType.Orchestration)
    def chip_orch(
        self,
        k_proj_norm: pl.Tensor[[BATCH, HEAD_DIM], pl.FP32],
        v_proj: pl.Tensor[[BATCH, HEAD_DIM], pl.FP32],
        q_proj_norm: pl.Tensor[[BATCH, Q_HEAD_BATCH * HEAD_DIM], pl.FP32],
        seq_lens: pl.Tensor[[BATCH], pl.INT32],
        block_table: pl.Tensor[[BATCH * MAX_BLOCKS_PER_SEQ], pl.INT32],
        slot_mapping: pl.Tensor[[BATCH], pl.INT32],
        rope_cos: pl.Tensor[[SEQ, ROTARY_DIM], pl.FP32],
        rope_sin: pl.Tensor[[SEQ, ROTARY_DIM], pl.FP32],
        k_cache: pl.Out[pl.Tensor[[LAYER_DYN * SEQ, HEAD_DIM], pl.BF16]],
        v_cache: pl.Out[pl.Tensor[[LAYER_DYN * SEQ, HEAD_DIM], pl.BF16]],
        all_q_padded: pl.Out[pl.Tensor[[BATCH * Q_HEAD_PAD, HEAD_DIM], pl.BF16]],
        layer_idx: pl.Scalar[pl.INT32],
    ) -> pl.Tensor[[BATCH * Q_HEAD_PAD, HEAD_DIM], pl.BF16]:
        decode_layer_cache_rows = pl.tensor.dim(k_cache, 0) // LAYER_DYN
        layer_cache_base = layer_idx * decode_layer_cache_rows
        bt_stride = pl.tensor.dim(block_table, 0) // BATCH
        for b in pl.parallel(BATCH):
            ctx_len = pl.tensor.read(seq_lens, [b])
            pos = ctx_len - 1
            slot = pl.tensor.read(slot_mapping, [b])
            slot_block = slot // BLOCK_SIZE
            slot_offset = slot - slot_block * BLOCK_SIZE
            cos_row = pl.slice(rope_cos, [1, ROTARY_DIM], [pos, 0])
            sin_row = pl.slice(rope_sin, [1, ROTARY_DIM], [pos, 0])
            cos_lo = pl.slice(cos_row, [1, HALF], [0, 0])
            cos_hi = pl.slice(cos_row, [1, HALF], [0, HALF])
            sin_lo = pl.slice(sin_row, [1, HALF], [0, 0])
            sin_hi = pl.slice(sin_row, [1, HALF], [0, HALF])
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="full_rope_kv_cache"):
                cache_row = layer_cache_base + (slot_block * 1 + 0) * BLOCK_SIZE + slot_offset
                k_lo = pl.slice(k_proj_norm, [1, HALF], [b, 0])
                k_hi = pl.slice(k_proj_norm, [1, HALF], [b, HALF])
                rot_k_lo = pl.sub(pl.col_expand_mul(k_lo, cos_lo), pl.col_expand_mul(k_hi, sin_lo))
                rot_k_hi = pl.add(pl.col_expand_mul(k_hi, cos_hi), pl.col_expand_mul(k_lo, sin_hi))
                k_cache = pl.assemble(k_cache, pl.cast(pl.slice(k_proj_norm, [1, HEAD_DIM], [b, 0]), target_type=pl.BF16), [cache_row, 0])
                k_cache = pl.assemble(k_cache, pl.cast(rot_k_lo, target_type=pl.BF16), [cache_row, 0])
                k_cache = pl.assemble(k_cache, pl.cast(rot_k_hi, target_type=pl.BF16), [cache_row, HALF])
                v_cache = pl.assemble(v_cache, pl.cast(pl.slice(v_proj, [1, HEAD_DIM], [b, 0]), target_type=pl.BF16), [cache_row, 0])
                q_block = pl.reshape(pl.slice(q_proj_norm, [1, Q_HEAD_BATCH * HEAD_DIM], [b, 0]), [Q_HEAD_BATCH, HEAD_DIM])
                q_lo = pl.slice(q_block, [Q_HEAD_BATCH, HALF], [0, 0])
                q_hi = pl.slice(q_block, [Q_HEAD_BATCH, HALF], [0, HALF])
                rot_q_lo = pl.sub(pl.col_expand_mul(q_lo, cos_lo), pl.col_expand_mul(q_hi, sin_lo))
                rot_q_hi = pl.add(pl.col_expand_mul(q_hi, cos_hi), pl.col_expand_mul(q_lo, sin_hi))
                base = b * Q_HEAD_PAD
                all_q_padded = pl.assemble(all_q_padded, pl.cast(q_block, target_type=pl.BF16), [base, 0])
                all_q_padded = pl.assemble(all_q_padded, pl.cast(rot_q_lo, target_type=pl.BF16), [base, 0])
                all_q_padded = pl.assemble(all_q_padded, pl.cast(rot_q_hi, target_type=pl.BF16), [base, HALF])
                all_q_padded = pl.assemble(all_q_padded, pl.cast(pl.full([Q_HEAD_PAD - Q_HEAD_BATCH, HEAD_DIM], dtype=pl.FP32, value=0.0), target_type=pl.BF16), [base + Q_HEAD_BATCH, 0])
        return all_q_padded

    @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
    def host_orch(
        self,
        k_proj_norm: pl.Tensor[[1, BATCH, HEAD_DIM], pl.FP32],
        v_proj: pl.Tensor[[1, BATCH, HEAD_DIM], pl.FP32],
        q_proj_norm: pl.Tensor[[1, BATCH, Q_HEAD_BATCH * HEAD_DIM], pl.FP32],
        seq_lens: pl.Tensor[[1, BATCH], pl.INT32],
        block_table: pl.Tensor[[1, BATCH * MAX_BLOCKS_PER_SEQ], pl.INT32],
        slot_mapping: pl.Tensor[[1, BATCH], pl.INT32],
        rope_cos: pl.Tensor[[1, SEQ, ROTARY_DIM], pl.FP32],
        rope_sin: pl.Tensor[[1, SEQ, ROTARY_DIM], pl.FP32],
        k_cache: pl.Out[pl.Tensor[[1, LAYER_DYN * SEQ, HEAD_DIM], pl.BF16]],
        v_cache: pl.Out[pl.Tensor[[1, LAYER_DYN * SEQ, HEAD_DIM], pl.BF16]],
        all_q_padded: pl.Out[pl.Tensor[[1, BATCH * Q_HEAD_PAD, HEAD_DIM], pl.BF16]],
        layer_idx: pl.Scalar[pl.INT32],
    ) -> pl.Tensor[[1, BATCH * Q_HEAD_PAD, HEAD_DIM], pl.BF16]:
        self.chip_orch(
            pl.tensor.slice(k_proj_norm, [1, BATCH, HEAD_DIM], [0, 0, 0], [], [0]),
            pl.tensor.slice(v_proj, [1, BATCH, HEAD_DIM], [0, 0, 0], [], [0]),
            pl.tensor.slice(q_proj_norm, [1, BATCH, Q_HEAD_BATCH * HEAD_DIM], [0, 0, 0], [], [0]),
            pl.tensor.slice(seq_lens, [1, BATCH], [0, 0], [], [0]),
            pl.tensor.slice(block_table, [1, BATCH * MAX_BLOCKS_PER_SEQ], [0, 0], [], [0]),
            pl.tensor.slice(slot_mapping, [1, BATCH], [0, 0], [], [0]),
            pl.tensor.slice(rope_cos, [1, SEQ, ROTARY_DIM], [0, 0, 0], [], [0]),
            pl.tensor.slice(rope_sin, [1, SEQ, ROTARY_DIM], [0, 0, 0], [], [0]),
            pl.tensor.slice(k_cache, [1, LAYER_DYN * SEQ, HEAD_DIM], [0, 0, 0], [], [0]),
            pl.tensor.slice(v_cache, [1, LAYER_DYN * SEQ, HEAD_DIM], [0, 0, 0], [], [0]),
            pl.tensor.slice(all_q_padded, [1, BATCH * Q_HEAD_PAD, HEAD_DIM], [0, 0, 0], [], [0]),
            layer_idx,
            device=0,
        )
        return all_q_padded


def main():
    set_backend_type(BackendType.Ascend910B)
    compiled = ir.compile(
        FullRopeFlattened,
        platform='a2a3',
        distributed_config=DistributedConfig(device_ids=[0], num_sub_workers=0, block_dim=3, aicpu_thread_num=4),
    )
    tensors = {
        'k_proj_norm': torch.ones(1, BATCH, HEAD_DIM, dtype=torch.float32),
        'v_proj': torch.ones(1, BATCH, HEAD_DIM, dtype=torch.float32),
        'q_proj_norm': torch.ones(1, BATCH, Q_HEAD_BATCH * HEAD_DIM, dtype=torch.float32),
        'seq_lens': torch.ones(1, BATCH, dtype=torch.int32),
        'block_table': torch.zeros(1, BATCH * MAX_BLOCKS_PER_SEQ, dtype=torch.int32),
        'slot_mapping': torch.arange(BATCH, dtype=torch.int32).unsqueeze(0),
        'rope_cos': torch.ones(1, SEQ, ROTARY_DIM, dtype=torch.float32),
        'rope_sin': torch.zeros(1, SEQ, ROTARY_DIM, dtype=torch.float32),
        'k_cache': torch.zeros(1, LAYER_DYN * SEQ, HEAD_DIM, dtype=torch.bfloat16),
        'v_cache': torch.zeros(1, LAYER_DYN * SEQ, HEAD_DIM, dtype=torch.bfloat16),
        'all_q_padded': torch.zeros(1, BATCH * Q_HEAD_PAD, HEAD_DIM, dtype=torch.bfloat16),
    }
    compiled(
        tensors['k_proj_norm'], tensors['v_proj'], tensors['q_proj_norm'],
        tensors['seq_lens'], tensors['block_table'], tensors['slot_mapping'],
        tensors['rope_cos'], tensors['rope_sin'], tensors['k_cache'], tensors['v_cache'],
        tensors['all_q_padded'], torch.tensor(0, dtype=torch.int32),
    )
    print('PASS')

if __name__ == '__main__':
    main()
