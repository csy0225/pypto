from pathlib import Path
import shutil
import torch

import pypto.language as pl
from pypto.backend import BackendType, set_backend_type
from pypto.ir.pass_manager import OptimizationStrategy
from pypto import ir
from pypto.ir.distributed_compiled_program import DistributedConfig
from pypto.runtime.golden_writer import generate_golden_source, _save_data_files
from pypto.runtime.tensor_spec import TensorSpec

BATCH = 16
HEAD_DIM = 128
HALF = 32
ROTARY_DIM = 64
Q_HEAD_BATCH = 8
Q_HEAD_PAD = 16
SEQ = 4096
BLOCK_SIZE = 16

@pl.program
class FullRopeHostSlice:

    @pl.function(type=pl.FunctionType.InCore)
    def warm_aiv(self, all_q_padded: pl.Out[pl.Tensor[[BATCH * Q_HEAD_PAD, HEAD_DIM], pl.BF16]]) -> pl.Tensor[[BATCH * Q_HEAD_PAD, HEAD_DIM], pl.BF16]:
        z = pl.cast(pl.full([1, HALF], dtype=pl.FP32, value=0.0), target_type=pl.BF16)
        all_q_padded = pl.assemble(all_q_padded, z, [0, 0])
        return all_q_padded

    @pl.function(type=pl.FunctionType.InCore)
    def warm_aic(self, x: pl.Tensor[[BATCH, HEAD_DIM], pl.FP32], y: pl.Out[pl.Tensor[[BATCH, HEAD_DIM], pl.FP32]]) -> pl.Tensor[[BATCH, HEAD_DIM], pl.FP32]:
        tile_a = pl.slice(x, [BATCH, 32], [0, 0])
        tile_b = pl.full([32, 32], dtype=pl.FP32, value=0.0)
        acc = pl.matmul(tile_a, tile_b, out_dtype=pl.FP32)
        y = pl.assemble(y, acc, [0, 0])
        return y
    @pl.function(type=pl.FunctionType.Orchestration)
    def chip_orch(
        self,
        k_proj_norm: pl.Tensor[[BATCH, HEAD_DIM], pl.FP32],
        v_proj: pl.Tensor[[BATCH, HEAD_DIM], pl.FP32],
        q_proj_norm: pl.Tensor[[BATCH, Q_HEAD_BATCH * HEAD_DIM], pl.FP32],
        seq_lens: pl.Tensor[[BATCH], pl.INT32],
        slot_mapping: pl.Tensor[[BATCH], pl.INT32],
        rope_cos: pl.Tensor[[SEQ, ROTARY_DIM], pl.FP32],
        rope_sin: pl.Tensor[[SEQ, ROTARY_DIM], pl.FP32],
        scratch: pl.Out[pl.Tensor[[BATCH, HEAD_DIM], pl.FP32]],
        k_cache: pl.Out[pl.Tensor[[SEQ, HEAD_DIM], pl.BF16]],
        v_cache: pl.Out[pl.Tensor[[SEQ, HEAD_DIM], pl.BF16]],
        all_q_padded: pl.Out[pl.Tensor[[BATCH * Q_HEAD_PAD, HEAD_DIM], pl.BF16]],
    ) -> pl.Tensor[[BATCH * Q_HEAD_PAD, HEAD_DIM], pl.BF16]:
        all_q_padded = self.warm_aiv(all_q_padded)
        scratch = self.warm_aic(k_proj_norm, scratch)
        all_q_padded = self.warm_aiv(all_q_padded)
        scratch = self.warm_aic(v_proj, scratch)
        all_q_padded = self.warm_aiv(all_q_padded)
        scratch = self.warm_aic(k_proj_norm, scratch)
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
                cache_row = slot_block * BLOCK_SIZE + slot_offset
                k_lo = pl.slice(k_proj_norm, [1, HALF], [b, 0])
                k_hi = pl.slice(k_proj_norm, [1, HALF], [b, HALF])
                rot_k_lo = pl.sub(pl.col_expand_mul(k_lo, cos_lo), pl.col_expand_mul(k_hi, sin_lo))
                rot_k_hi = pl.add(pl.col_expand_mul(k_hi, cos_hi), pl.col_expand_mul(k_lo, sin_hi))
                k_full_bf16 = pl.cast(pl.slice(k_proj_norm, [1, HEAD_DIM], [b, 0]), target_type=pl.BF16)
                k_cache = pl.assemble(k_cache, k_full_bf16, [cache_row, 0])
                k_cache = pl.assemble(k_cache, pl.cast(rot_k_lo, target_type=pl.BF16), [cache_row, 0])
                k_cache = pl.assemble(k_cache, pl.cast(rot_k_hi, target_type=pl.BF16), [cache_row, HALF])
                v_cache = pl.assemble(
                    v_cache,
                    pl.cast(pl.slice(v_proj, [1, HEAD_DIM], [b, 0]), target_type=pl.BF16),
                    [cache_row, 0],
                )

                q_block = pl.reshape(pl.slice(q_proj_norm, [1, Q_HEAD_BATCH * HEAD_DIM], [b, 0]), [Q_HEAD_BATCH, HEAD_DIM])
                q_lo = pl.slice(q_block, [Q_HEAD_BATCH, HALF], [0, 0])
                q_hi = pl.slice(q_block, [Q_HEAD_BATCH, HALF], [0, HALF])
                rot_q_lo = pl.sub(pl.col_expand_mul(q_lo, cos_lo), pl.col_expand_mul(q_hi, sin_lo))
                rot_q_hi = pl.add(pl.col_expand_mul(q_hi, cos_hi), pl.col_expand_mul(q_lo, sin_hi))
                q_block_bf16 = pl.cast(q_block, target_type=pl.BF16)
                pad_row_base = b * Q_HEAD_PAD
                all_q_padded = pl.assemble(all_q_padded, q_block_bf16, [pad_row_base, 0])
                all_q_padded = pl.assemble(all_q_padded, pl.cast(rot_q_lo, target_type=pl.BF16), [pad_row_base, 0])
                all_q_padded = pl.assemble(all_q_padded, pl.cast(rot_q_hi, target_type=pl.BF16), [pad_row_base, HALF])
                all_q_padded = pl.assemble(
                    all_q_padded,
                    pl.cast(pl.full([Q_HEAD_PAD - Q_HEAD_BATCH, HEAD_DIM], dtype=pl.FP32, value=0.0), target_type=pl.BF16),
                    [pad_row_base + Q_HEAD_BATCH, 0],
                )
        return all_q_padded


    @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
    def host_orch(
        self,
        k_proj_norm: pl.Tensor[[1, BATCH, HEAD_DIM], pl.FP32],
        v_proj: pl.Tensor[[1, BATCH, HEAD_DIM], pl.FP32],
        q_proj_norm: pl.Tensor[[1, BATCH, Q_HEAD_BATCH * HEAD_DIM], pl.FP32],
        seq_lens: pl.Tensor[[1, BATCH], pl.INT32],
        slot_mapping: pl.Tensor[[1, BATCH], pl.INT32],
        rope_cos: pl.Tensor[[1, SEQ, ROTARY_DIM], pl.FP32],
        rope_sin: pl.Tensor[[1, SEQ, ROTARY_DIM], pl.FP32],
        scratch: pl.Out[pl.Tensor[[1, BATCH, HEAD_DIM], pl.FP32]],
        k_cache: pl.Out[pl.Tensor[[1, SEQ, HEAD_DIM], pl.BF16]],
        v_cache: pl.Out[pl.Tensor[[1, SEQ, HEAD_DIM], pl.BF16]],
        all_q_padded: pl.Out[pl.Tensor[[1, BATCH * Q_HEAD_PAD, HEAD_DIM], pl.BF16]],
    ) -> pl.Tensor[[1, BATCH * Q_HEAD_PAD, HEAD_DIM], pl.BF16]:
        self.chip_orch(
            pl.tensor.slice(k_proj_norm, [1, BATCH, HEAD_DIM], [0, 0, 0], [], [0]),
            pl.tensor.slice(v_proj, [1, BATCH, HEAD_DIM], [0, 0, 0], [], [0]),
            pl.tensor.slice(q_proj_norm, [1, BATCH, Q_HEAD_BATCH * HEAD_DIM], [0, 0, 0], [], [0]),
            pl.tensor.slice(seq_lens, [1, BATCH], [0, 0], [], [0]),
            pl.tensor.slice(slot_mapping, [1, BATCH], [0, 0], [], [0]),
            pl.tensor.slice(rope_cos, [1, SEQ, ROTARY_DIM], [0, 0, 0], [], [0]),
            pl.tensor.slice(rope_sin, [1, SEQ, ROTARY_DIM], [0, 0, 0], [], [0]),
            pl.tensor.slice(scratch, [1, BATCH, HEAD_DIM], [0, 0, 0], [], [0]),
            pl.tensor.slice(k_cache, [1, SEQ, HEAD_DIM], [0, 0, 0], [], [0]),
            pl.tensor.slice(v_cache, [1, SEQ, HEAD_DIM], [0, 0, 0], [], [0]),
            pl.tensor.slice(all_q_padded, [1, BATCH * Q_HEAD_PAD, HEAD_DIM], [0, 0, 0], [], [0]),
            device=0,
        )
        return all_q_padded


def main():
    work_dir = Path('/tmp/p15_min_full_rope')
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)
    set_backend_type(BackendType.Ascend910B)
    compiled = ir.compile(
        FullRopeHostSlice,
        platform='a2a3',
        distributed_config=DistributedConfig(
            device_ids=[0],
            num_sub_workers=0,
            block_dim=3,
            aicpu_thread_num=4,
        ),
    )
    specs = [
        TensorSpec('k_proj_norm', [1, BATCH, HEAD_DIM], torch.float32, init_value=torch.ones(1, BATCH, HEAD_DIM, dtype=torch.float32)),
        TensorSpec('v_proj', [1, BATCH, HEAD_DIM], torch.float32, init_value=torch.ones(1, BATCH, HEAD_DIM, dtype=torch.float32)),
        TensorSpec('q_proj_norm', [1, BATCH, Q_HEAD_BATCH * HEAD_DIM], torch.float32, init_value=torch.ones(1, BATCH, Q_HEAD_BATCH * HEAD_DIM, dtype=torch.float32)),
        TensorSpec('seq_lens', [1, BATCH], torch.int32, init_value=torch.ones(1, BATCH, dtype=torch.int32)),
        TensorSpec('slot_mapping', [1, BATCH], torch.int32, init_value=torch.arange(BATCH, dtype=torch.int32).unsqueeze(0)),
        TensorSpec('rope_cos', [1, SEQ, ROTARY_DIM], torch.float32, init_value=torch.ones(1, SEQ, ROTARY_DIM, dtype=torch.float32)),
        TensorSpec('rope_sin', [1, SEQ, ROTARY_DIM], torch.float32, init_value=torch.zeros(1, SEQ, ROTARY_DIM, dtype=torch.float32)),
        TensorSpec('scratch', [1, BATCH, HEAD_DIM], torch.float32, is_output=True),
        TensorSpec('k_cache', [1, SEQ, HEAD_DIM], torch.bfloat16, is_output=True),
        TensorSpec('v_cache', [1, SEQ, HEAD_DIM], torch.bfloat16, is_output=True),
        TensorSpec('all_q_padded', [1, BATCH * Q_HEAD_PAD, HEAD_DIM], torch.bfloat16, is_output=True),
    ]
    tensors = {s.name: (s.init_value.clone() if not s.is_output else torch.zeros(*s.shape, dtype=s.dtype)) for s in specs}
    compiled(
        tensors['k_proj_norm'], tensors['v_proj'], tensors['q_proj_norm'],
        tensors['seq_lens'], tensors['slot_mapping'], tensors['rope_cos'], tensors['rope_sin'],
        tensors['scratch'], tensors['k_cache'], tensors['v_cache'], tensors['all_q_padded'],
    )
    print('PASS')

if __name__ == '__main__':
    main()
