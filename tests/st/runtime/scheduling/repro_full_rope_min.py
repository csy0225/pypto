from pathlib import Path
import shutil
import torch

import pypto.language as pl
from pypto.backend import BackendType, set_backend_type
from pypto.ir.pass_manager import OptimizationStrategy
from pypto.runtime import compile_program
from pypto.runtime.device_runner import compile_and_assemble
from pypto.runtime.runner import _execute_on_device, _DfxOpts
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
class FullRopeMin:

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
    def orchestrator(
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


def main():
    work_dir = Path('/tmp/p15_min_full_rope')
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)
    set_backend_type(BackendType.Ascend910B)
    compile_program(FullRopeMin, work_dir, strategy=OptimizationStrategy.Default, backend_type=BackendType.Ascend910B)
    specs = [
        TensorSpec('k_proj_norm', [BATCH, HEAD_DIM], torch.float32, init_value=torch.ones(BATCH, HEAD_DIM, dtype=torch.float32)),
        TensorSpec('v_proj', [BATCH, HEAD_DIM], torch.float32, init_value=torch.ones(BATCH, HEAD_DIM, dtype=torch.float32)),
        TensorSpec('q_proj_norm', [BATCH, Q_HEAD_BATCH * HEAD_DIM], torch.float32, init_value=torch.ones(BATCH, Q_HEAD_BATCH * HEAD_DIM, dtype=torch.float32)),
        TensorSpec('seq_lens', [BATCH], torch.int32, init_value=torch.ones(BATCH, dtype=torch.int32)),
        TensorSpec('slot_mapping', [BATCH], torch.int32, init_value=torch.arange(BATCH, dtype=torch.int32)),
        TensorSpec('rope_cos', [SEQ, ROTARY_DIM], torch.float32, init_value=torch.ones(SEQ, ROTARY_DIM, dtype=torch.float32)),
        TensorSpec('rope_sin', [SEQ, ROTARY_DIM], torch.float32, init_value=torch.zeros(SEQ, ROTARY_DIM, dtype=torch.float32)),
        TensorSpec('scratch', [BATCH, HEAD_DIM], torch.float32, is_output=True),
        TensorSpec('k_cache', [SEQ, HEAD_DIM], torch.bfloat16, is_output=True),
        TensorSpec('v_cache', [SEQ, HEAD_DIM], torch.bfloat16, is_output=True),
        TensorSpec('all_q_padded', [BATCH * Q_HEAD_PAD, HEAD_DIM], torch.bfloat16, is_output=True),
    ]
    golden_src = '''def compute_golden(tensors, params):
    tensors["scratch"][:] = 0
    tensors["k_cache"][:] = 0
    tensors["v_cache"][:] = 0
    tensors["all_q_padded"][:] = 0
'''
    (work_dir / 'golden.py').write_text(generate_golden_source(specs, None, 1e9, 1e9, compute_golden_src=golden_src))
    in_data = {s.name: s.init_value.clone() for s in specs if not s.is_output}
    _save_data_files(in_data, work_dir / 'data' / 'in')
    _save_data_files({s.name: torch.zeros(*s.shape, dtype=s.dtype) for s in specs if s.is_output}, work_dir / 'data' / 'out')
    chip_callable, runtime_name, _ = compile_and_assemble(work_dir, 'a2a3')
    timing = _execute_on_device(work_dir, work_dir / 'golden.py', chip_callable, runtime_name, 'a2a3', 0, dfx=_DfxOpts())
    print('PASS', timing)

if __name__ == '__main__':
    main()
