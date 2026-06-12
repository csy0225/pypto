from pathlib import Path
import argparse
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
class FullRopeConsumerMin:
    @pl.function(type=pl.FunctionType.InCore)
    def make_kv(
        self,
        src: pl.Tensor[[BATCH, HEAD_DIM], pl.FP32],
        out: pl.Out[pl.Tensor[[BATCH, HEAD_DIM], pl.FP32]],
    ) -> pl.Tensor[[BATCH, HEAD_DIM], pl.FP32]:
        for p in pl.range(HEAD_DIM // HALF):
            tile = pl.add(pl.slice(src, [BATCH, HALF], [0, p * HALF]), 0.0)
            out = pl.assemble(out, tile, [0, p * HALF])
        return out

    @pl.function(type=pl.FunctionType.InCore)
    def make_q(
        self,
        src: pl.Tensor[[BATCH, Q_HEAD_BATCH * HEAD_DIM], pl.FP32],
        out: pl.Out[pl.Tensor[[BATCH, Q_HEAD_BATCH * HEAD_DIM], pl.FP32]],
    ) -> pl.Tensor[[BATCH, Q_HEAD_BATCH * HEAD_DIM], pl.FP32]:
        for p in pl.range((Q_HEAD_BATCH * HEAD_DIM) // HALF):
            tile = pl.add(pl.slice(src, [BATCH, HALF], [0, p * HALF]), 0.0)
            out = pl.assemble(out, tile, [0, p * HALF])
        return out

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        k_src: pl.Tensor[[BATCH, HEAD_DIM], pl.FP32],
        v_src: pl.Tensor[[BATCH, HEAD_DIM], pl.FP32],
        q_src: pl.Tensor[[BATCH, Q_HEAD_BATCH * HEAD_DIM], pl.FP32],
        seq_lens: pl.Tensor[[BATCH], pl.INT32],
        slot_mapping: pl.Tensor[[BATCH], pl.INT32],
        rope_cos: pl.Tensor[[SEQ, ROTARY_DIM], pl.FP32],
        rope_sin: pl.Tensor[[SEQ, ROTARY_DIM], pl.FP32],
        k_cache: pl.Out[pl.Tensor[[SEQ, HEAD_DIM], pl.BF16]],
        v_cache: pl.Out[pl.Tensor[[SEQ, HEAD_DIM], pl.BF16]],
        out: pl.Out[pl.Tensor[[BATCH, Q_HEAD_PAD], pl.FP32]],
    ) -> pl.Tensor[[BATCH, Q_HEAD_PAD], pl.FP32]:
        k_proj_norm = pl.create_tensor([BATCH, HEAD_DIM], dtype=pl.FP32)
        v_proj = pl.create_tensor([BATCH, HEAD_DIM], dtype=pl.FP32)
        q_proj_norm = pl.create_tensor([BATCH, Q_HEAD_BATCH * HEAD_DIM], dtype=pl.FP32)
        all_q_padded = pl.create_tensor([BATCH * Q_HEAD_PAD, HEAD_DIM], dtype=pl.BF16)
        k_proj_norm = self.make_kv(k_src, k_proj_norm)
        v_proj = self.make_kv(v_src, v_proj)
        q_proj_norm = self.make_q(q_src, q_proj_norm)

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
                v_cache = pl.assemble(v_cache, pl.cast(pl.slice(v_proj, [1, HEAD_DIM], [b, 0]), target_type=pl.BF16), [cache_row, 0])
                q_block = pl.reshape(pl.slice(q_proj_norm, [1, Q_HEAD_BATCH * HEAD_DIM], [b, 0]), [Q_HEAD_BATCH, HEAD_DIM])
                q_lo = pl.slice(q_block, [Q_HEAD_BATCH, HALF], [0, 0])
                q_hi = pl.slice(q_block, [Q_HEAD_BATCH, HALF], [0, HALF])
                rot_q_lo = pl.sub(pl.col_expand_mul(q_lo, cos_lo), pl.col_expand_mul(q_hi, sin_lo))
                rot_q_hi = pl.add(pl.col_expand_mul(q_hi, cos_hi), pl.col_expand_mul(q_lo, sin_hi))
                pad_row_base = b * Q_HEAD_PAD
                all_q_padded = pl.assemble(all_q_padded, pl.cast(q_block, target_type=pl.BF16), [pad_row_base, 0])
                all_q_padded = pl.assemble(all_q_padded, pl.cast(rot_q_lo, target_type=pl.BF16), [pad_row_base, 0])
                all_q_padded = pl.assemble(all_q_padded, pl.cast(rot_q_hi, target_type=pl.BF16), [pad_row_base, HALF])
                all_q_padded = pl.assemble(
                    all_q_padded,
                    pl.cast(pl.full([Q_HEAD_PAD - Q_HEAD_BATCH, HEAD_DIM], dtype=pl.FP32, value=0.0), target_type=pl.BF16),
                    [pad_row_base + Q_HEAD_BATCH, 0],
                )
        for fa_b in pl.spmd(BATCH, name_hint="qk_consumer"):
            q_tile = pl.slice(all_q_padded, [Q_HEAD_PAD, HEAD_DIM], [fa_b * Q_HEAD_PAD, 0])
            k_tile = pl.slice(k_cache, [BLOCK_SIZE, HEAD_DIM], [0, 0])
            scores = pl.matmul(q_tile, k_tile, b_trans=True, out_dtype=pl.FP32)
            out = pl.assemble(out, scores, [fa_b, 0])
        return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='/tmp/p15_min_full_rope_consumer')
    args = parser.parse_args()
    work_dir = Path(args.out.replace('producer', 'consumer'))
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)
    set_backend_type(BackendType.Ascend910B)
    compile_program(FullRopeConsumerMin, work_dir, strategy=OptimizationStrategy.Default, backend_type=BackendType.Ascend910B)
    specs = [
        TensorSpec('k_src', [BATCH, HEAD_DIM], torch.float32, init_value=torch.ones(BATCH, HEAD_DIM, dtype=torch.float32)),
        TensorSpec('v_src', [BATCH, HEAD_DIM], torch.float32, init_value=torch.ones(BATCH, HEAD_DIM, dtype=torch.float32)),
        TensorSpec('q_src', [BATCH, Q_HEAD_BATCH * HEAD_DIM], torch.float32, init_value=torch.ones(BATCH, Q_HEAD_BATCH * HEAD_DIM, dtype=torch.float32)),
        TensorSpec('seq_lens', [BATCH], torch.int32, init_value=torch.ones(BATCH, dtype=torch.int32)),
        TensorSpec('slot_mapping', [BATCH], torch.int32, init_value=torch.arange(BATCH, dtype=torch.int32)),
        TensorSpec('rope_cos', [SEQ, ROTARY_DIM], torch.float32, init_value=torch.ones(SEQ, ROTARY_DIM, dtype=torch.float32)),
        TensorSpec('rope_sin', [SEQ, ROTARY_DIM], torch.float32, init_value=torch.zeros(SEQ, ROTARY_DIM, dtype=torch.float32)),
        TensorSpec('k_cache', [SEQ, HEAD_DIM], torch.bfloat16, is_output=True),
        TensorSpec('v_cache', [SEQ, HEAD_DIM], torch.bfloat16, is_output=True),
        TensorSpec('out', [BATCH, Q_HEAD_PAD], torch.float32, is_output=True),
    ]
    golden_src = 'def compute_golden(tensors, params):\n    tensors["k_cache"][:] = 0\n    tensors["v_cache"][:] = 0\n    tensors["out"][:] = 0\n'
    (work_dir / 'golden.py').write_text(generate_golden_source(specs, None, 1e9, 1e9, compute_golden_src=golden_src))
    in_data = {s.name: s.init_value.clone() for s in specs if not s.is_output}
    _save_data_files(in_data, work_dir / 'data' / 'in')
    _save_data_files({s.name: torch.zeros(*s.shape, dtype=s.dtype) for s in specs if s.is_output}, work_dir / 'data' / 'out')
    chip_callable, runtime_name, _ = compile_and_assemble(work_dir, 'a2a3')
    timing = _execute_on_device(work_dir, work_dir / 'golden.py', chip_callable, runtime_name, 'a2a3', 0, dfx=_DfxOpts())
    print('PASS', timing)

if __name__ == '__main__':
    main()
