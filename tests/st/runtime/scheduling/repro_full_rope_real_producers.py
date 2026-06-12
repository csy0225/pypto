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
K_CHUNK = 256
KV_CHUNK = 256
Q_OUT_CHUNK = 256
EPS = 1.0e-6
HIDDEN_INV = 1.0 / 4096.0
HEAD_DIM_INV = 1.0 / 128.0

@pl.program
class FullRopeRealProducers:
    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        current_hidden: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
        input_rms_weight: pl.Tensor[[1, HIDDEN], pl.FP32],
        wq: pl.Tensor[[HIDDEN, HIDDEN_Q], pl.BF16],
        wk: pl.Tensor[[HIDDEN, KV_HIDDEN], pl.BF16],
        wv: pl.Tensor[[HIDDEN, KV_HIDDEN], pl.BF16],
        q_norm_weight: pl.Tensor[[1, HEAD_DIM], pl.FP32],
        k_norm_weight: pl.Tensor[[1, HEAD_DIM], pl.FP32],
        seq_lens: pl.Tensor[[BATCH], pl.INT32],
        slot_mapping: pl.Tensor[[BATCH], pl.INT32],
        rope_cos: pl.Tensor[[SEQ, ROTARY_DIM], pl.FP32],
        rope_sin: pl.Tensor[[SEQ, ROTARY_DIM], pl.FP32],
        k_cache: pl.Out[pl.Tensor[[SEQ, HEAD_DIM], pl.BF16]],
        v_cache: pl.Out[pl.Tensor[[SEQ, HEAD_DIM], pl.BF16]],
        all_q_padded: pl.Out[pl.Tensor[[BATCH * Q_HEAD_PAD, HEAD_DIM], pl.BF16]],
    ) -> pl.Tensor[[BATCH * Q_HEAD_PAD, HEAD_DIM], pl.BF16]:
        normed_all = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
        q_proj = pl.create_tensor([BATCH, HIDDEN_Q], dtype=pl.FP32)
        k_proj = pl.create_tensor([BATCH, KV_HIDDEN], dtype=pl.FP32)
        v_proj = pl.create_tensor([BATCH, KV_HIDDEN], dtype=pl.FP32)
        q_proj_norm = pl.create_tensor([BATCH, HIDDEN_Q], dtype=pl.FP32)
        k_proj_norm = pl.create_tensor([BATCH, KV_HIDDEN], dtype=pl.FP32)

        with pl.at(level=pl.Level.CORE_GROUP, name_hint="mini_rmsnorm_zc"):
            partial_sq = pl.full([1, BATCH], dtype=pl.FP32, value=0.0)
            for kb in pl.pipeline(HIDDEN // K_CHUNK, stage=4):
                k0 = kb * K_CHUNK
                x = pl.cast(pl.slice(current_hidden, [BATCH, K_CHUNK], [0, k0]), target_type=pl.FP32)
                partial_sq = pl.add(partial_sq, pl.reshape(pl.row_sum(pl.mul(x, x)), [1, BATCH]))
            inv = pl.recip(pl.sqrt(pl.reshape(pl.add(pl.mul(partial_sq, HIDDEN_INV), EPS), [BATCH, 1])))
            for kb in pl.pipeline(HIDDEN // K_CHUNK, stage=4):
                k0 = kb * K_CHUNK
                x = pl.cast(pl.slice(current_hidden, [BATCH, K_CHUNK], [0, k0]), target_type=pl.FP32)
                gamma = pl.slice(input_rms_weight, [1, K_CHUNK], [0, k0])
                y = pl.col_expand_mul(pl.row_expand_mul(x, inv), pl.add(gamma, 1.0))
                normed_all = pl.assemble(normed_all, pl.cast(y, target_type=pl.BF16), [0, k0])

        for qi in pl.spmd(HIDDEN_Q // Q_OUT_CHUNK, name_hint="mini_q_proj"):
            o0 = qi * Q_OUT_CHUNK
            q_acc = pl.matmul(pl.slice(normed_all, [BATCH, K_CHUNK], [0, 0]), pl.slice(wq, [K_CHUNK, Q_OUT_CHUNK], [0, o0]), out_dtype=pl.FP32)
            for kb in pl.range(1, HIDDEN // K_CHUNK):
                k0 = kb * K_CHUNK
                q_acc = pl.matmul_acc(q_acc, pl.slice(normed_all, [BATCH, K_CHUNK], [0, k0]), pl.slice(wq, [K_CHUNK, Q_OUT_CHUNK], [k0, o0]))
            q_proj = pl.assemble(q_proj, q_acc, [0, o0])

        for _ in pl.spmd(1, name_hint="mini_k_proj"):
            k_acc = pl.matmul(pl.slice(normed_all, [BATCH, KV_CHUNK], [0, 0]), pl.slice(wk, [KV_CHUNK, KV_HIDDEN], [0, 0]), out_dtype=pl.FP32)
            for kb in pl.range(1, HIDDEN // KV_CHUNK):
                k0 = kb * KV_CHUNK
                k_acc = pl.matmul_acc(k_acc, pl.slice(normed_all, [BATCH, KV_CHUNK], [0, k0]), pl.slice(wk, [KV_CHUNK, KV_HIDDEN], [k0, 0]))
            k_proj = pl.assemble(k_proj, k_acc, [0, 0])

        for _ in pl.spmd(1, name_hint="mini_v_proj"):
            v_acc = pl.matmul(pl.slice(normed_all, [BATCH, KV_CHUNK], [0, 0]), pl.slice(wv, [KV_CHUNK, KV_HIDDEN], [0, 0]), out_dtype=pl.FP32)
            for kb in pl.range(1, HIDDEN // KV_CHUNK):
                k0 = kb * KV_CHUNK
                v_acc = pl.matmul_acc(v_acc, pl.slice(normed_all, [BATCH, KV_CHUNK], [0, k0]), pl.slice(wv, [KV_CHUNK, KV_HIDDEN], [k0, 0]))
            v_proj = pl.assemble(v_proj, v_acc, [0, 0])

        for _ in pl.spmd(1, name_hint="mini_qk_norm_zc"):
            gamma_q = pl.slice(q_norm_weight, [1, HEAD_DIM], [0, 0])
            for h in pl.range(Q_HEAD_BATCH):
                c0 = h * HEAD_DIM
                q = pl.slice(q_proj, [BATCH, HEAD_DIM], [0, c0])
                q_inv = pl.rsqrt(pl.add(pl.mul(pl.row_sum(pl.mul(q, q)), HEAD_DIM_INV), EPS))
                qn = pl.col_expand_mul(pl.row_expand_mul(q, q_inv), pl.add(gamma_q, 1.0))
                q_proj_norm = pl.assemble(q_proj_norm, qn, [0, c0])
            k = pl.slice(k_proj, [BATCH, HEAD_DIM], [0, 0])
            gamma_k = pl.slice(k_norm_weight, [1, HEAD_DIM], [0, 0])
            k_inv = pl.rsqrt(pl.add(pl.mul(pl.row_sum(pl.mul(k, k)), HEAD_DIM_INV), EPS))
            kn = pl.col_expand_mul(pl.row_expand_mul(k, k_inv), pl.add(gamma_k, 1.0))
            k_proj_norm = pl.assemble(k_proj_norm, kn, [0, 0])

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
                k_cache = pl.assemble(k_cache, pl.cast(k_proj_norm[b:b+1, :], target_type=pl.BF16), [cache_row, 0])
                k_cache = pl.assemble(k_cache, pl.cast(rot_k_lo, target_type=pl.BF16), [cache_row, 0])
                k_cache = pl.assemble(k_cache, pl.cast(rot_k_hi, target_type=pl.BF16), [cache_row, HALF])
                v_cache = pl.assemble(v_cache, pl.cast(v_proj[b:b+1, :], target_type=pl.BF16), [cache_row, 0])
                q_block = pl.reshape(q_proj_norm[b:b+1, 0:Q_HEAD_BATCH * HEAD_DIM], [Q_HEAD_BATCH, HEAD_DIM])
                q_lo = q_block[:, 0:HALF]
                q_hi = q_block[:, HALF:ROTARY_DIM]
                rot_q_lo = pl.sub(pl.col_expand_mul(q_lo, cos_lo), pl.col_expand_mul(q_hi, sin_lo))
                rot_q_hi = pl.add(pl.col_expand_mul(q_hi, cos_hi), pl.col_expand_mul(q_lo, sin_hi))
                base = b * Q_HEAD_PAD
                all_q_padded = pl.assemble(all_q_padded, pl.cast(q_block, target_type=pl.BF16), [base, 0])
                all_q_padded = pl.assemble(all_q_padded, pl.cast(rot_q_lo, target_type=pl.BF16), [base, 0])
                all_q_padded = pl.assemble(all_q_padded, pl.cast(rot_q_hi, target_type=pl.BF16), [base, HALF])
                all_q_padded = pl.assemble(all_q_padded, pl.cast(pl.full([Q_HEAD_PAD - Q_HEAD_BATCH, HEAD_DIM], dtype=pl.FP32, value=0.0), target_type=pl.BF16), [base + Q_HEAD_BATCH, 0])
        return all_q_padded


def main():
    work_dir = Path('/tmp/p15_min_rope_real_prod')
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)
    set_backend_type(BackendType.Ascend910B)
    compile_program(FullRopeRealProducers, work_dir, strategy=OptimizationStrategy.Default, backend_type=BackendType.Ascend910B)
    g = torch.Generator().manual_seed(0)
    specs = [
        TensorSpec('current_hidden', [BATCH, HIDDEN], torch.bfloat16, init_value=torch.zeros(BATCH, HIDDEN, dtype=torch.bfloat16)),
        TensorSpec('input_rms_weight', [1, HIDDEN], torch.float32, init_value=torch.ones(1, HIDDEN, dtype=torch.float32)),
        TensorSpec('wq', [HIDDEN, HIDDEN_Q], torch.bfloat16, init_value=(0.001 * torch.randn(HIDDEN, HIDDEN_Q, generator=g)).bfloat16()),
        TensorSpec('wk', [HIDDEN, KV_HIDDEN], torch.bfloat16, init_value=(0.001 * torch.randn(HIDDEN, KV_HIDDEN, generator=g)).bfloat16()),
        TensorSpec('wv', [HIDDEN, KV_HIDDEN], torch.bfloat16, init_value=(0.001 * torch.randn(HIDDEN, KV_HIDDEN, generator=g)).bfloat16()),
        TensorSpec('q_norm_weight', [1, HEAD_DIM], torch.float32, init_value=torch.ones(1, HEAD_DIM, dtype=torch.float32)),
        TensorSpec('k_norm_weight', [1, HEAD_DIM], torch.float32, init_value=torch.ones(1, HEAD_DIM, dtype=torch.float32)),
        TensorSpec('seq_lens', [BATCH], torch.int32, init_value=torch.ones(BATCH, dtype=torch.int32)),
        TensorSpec('slot_mapping', [BATCH], torch.int32, init_value=torch.arange(BATCH, dtype=torch.int32)),
        TensorSpec('rope_cos', [SEQ, ROTARY_DIM], torch.float32, init_value=torch.ones(SEQ, ROTARY_DIM, dtype=torch.float32)),
        TensorSpec('rope_sin', [SEQ, ROTARY_DIM], torch.float32, init_value=torch.zeros(SEQ, ROTARY_DIM, dtype=torch.float32)),
        TensorSpec('k_cache', [SEQ, HEAD_DIM], torch.bfloat16, is_output=True),
        TensorSpec('v_cache', [SEQ, HEAD_DIM], torch.bfloat16, is_output=True),
        TensorSpec('all_q_padded', [BATCH * Q_HEAD_PAD, HEAD_DIM], torch.bfloat16, is_output=True),
    ]
    golden_src = 'def compute_golden(tensors, params):\n    tensors["k_cache"][:] = 0\n    tensors["v_cache"][:] = 0\n    tensors["all_q_padded"][:] = 0\n'
    (work_dir / 'golden.py').write_text(generate_golden_source(specs, None, 1e9, 1e9, compute_golden_src=golden_src, use_data_files=True))
    in_data = {s.name: s.init_value.clone() for s in specs if not s.is_output}
    _save_data_files(in_data, work_dir / 'data' / 'in')
    _save_data_files({s.name: torch.zeros(*s.shape, dtype=s.dtype) for s in specs if s.is_output}, work_dir / 'data' / 'out')
    chip_callable, runtime_name, _ = compile_and_assemble(work_dir, 'a2a3')
    timing = _execute_on_device(work_dir, work_dir / 'golden.py', chip_callable, runtime_name, 'a2a3', 0, dfx=_DfxOpts())
    print('PASS', timing)

if __name__ == '__main__':
    main()
