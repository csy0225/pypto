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
LAYER_DYN = 45
K_CHUNK = 256
KV_CHUNK = 256
Q_OUT_CHUNK = 256
EPS = 1.0e-6
HIDDEN_INV = 1.0 / 4096.0
HEAD_DIM_INV = 1.0 / 128.0

@pl.program
class FullRopeTrueCacheId6:

    @pl.function(type=pl.FunctionType.InCore)
    def gate_proj(
        self,
        x: pl.Tensor[[BATCH, 256], pl.BF16],
        w: pl.Tensor[[256, 16], pl.BF16],
        gate: pl.Out[pl.Tensor[[BATCH, 16], pl.FP32]],
    ) -> pl.Tensor[[BATCH, 16], pl.FP32]:
        acc = pl.matmul(x, w, out_dtype=pl.FP32)
        gate = pl.assemble(gate, acc, [0, 0])
        return gate

    @pl.function(type=pl.FunctionType.Orchestration)
    def chip_orch(
        self,
        current_hidden: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
        input_rms_weight: pl.Tensor[[LAYER_DYN, HIDDEN], pl.FP32],
        wq: pl.Tensor[[HIDDEN, HIDDEN_Q], pl.BF16],
        wk: pl.Tensor[[HIDDEN, KV_HIDDEN], pl.BF16],
        wv: pl.Tensor[[HIDDEN, KV_HIDDEN], pl.BF16],
        q_norm_weight: pl.Tensor[[LAYER_DYN, HEAD_DIM], pl.FP32],
        k_norm_weight: pl.Tensor[[LAYER_DYN, HEAD_DIM], pl.FP32],
        seq_lens: pl.Tensor[[BATCH], pl.INT32],
        slot_mapping: pl.Tensor[[BATCH], pl.INT32],
        rope_cos: pl.Tensor[[SEQ, ROTARY_DIM], pl.FP32],
        rope_sin: pl.Tensor[[SEQ, ROTARY_DIM], pl.FP32],
        gate_x: pl.Tensor[[BATCH, 256], pl.BF16],
        gate_w: pl.Tensor[[256, 16], pl.BF16],
        gate_out: pl.Out[pl.Tensor[[BATCH, 16], pl.FP32]],
        k_cache: pl.Out[pl.Tensor[[SEQ, HEAD_DIM], pl.BF16]],
        v_cache: pl.Out[pl.Tensor[[SEQ, HEAD_DIM], pl.BF16]],
        all_q_padded: pl.Out[pl.Tensor[[BATCH * Q_HEAD_PAD, HEAD_DIM], pl.BF16]],
        layer_idx: pl.Scalar[pl.INT32],
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
                gamma = pl.slice(input_rms_weight, [1, K_CHUNK], [layer_idx, k0])
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
            gamma_q = pl.slice(q_norm_weight, [1, HEAD_DIM], [layer_idx, 0])
            for h in pl.range(Q_HEAD_BATCH):
                c0 = h * HEAD_DIM
                q = pl.slice(q_proj, [BATCH, HEAD_DIM], [0, c0])
                q_inv = pl.rsqrt(pl.add(pl.mul(pl.row_sum(pl.mul(q, q)), HEAD_DIM_INV), EPS))
                qn = pl.col_expand_mul(pl.row_expand_mul(q, q_inv), pl.add(gamma_q, 1.0))
                q_proj_norm = pl.assemble(q_proj_norm, qn, [0, c0])
            k = pl.slice(k_proj, [BATCH, HEAD_DIM], [0, 0])
            gamma_k = pl.slice(k_norm_weight, [1, HEAD_DIM], [layer_idx, 0])
            k_inv = pl.rsqrt(pl.add(pl.mul(pl.row_sum(pl.mul(k, k)), HEAD_DIM_INV), EPS))
            kn = pl.col_expand_mul(pl.row_expand_mul(k, k_inv), pl.add(gamma_k, 1.0))
            k_proj_norm = pl.assemble(k_proj_norm, kn, [0, 0])

        gate_out = self.gate_proj(gate_x, gate_w, gate_out)
        decode_layer_cache_rows = pl.tensor.dim(k_cache, 0) // LAYER_DYN
        layer_cache_base = layer_idx * decode_layer_cache_rows
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
                cache_row = layer_cache_base + slot_block * BLOCK_SIZE + slot_offset
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


    @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
    def host_orch(
        self,
        current_hidden: pl.Tensor[[1, BATCH, HIDDEN], pl.BF16],
        input_rms_weight: pl.Tensor[[1, LAYER_DYN, HIDDEN], pl.FP32],
        wq: pl.Tensor[[1, HIDDEN, HIDDEN_Q], pl.BF16],
        wk: pl.Tensor[[1, HIDDEN, KV_HIDDEN], pl.BF16],
        wv: pl.Tensor[[1, HIDDEN, KV_HIDDEN], pl.BF16],
        q_norm_weight: pl.Tensor[[1, LAYER_DYN, HEAD_DIM], pl.FP32],
        k_norm_weight: pl.Tensor[[1, LAYER_DYN, HEAD_DIM], pl.FP32],
        seq_lens: pl.Tensor[[1, BATCH], pl.INT32],
        slot_mapping: pl.Tensor[[1, BATCH], pl.INT32],
        rope_cos: pl.Tensor[[1, SEQ, ROTARY_DIM], pl.FP32],
        rope_sin: pl.Tensor[[1, SEQ, ROTARY_DIM], pl.FP32],
        gate_x: pl.Tensor[[1, BATCH, 256], pl.BF16],
        gate_w: pl.Tensor[[1, 256, 16], pl.BF16],
        gate_out: pl.Out[pl.Tensor[[1, BATCH, 16], pl.FP32]],
        k_cache: pl.Out[pl.Tensor[[1, SEQ, HEAD_DIM], pl.BF16]],
        v_cache: pl.Out[pl.Tensor[[1, SEQ, HEAD_DIM], pl.BF16]],
        all_q_padded: pl.Out[pl.Tensor[[1, BATCH * Q_HEAD_PAD, HEAD_DIM], pl.BF16]],
        layer_idx: pl.Scalar[pl.INT32],
    ) -> pl.Tensor[[1, BATCH * Q_HEAD_PAD, HEAD_DIM], pl.BF16]:
        self.chip_orch(
            pl.tensor.slice(current_hidden, [1, BATCH, HIDDEN], [0, 0, 0], [], [0]),
            pl.tensor.slice(input_rms_weight, [1, LAYER_DYN, HIDDEN], [0, 0, 0], [], [0]),
            pl.tensor.slice(wq, [1, HIDDEN, HIDDEN_Q], [0, 0, 0], [], [0]),
            pl.tensor.slice(wk, [1, HIDDEN, KV_HIDDEN], [0, 0, 0], [], [0]),
            pl.tensor.slice(wv, [1, HIDDEN, KV_HIDDEN], [0, 0, 0], [], [0]),
            pl.tensor.slice(q_norm_weight, [1, LAYER_DYN, HEAD_DIM], [0, 0, 0], [], [0]),
            pl.tensor.slice(k_norm_weight, [1, LAYER_DYN, HEAD_DIM], [0, 0, 0], [], [0]),
            pl.tensor.slice(seq_lens, [1, BATCH], [0, 0], [], [0]),
            pl.tensor.slice(slot_mapping, [1, BATCH], [0, 0], [], [0]),
            pl.tensor.slice(rope_cos, [1, SEQ, ROTARY_DIM], [0, 0, 0], [], [0]),
            pl.tensor.slice(rope_sin, [1, SEQ, ROTARY_DIM], [0, 0, 0], [], [0]),
            pl.tensor.slice(gate_x, [1, BATCH, 256], [0, 0, 0], [], [0]),
            pl.tensor.slice(gate_w, [1, 256, 16], [0, 0, 0], [], [0]),
            pl.tensor.slice(gate_out, [1, BATCH, 16], [0, 0, 0], [], [0]),
            pl.tensor.slice(k_cache, [1, SEQ, HEAD_DIM], [0, 0, 0], [], [0]),
            pl.tensor.slice(v_cache, [1, SEQ, HEAD_DIM], [0, 0, 0], [], [0]),
            pl.tensor.slice(all_q_padded, [1, BATCH * Q_HEAD_PAD, HEAD_DIM], [0, 0, 0], [], [0]),
            layer_idx,
            device=0,
        )
        return all_q_padded


def main():
    work_dir = Path('/tmp/p15_min_rope_real_prod_bd24')
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)
    set_backend_type(BackendType.Ascend910B)
    compiled = ir.compile(
        FullRopeTrueCacheId6,
        platform='a2a3',
        distributed_config=DistributedConfig(device_ids=[0], num_sub_workers=0, block_dim=24, aicpu_thread_num=4),
    )
    g = torch.Generator().manual_seed(0)
    tensors = {
        'current_hidden': torch.zeros(1, BATCH, HIDDEN, dtype=torch.bfloat16),
        'input_rms_weight': torch.ones(1, LAYER_DYN, HIDDEN, dtype=torch.float32),
        'wq': (0.001 * torch.randn(1, HIDDEN, HIDDEN_Q, generator=g)).bfloat16(),
        'wk': (0.001 * torch.randn(1, HIDDEN, KV_HIDDEN, generator=g)).bfloat16(),
        'wv': (0.001 * torch.randn(1, HIDDEN, KV_HIDDEN, generator=g)).bfloat16(),
        'q_norm_weight': torch.ones(1, LAYER_DYN, HEAD_DIM, dtype=torch.float32),
        'k_norm_weight': torch.ones(1, LAYER_DYN, HEAD_DIM, dtype=torch.float32),
        'seq_lens': torch.ones(1, BATCH, dtype=torch.int32),
        'slot_mapping': torch.arange(BATCH, dtype=torch.int32).unsqueeze(0),
        'rope_cos': torch.ones(1, SEQ, ROTARY_DIM, dtype=torch.float32),
        'rope_sin': torch.zeros(1, SEQ, ROTARY_DIM, dtype=torch.float32),
        'gate_x': torch.ones(1, BATCH, 256, dtype=torch.bfloat16),
        'gate_w': torch.ones(1, 256, 16, dtype=torch.bfloat16),
        'gate_out': torch.zeros(1, BATCH, 16, dtype=torch.float32),
        'k_cache': torch.zeros(1, SEQ, HEAD_DIM, dtype=torch.bfloat16),
        'v_cache': torch.zeros(1, SEQ, HEAD_DIM, dtype=torch.bfloat16),
        'all_q_padded': torch.zeros(1, BATCH * Q_HEAD_PAD, HEAD_DIM, dtype=torch.bfloat16),
    }
    compiled(
        tensors['current_hidden'], tensors['input_rms_weight'], tensors['wq'], tensors['wk'], tensors['wv'],
        tensors['q_norm_weight'], tensors['k_norm_weight'], tensors['seq_lens'], tensors['slot_mapping'],
        tensors['rope_cos'], tensors['rope_sin'], tensors['gate_x'], tensors['gate_w'], tensors['gate_out'], tensors['k_cache'], tensors['v_cache'], tensors['all_q_padded'], torch.tensor(0, dtype=torch.int32),
    )
    print('PASS')

if __name__ == '__main__':
    main()
