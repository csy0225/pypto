from pathlib import Path
import shutil
import torch

import pypto.language as pl
from pypto.backend import BackendType, set_backend_type
from pypto import ir
from pypto.ir.distributed_compiled_program import DistributedConfig

BATCH = 16
HIDDEN = 4096
HEAD_DIM = 128
HIDDEN_Q = 1024
KV_HIDDEN = 128
K_CHUNK = 256
Q_OUT_CHUNK = 256
LAYER_DYN = 45
SEQ = 4096
BATCH_TILE = 16

@pl.program
class FullrowTask6:
    @pl.function(type=pl.FunctionType.Orchestration)
    def chip_orch(
        self,
        current_hidden: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
        wq: pl.Tensor[[HIDDEN, HIDDEN_Q], pl.BF16],
        wk: pl.Tensor[[HIDDEN, KV_HIDDEN], pl.BF16],
        wv: pl.Tensor[[HIDDEN, KV_HIDDEN], pl.BF16],
        w_g: pl.Tensor[[HIDDEN, 16], pl.BF16],
        k_cache: pl.Out[pl.Tensor[[SEQ, HEAD_DIM], pl.BF16]],
    ) -> pl.Tensor[[SEQ, HEAD_DIM], pl.BF16]:
        normed_all = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
        q_proj = pl.create_tensor([BATCH, HIDDEN_Q], dtype=pl.FP32)
        k_proj = pl.create_tensor([BATCH, KV_HIDDEN], dtype=pl.FP32)
        v_proj = pl.create_tensor([BATCH, KV_HIDDEN], dtype=pl.FP32)
        k_proj_norm = pl.create_tensor([BATCH, HEAD_DIM], dtype=pl.FP32)
        gate_logits = pl.create_tensor([BATCH, 16], dtype=pl.FP32)

        with pl.at(level=pl.Level.CORE_GROUP, name_hint="dummy_rmsnorm_zc"):
            for kb in pl.pipeline(HIDDEN // K_CHUNK, stage=4):
                k0 = kb * K_CHUNK
                normed_all = pl.assemble(normed_all, pl.slice(current_hidden, [BATCH, K_CHUNK], [0, k0]), [0, k0])

        for qi in pl.spmd(HIDDEN_Q // Q_OUT_CHUNK, name_hint="dummy_q_proj"):
            o0 = qi * Q_OUT_CHUNK
            q_acc = pl.matmul(pl.slice(normed_all, [BATCH, K_CHUNK], [0, 0]), pl.slice(wq, [K_CHUNK, Q_OUT_CHUNK], [0, o0]), out_dtype=pl.FP32)
            q_proj = pl.assemble(q_proj, q_acc, [0, o0])

        for _ in pl.spmd(1, name_hint="dummy_k_proj"):
            k_acc = pl.matmul(pl.slice(normed_all, [BATCH, K_CHUNK], [0, 0]), pl.slice(wk, [K_CHUNK, KV_HIDDEN], [0, 0]), out_dtype=pl.FP32)
            k_proj = pl.assemble(k_proj, k_acc, [0, 0])

        for _ in pl.spmd(1, name_hint="dummy_v_proj"):
            v_acc = pl.matmul(pl.slice(normed_all, [BATCH, K_CHUNK], [0, 0]), pl.slice(wv, [K_CHUNK, KV_HIDDEN], [0, 0]), out_dtype=pl.FP32)
            v_proj = pl.assemble(v_proj, v_acc, [0, 0])

        for _ in pl.spmd(1, name_hint="dummy_qk_norm_zc"):
            k_proj_norm = pl.add(k_proj, 0.0)

        for _ in pl.spmd(1, name_hint="dummy_gate_proj"):
            gate_acc = pl.matmul(pl.slice(current_hidden, [BATCH, K_CHUNK], [0, 0]), pl.slice(w_g, [K_CHUNK, 16], [0, 0]), out_dtype=pl.FP32)
            gate_logits = pl.assemble(gate_logits, gate_acc, [0, 0])

        for b in pl.parallel(BATCH):
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="full_k_cache_fullrow"):
                k_cache = pl.assemble(k_cache, pl.cast(pl.slice(k_proj_norm, [1, HEAD_DIM], [b, 0]), target_type=pl.BF16), [b, 0])
        return k_cache

    @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
    def host_orch(
        self,
        current_hidden: pl.Tensor[[1, BATCH, HIDDEN], pl.BF16],
        wq: pl.Tensor[[1, HIDDEN, HIDDEN_Q], pl.BF16],
        wk: pl.Tensor[[1, HIDDEN, KV_HIDDEN], pl.BF16],
        wv: pl.Tensor[[1, HIDDEN, KV_HIDDEN], pl.BF16],
        w_g: pl.Tensor[[1, HIDDEN, 16], pl.BF16],
        k_cache: pl.Out[pl.Tensor[[1, SEQ, HEAD_DIM], pl.BF16]],
    ) -> pl.Tensor[[1, SEQ, HEAD_DIM], pl.BF16]:
        self.chip_orch(
            pl.tensor.slice(current_hidden, [1, BATCH, HIDDEN], [0, 0, 0], [], [0]),
            pl.tensor.slice(wq, [1, HIDDEN, HIDDEN_Q], [0, 0, 0], [], [0]),
            pl.tensor.slice(wk, [1, HIDDEN, KV_HIDDEN], [0, 0, 0], [], [0]),
            pl.tensor.slice(wv, [1, HIDDEN, KV_HIDDEN], [0, 0, 0], [], [0]),
            pl.tensor.slice(w_g, [1, HIDDEN, 16], [0, 0, 0], [], [0]),
            pl.tensor.slice(k_cache, [1, SEQ, HEAD_DIM], [0, 0, 0], [], [0]),
            device=0,
        )
        return k_cache


def main():
    work_dir = Path('/tmp/p15_min_fullrow_task6')
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)
    set_backend_type(BackendType.Ascend910B)
    compiled = ir.compile(
        FullrowTask6,
        platform='a2a3',
        distributed_config=DistributedConfig(device_ids=[0], num_sub_workers=0, block_dim=24, aicpu_thread_num=4),
    )
    g = torch.Generator().manual_seed(0)
    tensors = {
        'current_hidden': torch.zeros(1, BATCH, HIDDEN, dtype=torch.bfloat16),
        'wq': (0.001 * torch.randn(1, HIDDEN, HIDDEN_Q, generator=g)).bfloat16(),
        'wk': (0.001 * torch.randn(1, HIDDEN, KV_HIDDEN, generator=g)).bfloat16(),
        'wv': (0.001 * torch.randn(1, HIDDEN, KV_HIDDEN, generator=g)).bfloat16(),
        'w_g': (0.001 * torch.randn(1, HIDDEN, 16, generator=g)).bfloat16(),
        'k_cache': torch.zeros(1, SEQ, HEAD_DIM, dtype=torch.bfloat16),
    }
    compiled(tensors['current_hidden'], tensors['wq'], tensors['wk'], tensors['wv'], tensors['w_g'], tensors['k_cache'])
    print('PASS')

if __name__ == '__main__':
    main()
