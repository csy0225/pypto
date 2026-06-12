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

N = 32
ROWS = 8

@pl.program
class RopeKernel6Repro:
    @pl.function(type=pl.FunctionType.InCore)
    def k0(self, out: pl.Out[pl.Tensor[[ROWS, N], pl.BF16]]) -> pl.Tensor[[ROWS, N], pl.BF16]:
        z = pl.cast(pl.full([1, N], dtype=pl.FP32, value=0.0), target_type=pl.BF16)
        out = pl.assemble(out, z, [0, 0])
        return out

    @pl.function(type=pl.FunctionType.InCore)
    def k1(self, out: pl.Out[pl.Tensor[[ROWS, N], pl.BF16]]) -> pl.Tensor[[ROWS, N], pl.BF16]:
        z = pl.cast(pl.full([1, N], dtype=pl.FP32, value=1.0), target_type=pl.BF16)
        out = pl.assemble(out, z, [1, 0])
        return out

    @pl.function(type=pl.FunctionType.InCore)
    def k2(self, out: pl.Out[pl.Tensor[[ROWS, N], pl.BF16]]) -> pl.Tensor[[ROWS, N], pl.BF16]:
        z = pl.cast(pl.full([1, N], dtype=pl.FP32, value=2.0), target_type=pl.BF16)
        out = pl.assemble(out, z, [2, 0])
        return out

    @pl.function(type=pl.FunctionType.InCore)
    def k3(self, out: pl.Out[pl.Tensor[[ROWS, N], pl.BF16]]) -> pl.Tensor[[ROWS, N], pl.BF16]:
        z = pl.cast(pl.full([1, N], dtype=pl.FP32, value=3.0), target_type=pl.BF16)
        out = pl.assemble(out, z, [3, 0])
        return out

    @pl.function(type=pl.FunctionType.InCore)
    def k4(self, out: pl.Out[pl.Tensor[[ROWS, N], pl.BF16]]) -> pl.Tensor[[ROWS, N], pl.BF16]:
        z = pl.cast(pl.full([1, N], dtype=pl.FP32, value=4.0), target_type=pl.BF16)
        out = pl.assemble(out, z, [4, 0])
        return out

    @pl.function(type=pl.FunctionType.InCore)
    def k5(self, out: pl.Out[pl.Tensor[[ROWS, N], pl.BF16]]) -> pl.Tensor[[ROWS, N], pl.BF16]:
        z = pl.cast(pl.full([1, N], dtype=pl.FP32, value=5.0), target_type=pl.BF16)
        out = pl.assemble(out, z, [5, 0])
        return out

    @pl.function(type=pl.FunctionType.InCore)
    def rope_like(
        self,
        a: pl.Tensor[[1, N], pl.FP32],
        b: pl.Tensor[[1, N], pl.FP32],
        c: pl.Tensor[[1, N], pl.FP32],
        d: pl.Tensor[[1, N], pl.FP32],
        out: pl.Out[pl.Tensor[[ROWS, N], pl.BF16]],
    ) -> pl.Tensor[[ROWS, N], pl.BF16]:
        lo = pl.slice(a, [1, N], [0, 0])
        hi = pl.slice(b, [1, N], [0, 0])
        cos = pl.slice(c, [1, N], [0, 0])
        sin = pl.slice(d, [1, N], [0, 0])
        x = pl.sub(pl.col_expand_mul(lo, cos), pl.col_expand_mul(hi, sin))
        y = pl.add(pl.col_expand_mul(hi, cos), pl.col_expand_mul(lo, sin))
        out = pl.assemble(out, pl.cast(x, target_type=pl.BF16), [6, 0])
        out = pl.assemble(out, pl.cast(y, target_type=pl.BF16), [7, 0])
        return out

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        a: pl.Tensor[[1, N], pl.FP32],
        b: pl.Tensor[[1, N], pl.FP32],
        c: pl.Tensor[[1, N], pl.FP32],
        d: pl.Tensor[[1, N], pl.FP32],
        out: pl.Out[pl.Tensor[[ROWS, N], pl.BF16]],
    ) -> pl.Tensor[[ROWS, N], pl.BF16]:
        out = self.k0(out)
        out = self.k1(out)
        out = self.k2(out)
        out = self.k3(out)
        out = self.k4(out)
        out = self.k5(out)
        out = self.rope_like(a, b, c, d, out)
        return out


def main():
    work_dir = Path('/tmp/p15_min_rope6')
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)
    set_backend_type(BackendType.Ascend910B)
    compile_program(RopeKernel6Repro, work_dir, strategy=OptimizationStrategy.Default, backend_type=BackendType.Ascend910B)

    specs = [
        TensorSpec('a', [1, N], torch.float32, init_value=torch.ones(1, N, dtype=torch.float32)),
        TensorSpec('b', [1, N], torch.float32, init_value=torch.ones(1, N, dtype=torch.float32) * 2),
        TensorSpec('c', [1, N], torch.float32, init_value=torch.ones(1, N, dtype=torch.float32) * 3),
        TensorSpec('d', [1, N], torch.float32, init_value=torch.ones(1, N, dtype=torch.float32) * 4),
        TensorSpec('out', [ROWS, N], torch.bfloat16, is_output=True),
    ]
    golden_src = '''def compute_golden(tensors, params):
    import torch
    tensors["out"][:] = 0
    for i in range(6):
        tensors["out"][i, :] = i
    x = tensors["a"] * tensors["c"] - tensors["b"] * tensors["d"]
    y = tensors["b"] * tensors["c"] + tensors["a"] * tensors["d"]
    tensors["out"][6, :] = x.to(torch.bfloat16)
    tensors["out"][7, :] = y.to(torch.bfloat16)
'''
    (work_dir / 'golden.py').write_text(generate_golden_source(specs, None, 1e-5, 1e-5, compute_golden_src=golden_src))
    in_data = {s.name: s.init_value.clone() for s in specs if not s.is_output}
    _save_data_files(in_data, work_dir / 'data' / 'in')
    out = torch.zeros(ROWS, N, dtype=torch.bfloat16)
    for i in range(6):
        out[i, :] = i
    out[6, :] = (in_data['a'] * in_data['c'] - in_data['b'] * in_data['d']).to(torch.bfloat16)
    out[7, :] = (in_data['b'] * in_data['c'] + in_data['a'] * in_data['d']).to(torch.bfloat16)
    _save_data_files({'out': out}, work_dir / 'data' / 'out')

    chip_callable, runtime_name, _ = compile_and_assemble(work_dir, 'a2a3')
    timing = _execute_on_device(work_dir, work_dir / 'golden.py', chip_callable, runtime_name, 'a2a3', 0, dfx=_DfxOpts())
    print('PASS', timing)

if __name__ == '__main__':
    main()
