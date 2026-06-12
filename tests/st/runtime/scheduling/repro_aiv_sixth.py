from pathlib import Path
import tempfile
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
TASKS = 8

@pl.program
class AivSixthRepro:
    @pl.function(type=pl.FunctionType.InCore)
    def k0(self, out: pl.Out[pl.Tensor[[TASKS, N], pl.BF16]]) -> pl.Tensor[[TASKS, N], pl.BF16]:
        z = pl.cast(pl.full([1, N], dtype=pl.FP32, value=0.0), target_type=pl.BF16)
        out = pl.assemble(out, z, [0, 0])
        return out

    @pl.function(type=pl.FunctionType.InCore)
    def k1(self, out: pl.Out[pl.Tensor[[TASKS, N], pl.BF16]]) -> pl.Tensor[[TASKS, N], pl.BF16]:
        z = pl.cast(pl.full([1, N], dtype=pl.FP32, value=1.0), target_type=pl.BF16)
        out = pl.assemble(out, z, [1, 0])
        return out

    @pl.function(type=pl.FunctionType.InCore)
    def k2(self, out: pl.Out[pl.Tensor[[TASKS, N], pl.BF16]]) -> pl.Tensor[[TASKS, N], pl.BF16]:
        z = pl.cast(pl.full([1, N], dtype=pl.FP32, value=2.0), target_type=pl.BF16)
        out = pl.assemble(out, z, [2, 0])
        return out

    @pl.function(type=pl.FunctionType.InCore)
    def k3(self, out: pl.Out[pl.Tensor[[TASKS, N], pl.BF16]]) -> pl.Tensor[[TASKS, N], pl.BF16]:
        z = pl.cast(pl.full([1, N], dtype=pl.FP32, value=3.0), target_type=pl.BF16)
        out = pl.assemble(out, z, [3, 0])
        return out

    @pl.function(type=pl.FunctionType.InCore)
    def k4(self, out: pl.Out[pl.Tensor[[TASKS, N], pl.BF16]]) -> pl.Tensor[[TASKS, N], pl.BF16]:
        z = pl.cast(pl.full([1, N], dtype=pl.FP32, value=4.0), target_type=pl.BF16)
        out = pl.assemble(out, z, [4, 0])
        return out

    @pl.function(type=pl.FunctionType.InCore)
    def k5(self, out: pl.Out[pl.Tensor[[TASKS, N], pl.BF16]]) -> pl.Tensor[[TASKS, N], pl.BF16]:
        z = pl.cast(pl.full([1, N], dtype=pl.FP32, value=5.0), target_type=pl.BF16)
        out = pl.assemble(out, z, [5, 0])
        return out

    @pl.function(type=pl.FunctionType.InCore)
    def k6(self, out: pl.Out[pl.Tensor[[TASKS, N], pl.BF16]]) -> pl.Tensor[[TASKS, N], pl.BF16]:
        z = pl.cast(pl.full([1, N], dtype=pl.FP32, value=6.0), target_type=pl.BF16)
        out = pl.assemble(out, z, [6, 0])
        return out

    @pl.function(type=pl.FunctionType.InCore)
    def k7(self, out: pl.Out[pl.Tensor[[TASKS, N], pl.BF16]]) -> pl.Tensor[[TASKS, N], pl.BF16]:
        z = pl.cast(pl.full([1, N], dtype=pl.FP32, value=7.0), target_type=pl.BF16)
        out = pl.assemble(out, z, [7, 0])
        return out

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(self, out: pl.Out[pl.Tensor[[TASKS, N], pl.BF16]]) -> pl.Tensor[[TASKS, N], pl.BF16]:
        out = self.k0(out)
        out = self.k1(out)
        out = self.k2(out)
        out = self.k3(out)
        out = self.k4(out)
        out = self.k5(out)
        out = self.k6(out)
        out = self.k7(out)
        return out


def main():
    work_dir = Path('/tmp/p15_min_aiv6')
    if work_dir.exists():
        import shutil; shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)
    set_backend_type(BackendType.Ascend910B)
    compile_program(AivSixthRepro, work_dir, strategy=OptimizationStrategy.Default, backend_type=BackendType.Ascend910B)

    specs = [TensorSpec('out', [TASKS, N], torch.bfloat16, is_output=True)]
    golden = generate_golden_source(specs, None, 1e-5, 1e-5, compute_golden_src='def compute_golden(tensors, params):\n    import torch\n    for i in range(8):\n        tensors["out"][i, :] = i\n')
    (work_dir / 'golden.py').write_text(golden)
    _save_data_files({}, work_dir / 'data' / 'in')
    _save_data_files({'out': torch.zeros(TASKS, N, dtype=torch.bfloat16)}, work_dir / 'data' / 'out')

    chip_callable, runtime_name, _ = compile_and_assemble(work_dir, 'a2a3')
    timing = _execute_on_device(work_dir, work_dir / 'golden.py', chip_callable, runtime_name, 'a2a3', 0, dfx=_DfxOpts())
    print('PASS', timing)

if __name__ == '__main__':
    main()
