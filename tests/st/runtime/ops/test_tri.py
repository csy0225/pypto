# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Same-name hardware tests for lower/upper ``pto.ttri`` generation."""

from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import ONBOARD_PLATFORMS, DataType, PTOTestCase, TensorSpec

M = 16
N = 32
_PL_DTYPE = {
    DataType.INT8: pl.INT8,
    DataType.UINT8: pl.UINT8,
    DataType.INT16: pl.INT16,
    DataType.UINT16: pl.UINT16,
    DataType.INT32: pl.INT32,
    DataType.UINT32: pl.UINT32,
    DataType.BF16: pl.BF16,
    DataType.FP16: pl.FP16,
    DataType.FP32: pl.FP32,
}
_TORCH_DTYPE = {
    DataType.INT8: torch.int8,
    DataType.UINT8: torch.uint8,
    DataType.INT16: torch.int16,
    DataType.UINT16: torch.int16,
    DataType.INT32: torch.int32,
    DataType.UINT32: torch.int32,
    DataType.BF16: torch.bfloat16,
    DataType.FP16: torch.float16,
    DataType.FP32: torch.float32,
}


class TriTestCase(PTOTestCase):
    """Generate a triangular mask into a full or partial destination region."""

    __test__ = False

    def __init__(
        self,
        *,
        dtype: DataType = DataType.FP32,
        upper: bool = False,
        diagonal: int = 0,
        shape: tuple[int, int] = (M, N),
        valid_shape: tuple[int, int] | None = None,
        platform: str | None = None,
    ):
        super().__init__(platform=platform)
        self._dtype = dtype
        self._upper = upper
        self._diagonal = diagonal
        self._shape = shape
        self._valid_shape = valid_shape

    def get_name(self) -> str:
        side = "upper" if self._upper else "lower"
        rows, cols = self._shape
        valid = "implicit" if self._valid_shape is None else f"{self._valid_shape[0]}x{self._valid_shape[1]}"
        return f"tri_{self._dtype.value}_{side}_d{self._diagonal}_{rows}x{cols}_v{valid}".replace("-", "m")

    def define_tensors(self) -> list[TensorSpec]:
        rows, cols = self._shape
        return [
            TensorSpec(
                "out",
                [rows, cols],
                self._dtype,
                is_output=True,
                init_value=torch.zeros,
            )
        ]

    def get_program(self) -> Any:
        dtype = _PL_DTYPE[self._dtype]
        upper = self._upper
        diagonal = self._diagonal
        rows, cols = self._shape
        valid_shape = None if self._valid_shape is None else list(self._valid_shape)

        @pl.program
        class TriProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                out: pl.InOut[pl.Tensor[[rows, cols], dtype]],
            ) -> pl.Tensor[[rows, cols], dtype]:
                result: pl.Tile[[rows, cols], dtype] = pl.tile.tri(
                    diagonal,
                    [rows, cols],
                    valid_shape=valid_shape,
                    dtype=dtype,
                    upper=upper,
                )
                return pl.store(result, [0, 0], out)

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                out: pl.InOut[pl.Tensor[[rows, cols], dtype]],
            ) -> pl.Tensor[[rows, cols], dtype]:
                return self.kernel(out)

        return TriProgram

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        physical_rows, physical_cols = self._shape
        valid_rows, valid_cols = self._valid_shape or self._shape
        base = torch.ones(valid_rows, valid_cols)
        valid = (
            torch.triu(base, diagonal=self._diagonal)
            if self._upper
            else torch.tril(base, diagonal=self._diagonal)
        )
        expected = torch.zeros((physical_rows, physical_cols), dtype=_TORCH_DTYPE[self._dtype])
        expected[:valid_rows, :valid_cols] = valid.to(_TORCH_DTYPE[self._dtype])
        tensors["out"][:] = expected


class TriScalarDiagonalTestCase(PTOTestCase):
    """Generate TTRI with a runtime INT32 Scalar diagonal."""

    __test__ = False

    def __init__(self, *, platform: str | None = None):
        super().__init__(platform=platform)

    def get_name(self) -> str:
        return "tri_fp32_runtime_scalar_diagonal"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("diagonal", [1], DataType.INT32, init_value=torch.tensor([2], dtype=torch.int32)),
            TensorSpec("out", [M, N], DataType.FP32, is_output=True, init_value=torch.zeros),
        ]

    def get_program(self) -> Any:
        @pl.program
        class TriScalarDiagonalProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                diagonal: pl.Tensor[[1], pl.INT32],
                out: pl.InOut[pl.Tensor[[M, N], pl.FP32]],
            ) -> pl.Tensor[[M, N], pl.FP32]:
                diagonal_value: pl.Scalar[pl.INT32] = pl.tensor.read(diagonal, [0])
                result: pl.Tile[[M, N], pl.FP32] = pl.tile.tri(diagonal_value, [M, N], dtype=pl.FP32)
                return pl.store(result, [0, 0], out)

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                diagonal: pl.Tensor[[1], pl.INT32],
                out: pl.InOut[pl.Tensor[[M, N], pl.FP32]],
            ) -> pl.Tensor[[M, N], pl.FP32]:
                return self.kernel(diagonal, out)

        return TriScalarDiagonalProgram

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        tensors["out"][:] = torch.tril(torch.ones(M, N), diagonal=2)


class TestTri:
    """TTRI dtype, side, diagonal, and valid-shape branches."""

    @pytest.mark.parametrize("platform", ONBOARD_PLATFORMS)
    @pytest.mark.parametrize(
        "dtype",
        [
            DataType.INT16,
            DataType.UINT16,
            DataType.INT32,
            DataType.UINT32,
            DataType.FP16,
            DataType.FP32,
        ],
    )
    def test_dtypes(self, test_runner, platform, dtype):
        result = test_runner.run(TriTestCase(dtype=dtype, platform=platform))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.platforms("a5")
    @pytest.mark.parametrize("platform", ["a5"])
    @pytest.mark.parametrize("dtype", [DataType.INT8, DataType.UINT8, DataType.BF16])
    def test_a5_only_dtypes(self, test_runner, platform, dtype):
        result = test_runner.run(TriTestCase(dtype=dtype, platform=platform))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("platform", ONBOARD_PLATFORMS)
    @pytest.mark.parametrize(
        ("upper", "diagonal"),
        [
            (False, 0),
            (True, 0),
            (False, 2),
            (True, -1),
            (False, -M),
            (True, N + 1),
            (False, N),
            (True, -M),
        ],
        ids=(
            "lower-main",
            "upper-main",
            "lower-plus2",
            "upper-minus1",
            "lower-empty",
            "upper-empty",
            "lower-full",
            "upper-full",
        ),
    )
    def test_side_and_diagonal(self, test_runner, platform, upper, diagonal):
        result = test_runner.run(TriTestCase(upper=upper, diagonal=diagonal, platform=platform))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("platform", ONBOARD_PLATFORMS)
    @pytest.mark.parametrize(
        "valid_shape",
        [(9, N), (M, 21), (9, 21), (1, 1)],
        ids=("row-tail", "col-tail", "row-col-tail", "scalar-boundary"),
    )
    def test_valid_shape(self, test_runner, platform, valid_shape):
        result = test_runner.run(TriTestCase(valid_shape=valid_shape, platform=platform))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("platform", ONBOARD_PLATFORMS)
    @pytest.mark.parametrize("shape", [(1, 32), (32, 8)], ids=("single-row", "tall-narrow"))
    def test_physical_shape_boundaries(self, test_runner, platform, shape):
        result = test_runner.run(TriTestCase(shape=shape, platform=platform))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("platform", ONBOARD_PLATFORMS)
    def test_runtime_scalar_diagonal(self, test_runner, platform):
        result = test_runner.run(TriScalarDiagonalTestCase(platform=platform))
        assert result.passed, f"Test failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
