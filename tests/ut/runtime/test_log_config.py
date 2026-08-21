# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for PyPTO runtime log-level synchronization."""

from unittest.mock import patch

import pytest
from pypto import LogLevel
from pypto.runtime.log_config import _sync_to_pypto


@pytest.mark.parametrize(
    ("threshold", "expected"),
    [
        (10, LogLevel.DEBUG),
        (11, LogLevel.INFO),
        (20, LogLevel.INFO),
        (21, LogLevel.WARN),
        (25, LogLevel.WARN),
        (30, LogLevel.WARN),
        (31, LogLevel.ERROR),
        (40, LogLevel.ERROR),
        (41, LogLevel.NONE),
        (60, LogLevel.NONE),
    ],
)
def test_sync_to_pypto_matches_simpler_severity_thresholds(threshold, expected):
    with patch("pypto.pypto_core.set_log_level") as set_log_level:
        _sync_to_pypto(threshold)

    set_log_level.assert_called_once_with(expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
