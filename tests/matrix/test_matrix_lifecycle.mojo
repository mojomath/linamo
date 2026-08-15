"""
Tests for Matrix copy and move semantics.
"""

import std.testing as testing
import linamo as mm


def test_matrix_copy() raises:
    """Test that copying a matrix creates an independent copy."""
    var a = mm.matrix[DType.float64]([[1.0, 2.0], [3.0, 4.0]])
    var b = a.copy()
    # Verify values match
    testing.assert_equal(b[0, 0], 1.0)
    testing.assert_equal(b[1, 1], 4.0)
    # Mutating b should not affect a
    b[0, 0] = 99.0
    testing.assert_equal(b[0, 0], 99.0)
    testing.assert_equal(a[0, 0], 1.0)


def test_matrix_copy_preserves_layout() raises:
    """Test that copy preserves memory layout (strides)."""
    var a = mm.matrix[DType.float64](
        [[1.0, 2.0], [3.0, 4.0]],
        order="F",
    )
    var b = a.copy()
    testing.assert_equal(b.row_stride, a.row_stride)
    testing.assert_equal(b.col_stride, a.col_stride)
    testing.assert_equal(b.nrows, a.nrows)
    testing.assert_equal(b.ncols, a.ncols)


def main() raises:
    testing.TestSuite.discover_tests[__functions_in_module()]().run()
