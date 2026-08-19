"""
Tests for Matrix element access and indexing.
"""

import std.testing as testing
import linamo as la


def test_matrix_getitem_basic() raises:
    """Test basic element access with __getitem__."""
    var mat = la.matrix[DType.float64](
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ]
    )
    testing.assert_equal(mat[0, 0], 1.0)
    testing.assert_equal(mat[0, 2], 3.0)
    testing.assert_equal(mat[1, 1], 5.0)
    testing.assert_equal(mat[2, 0], 7.0)
    testing.assert_equal(mat[2, 2], 9.0)


def test_matrix_getitem_col_major() raises:
    """Test element access on column-major matrix."""
    var mat = la.matrix[DType.float64](
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ],
        order="F",
    )
    testing.assert_equal(mat[0, 0], 1.0)
    testing.assert_equal(mat[0, 1], 2.0)
    testing.assert_equal(mat[1, 0], 3.0)
    testing.assert_equal(mat[1, 1], 4.0)
    testing.assert_equal(mat[2, 0], 5.0)
    testing.assert_equal(mat[2, 1], 6.0)


def test_matrix_getitem_out_of_bounds_raises() raises:
    """Test that out-of-bounds access raises IndexError."""
    var mat = la.matrix[DType.float64]([[1.0, 2.0], [3.0, 4.0]])
    var raised = False

    # Row out of bounds
    try:
        var _val = mat[2, 0]
    except:
        raised = True
    testing.assert_true(raised, "Row out of bounds should raise IndexError")

    # Column out of bounds
    raised = False
    try:
        var _val = mat[0, 2]
    except:
        raised = True
    testing.assert_true(raised, "Column out of bounds should raise IndexError")

    # Negative index
    raised = False
    try:
        var _val = mat[-1, 0]
    except:
        raised = True
    testing.assert_true(raised, "Negative row index should raise IndexError")


def test_matrix_setitem_via_ref() raises:
    """Test element mutation via __getitem__ returning a mutable ref."""
    var mat = la.matrix[DType.float64]([[1.0, 2.0], [3.0, 4.0]])
    mat[0, 0] = 99.0
    mat[1, 1] = 42.0
    testing.assert_equal(mat[0, 0], 99.0)
    testing.assert_equal(mat[1, 1], 42.0)
    # Other elements unchanged
    testing.assert_equal(mat[0, 1], 2.0)
    testing.assert_equal(mat[1, 0], 3.0)


def test_matrix_get_unsafe() raises:
    """Test unsafe element access (no bounds checking)."""
    var mat = la.matrix[DType.float64](
        [
            [10.0, 20.0, 30.0],
            [40.0, 50.0, 60.0],
        ]
    )
    testing.assert_equal(mat.get_unsafe(0, 0), 10.0)
    testing.assert_equal(mat.get_unsafe(0, 2), 30.0)
    testing.assert_equal(mat.get_unsafe(1, 1), 50.0)
    testing.assert_equal(mat.get_unsafe(1, 2), 60.0)


def test_matrix_attributes() raises:
    """Test that matrix attribute getters work correctly."""
    var mat = la.matrix[DType.float64](
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ]
    )
    testing.assert_equal(mat.nrows(), 2)
    testing.assert_equal(mat.ncols(), 3)
    testing.assert_equal(mat.row_stride(), 3)
    testing.assert_equal(mat.col_stride(), 1)
    testing.assert_equal(mat.offset(), 0)
    testing.assert_equal(mat.size(), 6)


def main() raises:
    testing.TestSuite.discover_tests[__functions_in_module()]().run()
