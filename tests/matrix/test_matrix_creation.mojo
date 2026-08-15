"""
Tests for Matrix creation routines.
"""

import std.testing as testing
import matmojo as mm


def test_matrix_from_nested_list_row_major() raises:
    """Test creating a matrix from nested lists in row-major order."""
    var mat = mm.matrix[DType.float64](
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ],
        order="C",
    )
    testing.assert_equal(mat.nrows, 2)
    testing.assert_equal(mat.ncols, 3)
    testing.assert_equal(mat.row_stride, 3)
    testing.assert_equal(mat.col_stride, 1)
    testing.assert_equal(mat[0, 0], 1.0)
    testing.assert_equal(mat[0, 2], 3.0)
    testing.assert_equal(mat[1, 0], 4.0)
    testing.assert_equal(mat[1, 2], 6.0)


def test_matrix_from_nested_list_col_major() raises:
    """Test creating a matrix from nested lists in column-major order."""
    var mat = mm.matrix[DType.float64](
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ],
        order="F",
    )
    testing.assert_equal(mat.nrows, 2)
    testing.assert_equal(mat.ncols, 3)
    testing.assert_equal(mat.row_stride, 1)
    testing.assert_equal(mat.col_stride, 2)
    testing.assert_equal(mat[0, 0], 1.0)
    testing.assert_equal(mat[0, 2], 3.0)
    testing.assert_equal(mat[1, 0], 4.0)
    testing.assert_equal(mat[1, 2], 6.0)


def test_matrix_from_flat_list_row_major() raises:
    """Test creating a matrix from a flat list with shape in row-major order."""
    var mat = mm.matrix[DType.int64](
        flat_list=[1, 2, 3, 4, 5, 6],
        nrows=2,
        ncols=3,
        order="C",
    )
    testing.assert_equal(mat.nrows, 2)
    testing.assert_equal(mat.ncols, 3)
    testing.assert_equal(mat[0, 0], 1)
    testing.assert_equal(mat[0, 1], 2)
    testing.assert_equal(mat[0, 2], 3)
    testing.assert_equal(mat[1, 0], 4)
    testing.assert_equal(mat[1, 1], 5)
    testing.assert_equal(mat[1, 2], 6)


def test_matrix_from_flat_list_col_major() raises:
    """Test creating a matrix from a flat list with shape in column-major order.
    """
    var mat = mm.matrix[DType.float64](
        flat_list=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        nrows=2,
        ncols=3,
        order="F",
    )
    testing.assert_equal(mat.nrows, 2)
    testing.assert_equal(mat.ncols, 3)
    # In column-major order, flat_list is stored column-by-column
    # So element (0,0)=1, (1,0)=2, (0,1)=3, (1,1)=4, (0,2)=5, (1,2)=6
    testing.assert_equal(mat[0, 0], 1.0)
    testing.assert_equal(mat[1, 0], 2.0)
    testing.assert_equal(mat[0, 1], 3.0)
    testing.assert_equal(mat[1, 1], 4.0)
    testing.assert_equal(mat[0, 2], 5.0)
    testing.assert_equal(mat[1, 2], 6.0)


def test_matrix_creation_empty_raises() raises:
    """Test that creating a matrix from an empty list raises ValueError."""
    var raised = False
    try:
        var _mat = mm.matrix[DType.float64](
            List[List[Float64]](),
        )
    except:
        raised = True
    testing.assert_true(raised, "Empty list should raise ValueError")


def test_matrix_creation_mismatched_rows_raises() raises:
    """Test that rows of different lengths raise ValueError."""
    var raised = False
    try:
        var _mat = mm.matrix[DType.float64](
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0],
            ],
        )
    except:
        raised = True
    testing.assert_true(
        raised, "Mismatched row lengths should raise ValueError"
    )


def test_matrix_creation_flat_list_size_mismatch_raises() raises:
    """Test that flat list size not matching shape raises ValueError."""
    var raised = False
    try:
        var _mat = mm.matrix[DType.float64](
            flat_list=[1.0, 2.0, 3.0, 4.0],
            nrows=2,
            ncols=3,
        )
    except:
        raised = True
    testing.assert_true(raised, "Size mismatch should raise ValueError")


def test_matrix_creation_invalid_order_raises() raises:
    """Test that an invalid order string raises ValueError."""
    var raised = False
    try:
        var _mat = mm.matrix[DType.float64](
            [[1.0, 2.0], [3.0, 4.0]],
            order="X",
        )
    except:
        raised = True
    testing.assert_true(raised, "Invalid order should raise ValueError")


def test_matrix_integer_types() raises:
    """Test creating matrices with various integer dtypes."""
    var mat_i32 = mm.matrix[DType.int32](
        [[1, 2], [3, 4]],
    )
    testing.assert_equal(mat_i32[0, 0], Int32(1))
    testing.assert_equal(mat_i32[1, 1], Int32(4))

    var mat_i64 = mm.matrix[DType.int64](
        [[10, 20], [30, 40]],
    )
    testing.assert_equal(mat_i64[0, 0], Int64(10))
    testing.assert_equal(mat_i64[1, 1], Int64(40))


def test_matrix_single_element() raises:
    """Test creating a 1x1 matrix."""
    var mat = mm.matrix[DType.float64]([[42.0]])
    testing.assert_equal(mat.nrows, 1)
    testing.assert_equal(mat.ncols, 1)
    testing.assert_equal(mat[0, 0], 42.0)


def test_matrix_get_size() raises:
    """Test the get_size method."""
    var mat = mm.matrix[DType.float64](
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
    )
    testing.assert_equal(mat.get_size(), 6)


def main() raises:
    testing.TestSuite.discover_tests[__functions_in_module()]().run()
