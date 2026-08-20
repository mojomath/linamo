"""
Tests for StaticMatrix creation and element access.
"""

import std.testing as testing
import linamo as la


def test_smatrix_from_nested_list() raises:
    """Test creating a static matrix from nested lists."""
    var mat = la.smatrix[2, 3, Float64](
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ]
    )
    testing.assert_equal(mat.nrows(), 2)
    testing.assert_equal(mat.ncols(), 3)
    testing.assert_equal(mat[0, 0], 1.0)
    testing.assert_equal(mat[0, 2], 3.0)
    testing.assert_equal(mat[1, 0], 4.0)
    testing.assert_equal(mat[1, 2], 6.0)


def test_smatrix_from_flat_list() raises:
    """Test creating a static matrix from a flat list."""
    var mat = la.smatrix[2, 3, Float64](
        flat_list=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    )
    testing.assert_equal(mat[0, 0], 1.0)
    testing.assert_equal(mat[0, 1], 2.0)
    testing.assert_equal(mat[0, 2], 3.0)
    testing.assert_equal(mat[1, 0], 4.0)
    testing.assert_equal(mat[1, 1], 5.0)
    testing.assert_equal(mat[1, 2], 6.0)


def test_smatrix_default_zeros() raises:
    """Test that default-constructed static matrix is all zeros."""
    from linamo.types.static_matrix import StaticMatrix

    var mat = StaticMatrix[Float64, 3, 3]()
    for i in range(3):
        for j in range(3):
            testing.assert_equal(mat[i, j], 0.0)


def test_smatrix_wrong_rows_raises() raises:
    """Test that wrong number of rows raises ValueError."""
    var raised = False
    try:
        var _mat = la.smatrix[3, 2, Float64](
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ]
        )
    except:
        raised = True
    testing.assert_true(raised, "Wrong number of rows should raise ValueError")


def test_smatrix_wrong_cols_raises() raises:
    """Test that wrong number of columns raises ValueError."""
    var raised = False
    try:
        var _mat = la.smatrix[2, 3, Float64](
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ]
        )
    except:
        raised = True
    testing.assert_true(raised, "Wrong number of cols should raise ValueError")


def test_smatrix_flat_list_size_mismatch_raises() raises:
    """Test that flat list size mismatch raises ValueError."""
    var raised = False
    try:
        var _mat = la.smatrix[2, 3, Float64](flat_list=[1.0, 2.0, 3.0, 4.0])
    except:
        raised = True
    testing.assert_true(
        raised, "Flat list size mismatch should raise ValueError"
    )


def test_smatrix_size() raises:
    """Test the size method for static matrices."""
    var mat = la.smatrix[4, 5, Float64](
        [
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [6.0, 7.0, 8.0, 9.0, 10.0],
            [11.0, 12.0, 13.0, 14.0, 15.0],
            [16.0, 17.0, 18.0, 19.0, 20.0],
        ]
    )
    testing.assert_equal(mat.size(), 20)


def test_smatrix_integer_type() raises:
    """Test creating a static matrix with integer dtype."""
    var mat = la.smatrix[2, 2, Int64]([[10, 20], [30, 40]])
    testing.assert_equal(mat[0, 0], Int64(10))
    testing.assert_equal(mat[1, 1], Int64(40))


def test_smatrix_str() raises:
    """Test StaticMatrix string representation."""
    var mat = la.smatrix[2, 2, Float64]([[1.0, 2.0], [3.0, 4.0]])
    var s = String(mat)
    testing.assert_true("1.0" in s, "String should contain 1.0")
    testing.assert_true("4.0" in s, "String should contain 4.0")


def test_smatrix_copy() raises:
    """Test that copying a static matrix creates an independent copy."""
    var a = la.smatrix[2, 2, Float64]([[1.0, 2.0], [3.0, 4.0]])
    var b = a.copy()
    testing.assert_equal(b[0, 0], 1.0)
    testing.assert_equal(b[1, 1], 4.0)


# ===----------------------------------------------------------------------===#
# to_matrix
# ===----------------------------------------------------------------------===#
# `to_matrix()` is the only bridge between `StaticMatrix` and the rest of the
# library, so these cover the two things that make it more than a field copy:
# the power-of-two row padding must not survive into the result, and the copy
# must be independent of its source.


def test_to_matrix_basic() raises:
    """Test converting a static matrix into a dynamic one."""
    var s = la.smatrix[2, 3, Float64]([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    var m = s.to_matrix()
    testing.assert_equal(m.nrows(), 2)
    testing.assert_equal(m.ncols(), 3)
    for i in range(2):
        for j in range(3):
            testing.assert_equal(m[i, j], s[i, j])


def test_to_matrix_strips_padding() raises:
    """Test that the result is dense, not padded like the source."""
    # A 3x3 StaticMatrix pads its rows to 4, so its row stride is 4.
    var s = la.smatrix[3, 3, Float64](
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
    )
    testing.assert_equal(s.row_stride(), 4)
    var m = s.to_matrix()
    testing.assert_equal(m.row_stride(), 3)
    testing.assert_equal(m.col_stride(), 1)
    testing.assert_true(m.is_c_contiguous())
    testing.assert_equal(len(m.data()), 9)


def test_to_matrix_is_a_copy() raises:
    """Test that the result does not alias the static matrix."""
    var s = la.smatrix[2, 2, Float64]([[1.0, 2.0], [3.0, 4.0]])
    var m = s.to_matrix()
    m[0, 0] = 99.0
    testing.assert_equal(s[0, 0], 1.0)
    testing.assert_equal(m[0, 0], 99.0)


def test_to_matrix_int_dtype() raises:
    """Test that the conversion carries a non-default dtype."""
    var s = la.smatrix[2, 2, Int64]([[1, 2], [3, 4]])
    var m = s.to_matrix()
    testing.assert_equal(m[1, 1], Int64(4))
    testing.assert_equal(m.nrows(), 2)


def test_to_matrix_single_element() raises:
    """Test converting a 1x1 static matrix."""
    var s = la.smatrix[1, 1, Float64]([[7.0]])
    var m = s.to_matrix()
    testing.assert_equal(m.nrows(), 1)
    testing.assert_equal(m.ncols(), 1)
    testing.assert_equal(m[0, 0], 7.0)


def test_to_matrix_non_square_padding() raises:
    """Test a shape whose row padding differs from its column padding."""
    # 3x5 pads to a 4x8 buffer, so the source row stride is 8.
    var s = la.smatrix[3, 5, Float64](
        [
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [6.0, 7.0, 8.0, 9.0, 10.0],
            [11.0, 12.0, 13.0, 14.0, 15.0],
        ]
    )
    testing.assert_equal(s.row_stride(), 8)
    var m = s.to_matrix()
    testing.assert_equal(m.row_stride(), 5)
    testing.assert_equal(len(m.data()), 15)
    for i in range(3):
        for j in range(5):
            testing.assert_equal(m[i, j], s[i, j])


def test_to_matrix_reaches_operators_and_routines() raises:
    """Test that the converted matrix reaches the rest of the library."""
    var s = la.smatrix[2, 2, Float64]([[1.0, 2.0], [3.0, 4.0]])
    var d = la.matrix[Float64]([[10.0, 10.0], [10.0, 10.0]])
    var total = d + s.to_matrix()
    testing.assert_equal(total[0, 0], 11.0)
    testing.assert_equal(total[1, 1], 14.0)
    testing.assert_equal(la.sum(s.to_matrix()), 10.0)
    var t = la.transpose(s.to_matrix())
    testing.assert_equal(t[0, 1], 3.0)
    testing.assert_equal(t[1, 0], 2.0)


def main() raises:
    testing.TestSuite.discover_tests[__functions_in_module()]().run()
