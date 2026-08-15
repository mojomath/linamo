"""
Tests for StaticMatrix creation and element access.
"""

import std.testing as testing
import linamo as la


def test_smatrix_from_nested_list() raises:
    """Test creating a static matrix from nested lists."""
    var mat = la.smatrix[2, 3, DType.float64](
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ]
    )
    testing.assert_equal(mat.get_nrows(), 2)
    testing.assert_equal(mat.get_ncols(), 3)
    testing.assert_equal(mat[0, 0], 1.0)
    testing.assert_equal(mat[0, 2], 3.0)
    testing.assert_equal(mat[1, 0], 4.0)
    testing.assert_equal(mat[1, 2], 6.0)


def test_smatrix_from_flat_list() raises:
    """Test creating a static matrix from a flat list."""
    var mat = la.smatrix[2, 3, DType.float64](
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

    var mat = StaticMatrix[DType.float64, 3, 3]()
    for i in range(3):
        for j in range(3):
            testing.assert_equal(mat[i, j], 0.0)


def test_smatrix_wrong_rows_raises() raises:
    """Test that wrong number of rows raises ValueError."""
    var raised = False
    try:
        var _mat = la.smatrix[3, 2, DType.float64](
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
        var _mat = la.smatrix[2, 3, DType.float64](
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
        var _mat = la.smatrix[2, 3, DType.float64](
            flat_list=[1.0, 2.0, 3.0, 4.0]
        )
    except:
        raised = True
    testing.assert_true(
        raised, "Flat list size mismatch should raise ValueError"
    )


def test_smatrix_get_size() raises:
    """Test the get_size method for static matrices."""
    var mat = la.smatrix[4, 5, DType.float64](
        [
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [6.0, 7.0, 8.0, 9.0, 10.0],
            [11.0, 12.0, 13.0, 14.0, 15.0],
            [16.0, 17.0, 18.0, 19.0, 20.0],
        ]
    )
    testing.assert_equal(mat.get_size(), 20)


def test_smatrix_integer_type() raises:
    """Test creating a static matrix with integer dtype."""
    var mat = la.smatrix[2, 2, DType.int64]([[10, 20], [30, 40]])
    testing.assert_equal(mat[0, 0], Int64(10))
    testing.assert_equal(mat[1, 1], Int64(40))


def test_smatrix_str() raises:
    """Test StaticMatrix string representation."""
    var mat = la.smatrix[2, 2, DType.float64]([[1.0, 2.0], [3.0, 4.0]])
    var s = String(mat)
    testing.assert_true("1.0" in s, "String should contain 1.0")
    testing.assert_true("4.0" in s, "String should contain 4.0")


def test_smatrix_copy() raises:
    """Test that copying a static matrix creates an independent copy."""
    var a = la.smatrix[2, 2, DType.float64]([[1.0, 2.0], [3.0, 4.0]])
    var b = a.copy()
    testing.assert_equal(b[0, 0], 1.0)
    testing.assert_equal(b[1, 1], 4.0)


def main() raises:
    testing.TestSuite.discover_tests[__functions_in_module()]().run()
