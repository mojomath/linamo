"""
Tests for MatrixView creation and element access.
"""

import std.testing as testing
import linamo as la


def test_view_from_matrix_full() raises:
    """Test creating a full view from a matrix."""
    var mat = la.matrix[Float64](
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ]
    )
    var v = mat.view()
    testing.assert_equal(v.nrows(), 3)
    testing.assert_equal(v.ncols(), 3)
    testing.assert_equal(v[0, 0], 1.0)
    testing.assert_equal(v[1, 1], 5.0)
    testing.assert_equal(v[2, 2], 9.0)


def test_view_slice_rows_and_cols() raises:
    """Test creating a slice view with row and column ranges."""
    var mat = la.matrix[Float64](
        [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
        ]
    )
    # Slice rows 0:2, cols 1:3 -> 2x2 submatrix
    var v = mat[0:2, 1:3]
    testing.assert_equal(v.nrows(), 2)
    testing.assert_equal(v.ncols(), 2)
    testing.assert_equal(v[0, 0], 2.0)
    testing.assert_equal(v[0, 1], 3.0)
    testing.assert_equal(v[1, 0], 6.0)
    testing.assert_equal(v[1, 1], 7.0)


def test_view_slice_with_step() raises:
    """Test creating a slice view with step > 1."""
    var mat = la.matrix[Float64](
        [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0],
        ]
    )
    # Every other row and column: rows 0:4:2, cols 0:4:2
    var v = mat[0:4:2, 0:4:2]
    testing.assert_equal(v.nrows(), 2)
    testing.assert_equal(v.ncols(), 2)
    testing.assert_equal(v[0, 0], 1.0)
    testing.assert_equal(v[0, 1], 3.0)
    testing.assert_equal(v[1, 0], 9.0)
    testing.assert_equal(v[1, 1], 11.0)


def test_view_shares_data() raises:
    """Test that a view shares data with the original matrix."""
    var mat = la.matrix[Float64]([[1.0, 2.0], [3.0, 4.0]])
    var v = mat.view()
    # View should reflect original data
    testing.assert_equal(v[0, 0], 1.0)
    # Mutate original
    mat[0, 0] = 99.0
    # View should see the change (shared data)
    testing.assert_equal(v[0, 0], 99.0)


def test_view_on_view() raises:
    """Test creating a view from another view."""
    var mat = la.matrix[Float64](
        [
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [6.0, 7.0, 8.0, 9.0, 10.0],
            [11.0, 12.0, 13.0, 14.0, 15.0],
            [16.0, 17.0, 18.0, 19.0, 20.0],
        ]
    )
    var v1 = mat[0:4, 1:4]  # 4x3 view
    testing.assert_equal(v1.nrows(), 4)
    testing.assert_equal(v1.ncols(), 3)
    testing.assert_equal(v1[0, 0], 2.0)

    var v2 = v1[1:3, 0:2]  # 2x2 view from v1
    testing.assert_equal(v2.nrows(), 2)
    testing.assert_equal(v2.ncols(), 2)
    testing.assert_equal(v2[0, 0], 7.0)
    testing.assert_equal(v2[0, 1], 8.0)
    testing.assert_equal(v2[1, 0], 12.0)
    testing.assert_equal(v2[1, 1], 13.0)


def test_view_str() raises:
    """Test MatrixView string representation."""
    var mat = la.matrix[Float64]([[1.0, 2.0], [3.0, 4.0]])
    var v = mat.view()
    var s = String(v)
    testing.assert_true("1.0" in s, "View str should contain 1.0")
    testing.assert_true("4.0" in s, "View str should contain 4.0")


def test_view_write_to_metadata() raises:
    """Test that MatrixView write_to includes metadata."""
    var mat = la.matrix[Float64]([[1.0, 2.0], [3.0, 4.0]])
    var v = mat.view()
    var s = String("")
    v.write_to(s)
    testing.assert_true(
        "MatrixView" in s, "write_to should include 'MatrixView'"
    )
    testing.assert_true("float64" in s, "write_to should include dtype")


def test_view_unsafe_get() raises:
    """Test unsafe element access on a view."""
    var mat = la.matrix[Float64](
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ]
    )
    var v = mat[0:2, 1:3]
    testing.assert_equal(v.unsafe_get(0, 0), 2.0)
    testing.assert_equal(v.unsafe_get(0, 1), 3.0)
    testing.assert_equal(v.unsafe_get(1, 0), 5.0)
    testing.assert_equal(v.unsafe_get(1, 1), 6.0)


def main() raises:
    testing.TestSuite.discover_tests[__functions_in_module()]().run()
