"""
Tests for convenience creation routines: zeros, ones, full, eye, identity, diag.
"""

import std.testing as testing
import linamo as la
from linamo.routines.creation import zeros, ones, full, eye, identity, diag


# ===----------------------------------------------------------------------===#
# zeros
# ===----------------------------------------------------------------------===#


def test_zeros_basic() raises:
    """Test creating a matrix of zeros."""
    var mat = zeros[DType.float64](3, 4)
    testing.assert_equal(mat.nrows, 3)
    testing.assert_equal(mat.ncols, 4)
    for i in range(3):
        for j in range(4):
            testing.assert_equal(mat[i, j], 0.0)


def test_zeros_single() raises:
    """Test creating a 1x1 matrix of zeros."""
    var mat = zeros[DType.float64](1, 1)
    testing.assert_equal(mat[0, 0], 0.0)


def test_zeros_int_dtype() raises:
    """Test zeros with integer dtype."""
    var mat = zeros[DType.int64](2, 3)
    testing.assert_equal(mat[0, 0], Int64(0))
    testing.assert_equal(mat[1, 2], Int64(0))


# ===----------------------------------------------------------------------===#
# ones
# ===----------------------------------------------------------------------===#


def test_ones_basic() raises:
    """Test creating a matrix of ones."""
    var mat = ones[DType.float64](2, 3)
    testing.assert_equal(mat.nrows, 2)
    testing.assert_equal(mat.ncols, 3)
    for i in range(2):
        for j in range(3):
            testing.assert_equal(mat[i, j], 1.0)


def test_ones_int_dtype() raises:
    """Test ones with integer dtype."""
    var mat = ones[DType.int32](3, 2)
    testing.assert_equal(mat[0, 0], Int32(1))
    testing.assert_equal(mat[2, 1], Int32(1))


# ===----------------------------------------------------------------------===#
# full
# ===----------------------------------------------------------------------===#


def test_full_basic() raises:
    """Test creating a matrix filled with a value."""
    var mat = full[DType.float64](2, 3, 7.5)
    for i in range(2):
        for j in range(3):
            testing.assert_equal(mat[i, j], 7.5)


def test_full_negative() raises:
    """Test full with a negative value."""
    var mat = full[DType.float64](3, 3, -1.0)
    testing.assert_equal(mat[0, 0], -1.0)
    testing.assert_equal(mat[2, 2], -1.0)


# ===----------------------------------------------------------------------===#
# eye / identity
# ===----------------------------------------------------------------------===#


def test_eye_basic() raises:
    """Test creating an identity matrix with eye."""
    var mat = eye[DType.float64](3)
    testing.assert_equal(mat.nrows, 3)
    testing.assert_equal(mat.ncols, 3)
    testing.assert_equal(mat[0, 0], 1.0)
    testing.assert_equal(mat[1, 1], 1.0)
    testing.assert_equal(mat[2, 2], 1.0)
    testing.assert_equal(mat[0, 1], 0.0)
    testing.assert_equal(mat[1, 0], 0.0)
    testing.assert_equal(mat[0, 2], 0.0)


def test_eye_1x1() raises:
    """Test eye for 1x1 matrix."""
    var mat = eye[DType.float64](1)
    testing.assert_equal(mat[0, 0], 1.0)


def test_identity_equals_eye() raises:
    """Test that identity() is an alias for eye()."""
    var e = eye[DType.float64](4)
    var id = identity[DType.float64](4)
    for i in range(4):
        for j in range(4):
            testing.assert_equal(e[i, j], id[i, j])


def test_eye_int_dtype() raises:
    """Test eye with integer dtype."""
    var mat = eye[DType.int64](3)
    testing.assert_equal(mat[0, 0], Int64(1))
    testing.assert_equal(mat[0, 1], Int64(0))


# ===----------------------------------------------------------------------===#
# diag (construct)
# ===----------------------------------------------------------------------===#


def test_diag_construct() raises:
    """Test constructing a diagonal matrix from a list."""
    var vals: List[Float64] = [1.0, 2.0, 3.0]
    var mat = diag(vals^)
    testing.assert_equal(mat.nrows, 3)
    testing.assert_equal(mat.ncols, 3)
    testing.assert_equal(mat[0, 0], 1.0)
    testing.assert_equal(mat[1, 1], 2.0)
    testing.assert_equal(mat[2, 2], 3.0)
    testing.assert_equal(mat[0, 1], 0.0)
    testing.assert_equal(mat[1, 0], 0.0)
    testing.assert_equal(mat[0, 2], 0.0)


def test_diag_single() raises:
    """Test constructing a 1x1 diagonal matrix."""
    var vals: List[Float64] = [42.0]
    var mat = diag(vals^)
    testing.assert_equal(mat[0, 0], 42.0)


# ===----------------------------------------------------------------------===#
# diag (extract)
# ===----------------------------------------------------------------------===#


def test_diag_extract() raises:
    """Test extracting diagonal from a matrix."""
    var mat = la.matrix[DType.float64](
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
    )
    var d = diag(mat)
    testing.assert_equal(len(d), 3)
    testing.assert_equal(d[0], 1.0)
    testing.assert_equal(d[1], 5.0)
    testing.assert_equal(d[2], 9.0)


def test_diag_extract_nonsquare_raises() raises:
    """Test that extracting diagonal from non-square raises ValueError."""
    var mat = la.matrix[DType.float64]([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    var raised = False
    try:
        var _d = diag(mat)
    except:
        raised = True
    testing.assert_true(raised, "Non-square matrix should raise ValueError")


def main() raises:
    testing.TestSuite.discover_tests[__functions_in_module()]().run()
