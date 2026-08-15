"""
Tests for numpy interoperability API.

Tests matrix_from_numpy() and to_numpy() — the core conversion functions.
For numpy-as-ground-truth tests of math/linalg/creation, see the
corresponding test_*_numpy.mojo files.
"""

import std.testing as testing
from linamo.routines.numpy_interop import matrix_from_numpy, to_numpy
from std.python import Python


# ===----------------------------------------------------------------------===#
# matrix_from_numpy tests
# ===----------------------------------------------------------------------===#


def test_matrix_from_numpy_basic() raises:
    """Test basic creation of a Matrix from a numpy 2D array."""
    var np = Python.import_module("numpy")
    var np_arr = np.arange(1.0, 7.0).reshape(2, 3)
    var mat = matrix_from_numpy(np_arr)
    testing.assert_equal(mat.nrows, 2)
    testing.assert_equal(mat.ncols, 3)
    testing.assert_equal(mat[0, 0], 1.0)
    testing.assert_equal(mat[0, 1], 2.0)
    testing.assert_equal(mat[0, 2], 3.0)
    testing.assert_equal(mat[1, 0], 4.0)
    testing.assert_equal(mat[1, 1], 5.0)
    testing.assert_equal(mat[1, 2], 6.0)


def test_matrix_from_numpy_f_contiguous() raises:
    """Test that F-contiguous numpy arrays are handled correctly."""
    var np = Python.import_module("numpy")
    var np_arr = np.asfortranarray(np.arange(1.0, 5.0).reshape(2, 2))
    var mat = matrix_from_numpy(np_arr)
    testing.assert_equal(mat[0, 0], 1.0)
    testing.assert_equal(mat[0, 1], 2.0)
    testing.assert_equal(mat[1, 0], 3.0)
    testing.assert_equal(mat[1, 1], 4.0)
    # Result should always be C-contiguous
    testing.assert_true(mat.is_c_contiguous())


def test_matrix_from_numpy_1d_raises() raises:
    """Test that a 1D numpy array raises an error."""
    var np = Python.import_module("numpy")
    var np_1d = np.arange(3.0)
    var got_error = False
    try:
        var _ = matrix_from_numpy(np_1d)
    except:
        got_error = True
    testing.assert_true(got_error, msg="Should raise for 1D array")


def test_matrix_from_numpy_3d_raises() raises:
    """Test that a 3D numpy array raises an error."""
    var np = Python.import_module("numpy")
    var np_3d = np.arange(24.0).reshape(2, 3, 4)
    var got_error = False
    try:
        var _ = matrix_from_numpy(np_3d)
    except:
        got_error = True
    testing.assert_true(got_error, msg="Should raise for 3D array")


def test_matrix_from_numpy_single_element() raises:
    """Test creation from a 1x1 numpy array."""
    var np = Python.import_module("numpy")
    var np_arr = np.arange(42.0, 43.0).reshape(1, 1)
    var mat = matrix_from_numpy(np_arr)
    testing.assert_equal(mat.nrows, 1)
    testing.assert_equal(mat.ncols, 1)
    testing.assert_equal(mat[0, 0], 42.0)


# ===----------------------------------------------------------------------===#
# to_numpy round-trip tests
# ===----------------------------------------------------------------------===#


def test_to_numpy_round_trip() raises:
    """Test numpy -> matrix -> numpy round-trip preserves data."""
    var np = Python.import_module("numpy")
    var np_arr = np.arange(1.0, 13.0).reshape(3, 4)
    var mat = matrix_from_numpy(np_arr)
    var np_back = to_numpy(mat)
    testing.assert_true(
        Bool(np.array_equal(np_arr, np_back)),
        msg="Round-trip should preserve data",
    )


def test_to_numpy_round_trip_random() raises:
    """Test round-trip with random data."""
    var np = Python.import_module("numpy")
    var np_arr = np.random.rand(5, 7)
    var mat = matrix_from_numpy(np_arr)
    var np_back = to_numpy(mat)
    testing.assert_true(
        Bool(np.allclose(np_arr, np_back)),
        msg="Round-trip should preserve random data",
    )


# ===----------------------------------------------------------------------===#
# Test runner
# ===----------------------------------------------------------------------===#


def main() raises:
    testing.TestSuite.discover_tests[__functions_in_module()]().run()
