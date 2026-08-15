"""
Numpy-powered integration tests for matrix creation routines.

Verifies round-trip fidelity: numpy -> matmojo -> numpy for
zeros, ones, full, eye, diag, and random matrix creation.
"""

import std.testing as testing
from matmojo.routines.creation import zeros, ones, full, eye, diag
from matmojo.routines.numpy_interop import matrix_from_numpy, to_numpy
from matmojo.utils.test_utils import (
    assert_matrices_equal,
    assert_matrices_close,
)
from std.python import Python, PythonObject


# ===----------------------------------------------------------------------===#
# Helpers
# ===----------------------------------------------------------------------===#


def _test_round_trip(np: PythonObject, r: Int, c: Int) raises:
    """Helper: round-trip numpy -> Matrix -> numpy for one shape."""
    var a_np = np.random.rand(r, c)
    var mat = matrix_from_numpy(a_np)
    var back = to_numpy(mat)
    testing.assert_true(
        Bool(np.allclose(a_np, back)),
        msg="round-trip failed for " + String(r) + "x" + String(c),
    )


def _test_zeros(np: PythonObject, r: Int, c: Int) raises:
    """Helper: zeros(r, c) matches np.zeros."""
    var mat = zeros[DType.float64](r, c)
    var back = to_numpy(mat)
    var expected = np.zeros(r * c).reshape(r, c)
    testing.assert_true(
        Bool(np.array_equal(back, expected)),
        msg="zeros mismatch for " + String(r) + "x" + String(c),
    )


def _test_ones(np: PythonObject, r: Int, c: Int) raises:
    """Helper: ones(r, c) matches np.ones."""
    var mat = ones[DType.float64](r, c)
    var back = to_numpy(mat)
    var expected = np.ones(r * c).reshape(r, c)
    testing.assert_true(
        Bool(np.array_equal(back, expected)),
        msg="ones mismatch for " + String(r) + "x" + String(c),
    )


def _test_eye(np: PythonObject, n: Int) raises:
    """Helper: eye(n) matches np.eye(n)."""
    var mat = eye[DType.float64](n)
    var back = to_numpy(mat)
    var expected = np.eye(n)
    testing.assert_true(
        Bool(np.array_equal(back, expected)),
        msg="eye mismatch for n=" + String(n),
    )


# ===----------------------------------------------------------------------===#
# Round-trip: numpy -> Matrix -> numpy
# ===----------------------------------------------------------------------===#


def test_round_trip_random_shapes() raises:
    """Round-trip for random data across various shapes."""
    var np = Python.import_module("numpy")
    _test_round_trip(np, 1, 1)
    _test_round_trip(np, 2, 3)
    _test_round_trip(np, 5, 5)
    _test_round_trip(np, 1, 10)
    _test_round_trip(np, 10, 1)
    _test_round_trip(np, 7, 13)
    _test_round_trip(np, 16, 16)


def test_round_trip_integer_data() raises:
    """Round-trip preserves integer-valued data."""
    var np = Python.import_module("numpy")
    var a_np = np.arange(1.0, 21.0).reshape(4, 5)
    var mat = matrix_from_numpy(a_np)
    var back = to_numpy(mat)
    testing.assert_true(
        Bool(np.array_equal(a_np, back)),
        msg="integer round-trip failed",
    )


def test_round_trip_f_contiguous() raises:
    """Round-trip from F-contiguous numpy array."""
    var np = Python.import_module("numpy")
    var a_np = np.asfortranarray(np.random.rand(4, 6))
    var mat = matrix_from_numpy(a_np)
    var back = to_numpy(mat)
    testing.assert_true(
        Bool(np.allclose(a_np, back)),
        msg="F-contiguous round-trip failed",
    )


# ===----------------------------------------------------------------------===#
# Creation routines vs numpy equivalents
# ===----------------------------------------------------------------------===#


def test_zeros_vs_numpy() raises:
    """Zeros(m, n) matches np.zeros(m, n)."""
    var np = Python.import_module("numpy")
    _test_zeros(np, 3, 4)
    _test_zeros(np, 1, 1)
    _test_zeros(np, 10, 10)


def test_ones_vs_numpy() raises:
    """Ones(m, n) matches np.ones(m, n)."""
    var np = Python.import_module("numpy")
    _test_ones(np, 2, 5)
    _test_ones(np, 1, 1)
    _test_ones(np, 8, 8)


def test_full_vs_numpy() raises:
    """Full(m, n, v) matches np.full((m, n), v)."""
    var np = Python.import_module("numpy")
    var mat = full[DType.float64](4, 6, 3.14)
    var back = to_numpy(mat)
    var expected = np.zeros(24).reshape(4, 6) + 3.14
    testing.assert_true(
        Bool(np.allclose(back, expected)),
        msg="full mismatch",
    )


def test_eye_vs_numpy() raises:
    """Eye(n) matches np.eye(n)."""
    var np = Python.import_module("numpy")
    _test_eye(np, 1)
    _test_eye(np, 3)
    _test_eye(np, 5)
    _test_eye(np, 10)


def test_diag_construct_vs_numpy() raises:
    """Diag([v1, v2, ...]) matches np.diag([v1, v2, ...])."""
    var np = Python.import_module("numpy")
    var vals: List[Float64] = [1.0, 2.0, 3.0, 4.0, 5.0]
    var mat = diag(vals.copy())
    var back = to_numpy(mat)
    var expected = np.diag(np.arange(1.0, 6.0))
    testing.assert_true(
        Bool(np.array_equal(back, expected)),
        msg="diag construct mismatch",
    )


def test_diag_extract_vs_numpy() raises:
    """Diag(A) matches np.diag(A) for extracting diagonal."""
    var np = Python.import_module("numpy")
    var a_np = np.random.rand(5, 5)
    var a = matrix_from_numpy(a_np)
    var d = diag(a)
    var d_np = np.diag(a_np)
    for i in range(5):
        var got = Float64(d[i])
        var exp = Float64(py=d_np[i])
        var diff = got - exp
        if diff < 0:
            diff = -diff
        testing.assert_true(
            diff < 1e-12,
            msg="diag extract mismatch at " + String(i),
        )


# ===----------------------------------------------------------------------===#
# Test runner
# ===----------------------------------------------------------------------===#


def main() raises:
    testing.TestSuite.discover_tests[__functions_in_module()]().run()
