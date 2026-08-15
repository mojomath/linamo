"""
Numpy-powered integration tests for Phase 3 — Solvers & Inverse.

Tests det, solve, inv, lstsq against numpy.linalg as ground truth.
"""

import std.testing as testing
from linamo.routines.linalg import det, solve, inv, lstsq, transpose
from linamo.routines.math import matmul
from linamo.routines.numpy_interop import matrix_from_numpy, to_numpy
from linamo.utils.test_utils import assert_matrices_close
from std.python import Python, PythonObject


# ===----------------------------------------------------------------------===#
# Helpers
# ===----------------------------------------------------------------------===#


def _make_invertible(np: PythonObject, n: Int) raises -> PythonObject:
    """Create a random n×n invertible matrix (diagonal boost)."""
    return np.random.rand(n, n) + np.eye(n) * Float64(n)


# ===----------------------------------------------------------------------===#
# det() vs numpy
# ===----------------------------------------------------------------------===#


def _test_det(np: PythonObject, n: Int) raises:
    """Helper: compare det(A) vs np.linalg.det(A)."""
    var a_np = _make_invertible(np, n)
    var a = matrix_from_numpy(a_np)
    var got = Float64(det(a))
    var expected = Float64(py=np.linalg.det(a_np))
    # Use relative tolerance for large determinants
    var denom = expected
    if denom < 0:
        denom = -denom
    if denom < 1.0:
        denom = 1.0
    var diff = got - expected
    if diff < 0:
        diff = -diff
    testing.assert_true(
        diff / denom < 1e-8,
        msg="det mismatch for "
        + String(n)
        + "x"
        + String(n)
        + ": got "
        + String(got)
        + " expected "
        + String(expected),
    )


def test_det_random() raises:
    """Det(A) == np.linalg.det(A) for random invertible matrices."""
    var np = Python.import_module("numpy")
    _test_det(np, 1)
    _test_det(np, 2)
    _test_det(np, 3)
    _test_det(np, 5)
    _test_det(np, 8)
    _test_det(np, 10)


def test_det_permutation_sign() raises:
    """Det correctly tracks sign through row permutations."""
    var np = Python.import_module("numpy")
    # Run several random matrices and compare signs
    for _ in range(10):
        var a_np = np.random.rand(4, 4) + np.eye(4) * 2.0
        var a = matrix_from_numpy(a_np)
        var got = Float64(det(a))
        var expected = Float64(py=np.linalg.det(a_np))
        # Signs must match
        var same_sign = (
            (got > 0 and expected > 0)
            or (got < 0 and expected < 0)
            or (got == 0.0 and expected == 0.0)
        )
        testing.assert_true(
            same_sign,
            msg="det sign mismatch: got "
            + String(got)
            + " expected "
            + String(expected),
        )


# ===----------------------------------------------------------------------===#
# solve() vs numpy
# ===----------------------------------------------------------------------===#


def _test_solve(np: PythonObject, n: Int, k: Int) raises:
    """Helper: compare solve(A, b) vs np.linalg.solve(A, b)."""
    var a_np = _make_invertible(np, n)
    var b_np = np.random.rand(n, k)
    var a = matrix_from_numpy(a_np)
    var b = matrix_from_numpy(b_np)
    var x = solve(a, b)
    var x_np = np.linalg.solve(a_np, b_np)
    var x_expected = matrix_from_numpy(x_np)
    assert_matrices_close(
        x,
        x_expected,
        msg="solve "
        + String(n)
        + "x"
        + String(n)
        + " with "
        + String(k)
        + " rhs",
        atol=1e-8,
    )


def test_solve_random() raises:
    """Solve(A, b) == np.linalg.solve(A, b) for random systems."""
    var np = Python.import_module("numpy")
    _test_solve(np, 2, 1)
    _test_solve(np, 3, 1)
    _test_solve(np, 5, 1)
    _test_solve(np, 8, 1)
    _test_solve(np, 10, 1)


def test_solve_multiple_rhs_random() raises:
    """Solve with multiple right-hand sides vs numpy."""
    var np = Python.import_module("numpy")
    _test_solve(np, 4, 3)
    _test_solve(np, 6, 5)
    _test_solve(np, 8, 2)


def test_solve_reconstruction_random() raises:
    """A @ solve(A, b) == b for random systems."""
    var np = Python.import_module("numpy")
    var a_np = _make_invertible(np, 6)
    var b_np = np.random.rand(6, 2)
    var a = matrix_from_numpy(a_np)
    var b = matrix_from_numpy(b_np)
    var x = solve(a, b)
    var Ax = matmul(a, x)
    assert_matrices_close(
        Ax,
        b,
        msg="A @ solve(A,b) == b",
        atol=1e-8,
    )


# ===----------------------------------------------------------------------===#
# inv() vs numpy
# ===----------------------------------------------------------------------===#


def _test_inv(np: PythonObject, n: Int) raises:
    """Helper: compare inv(A) vs np.linalg.inv(A)."""
    var a_np = _make_invertible(np, n)
    var a = matrix_from_numpy(a_np)
    var a_inv = inv(a)
    var a_inv_np = np.linalg.inv(a_np)
    var a_inv_expected = matrix_from_numpy(a_inv_np)
    assert_matrices_close(
        a_inv,
        a_inv_expected,
        msg="inv " + String(n) + "x" + String(n),
        atol=1e-8,
    )


def test_inv_random() raises:
    """Inv(A) == np.linalg.inv(A) for random invertible matrices."""
    var np = Python.import_module("numpy")
    _test_inv(np, 2)
    _test_inv(np, 3)
    _test_inv(np, 5)
    _test_inv(np, 8)


def test_inv_roundtrip_random() raises:
    """A @ inv(A) == I for random matrices."""
    var np = Python.import_module("numpy")
    for _ in range(3):
        var a_np = _make_invertible(np, 6)
        var a = matrix_from_numpy(a_np)
        var a_inv = inv(a)
        var product = matmul(a, a_inv)
        var I_np = np.eye(6)
        var I = matrix_from_numpy(I_np)
        assert_matrices_close(
            product,
            I,
            msg="A @ inv(A) == I",
            atol=1e-8,
        )


# ===----------------------------------------------------------------------===#
# lstsq() vs numpy
# ===----------------------------------------------------------------------===#


def _test_lstsq(np: PythonObject, m: Int, n: Int, k: Int) raises:
    """Helper: compare lstsq(A, b) vs np.linalg.lstsq(A, b)."""
    var a_np = np.random.rand(m, n)
    var b_np = np.random.rand(m, k)
    var a = matrix_from_numpy(a_np)
    var b = matrix_from_numpy(b_np)
    var x = lstsq(a, b)
    var result_np = np.linalg.lstsq(a_np, b_np, rcond=PythonObject(None))
    var x_expected = matrix_from_numpy(result_np[0])
    assert_matrices_close(
        x,
        x_expected,
        msg="lstsq "
        + String(m)
        + "x"
        + String(n)
        + " with "
        + String(k)
        + " rhs",
        atol=1e-8,
    )


def test_lstsq_random() raises:
    """Lstsq(A, b) == np.linalg.lstsq(A, b) for random overdetermined."""
    var np = Python.import_module("numpy")
    _test_lstsq(np, 5, 2, 1)
    _test_lstsq(np, 10, 3, 1)
    _test_lstsq(np, 20, 5, 1)
    _test_lstsq(np, 8, 4, 1)


def test_lstsq_multiple_rhs_random() raises:
    """Lstsq with multiple RHS vs numpy."""
    var np = Python.import_module("numpy")
    _test_lstsq(np, 10, 3, 4)
    _test_lstsq(np, 15, 5, 3)


def test_lstsq_exact_fit_random() raises:
    """Lstsq on consistent square system matches solve."""
    var np = Python.import_module("numpy")
    var a_np = _make_invertible(np, 5)
    var x_true_np = np.random.rand(5, 1)
    var b_np = np.matmul(a_np, x_true_np)
    var a = matrix_from_numpy(a_np)
    var b = matrix_from_numpy(b_np)
    var x_lstsq = lstsq(a, b)
    var x_true = matrix_from_numpy(x_true_np)
    assert_matrices_close(
        x_lstsq,
        x_true,
        msg="lstsq exact fit",
        atol=1e-8,
    )


def test_lstsq_normal_equations_random() raises:
    """A^T A x = A^T b for lstsq solution (normal equations)."""
    var np = Python.import_module("numpy")
    var a_np = np.random.rand(12, 4)
    var b_np = np.random.rand(12, 1)
    var a = matrix_from_numpy(a_np)
    var b = matrix_from_numpy(b_np)
    var x = lstsq(a, b)
    var At = transpose(a)
    var AtA = matmul(At, a)
    var Atb = matmul(At, b)
    var AtAx = matmul(AtA, x)
    assert_matrices_close(
        AtAx,
        Atb,
        msg="Normal equations",
        atol=1e-8,
    )


# ===----------------------------------------------------------------------===#
# Test runner
# ===----------------------------------------------------------------------===#


def main() raises:
    testing.TestSuite.discover_tests[__functions_in_module()]().run()
