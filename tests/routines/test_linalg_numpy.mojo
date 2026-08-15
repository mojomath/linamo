"""
Numpy-powered integration tests for linear algebra routines.

Uses numpy.random to generate matrices and compares linamo's
transpose, trace, LU, Cholesky, QR against numpy.linalg as ground truth.
"""

import std.testing as testing
from linamo.routines.creation import eye
from linamo.routines.linalg import transpose, trace, lu, cholesky, qr
from linamo.routines.math import matmul
from linamo.routines.numpy_interop import matrix_from_numpy, to_numpy
from linamo.types.matrix import Matrix
from linamo.utils.test_utils import assert_matrices_close
from std.python import Python, PythonObject


# ===----------------------------------------------------------------------===#
# Helpers
# ===----------------------------------------------------------------------===#


def _make_spd(np: PythonObject, n: Int) raises -> PythonObject:
    """Create a random n×n symmetric positive-definite numpy matrix."""
    var A = np.random.rand(n, n)
    return np.matmul(A, A.T) + np.eye(n) * Float64(n)


def _test_transpose(np: PythonObject, r: Int, c: Int) raises:
    """Helper: test transpose for one shape."""
    var a_np = np.random.rand(r, c)
    var a = matrix_from_numpy(a_np)
    var expected = matrix_from_numpy(a_np.T)
    assert_matrices_close(
        transpose(a),
        expected,
        msg="transpose " + String(r) + "x" + String(c),
        atol=1e-15,
    )


def _test_trace(np: PythonObject, n: Int) raises:
    """Helper: test trace for one size."""
    var a_np = np.random.rand(n, n)
    var a = matrix_from_numpy(a_np)
    var expected = Float64(py=np.trace(a_np))
    var got = Float64(trace(a))
    var diff = got - expected
    if diff < 0:
        diff = -diff
    testing.assert_true(
        diff < 1e-9,
        msg="trace mismatch for "
        + String(n)
        + "x"
        + String(n)
        + ": got "
        + String(got)
        + " expected "
        + String(expected),
    )


def _permute_rows_mm(
    mat: Matrix[DType.float64],
    piv: List[Int],
) raises -> Matrix[DType.float64]:
    """Reorder rows of mat according to piv."""
    var n = mat.nrows
    var data = List[Float64](unsafe_uninit_length=n * mat.ncols)
    for i in range(n):
        var src = piv[i]
        for j in range(mat.ncols):
            data[i * mat.ncols + j] = mat[src, j]
    return Matrix[DType.float64](
        data=data^,
        nrows=n,
        ncols=mat.ncols,
        row_stride=mat.ncols,
        col_stride=1,
    )


def _test_cholesky(np: PythonObject, n: Int) raises:
    """Helper: test Cholesky A == L @ L^T for one size."""
    var a_np = _make_spd(np, n)
    var a = matrix_from_numpy(a_np)
    var L = cholesky(a)
    var Lt = transpose(L)
    var LLt = matmul(L, Lt)
    assert_matrices_close(
        a,
        LLt,
        msg="cholesky " + String(n) + "x" + String(n),
        atol=1e-9,
    )


def _test_qr_square(np: PythonObject, n: Int) raises:
    """Helper: test QR A == Q @ R for one square size."""
    var a_np = np.random.rand(n, n)
    var a = matrix_from_numpy(a_np)
    var result = qr(a)
    ref Q = result[0]
    ref R = result[1]
    var QR = matmul(Q, R)
    assert_matrices_close(
        a,
        QR,
        msg="QR " + String(n) + "x" + String(n),
        atol=1e-9,
    )


def _test_qr_tall(np: PythonObject, m: Int, n: Int) raises:
    """Helper: test QR A == Q @ R for one tall shape."""
    var a_np = np.random.rand(m, n)
    var a = matrix_from_numpy(a_np)
    var result = qr(a)
    ref Q = result[0]
    ref R = result[1]
    var QR = matmul(Q, R)
    assert_matrices_close(
        a,
        QR,
        msg="QR " + String(m) + "x" + String(n),
        atol=1e-9,
    )


def _test_qr_orthogonality(np: PythonObject, n: Int) raises:
    """Helper: test Q^T @ Q == I for one square size."""
    var a_np = np.random.rand(n, n)
    var a = matrix_from_numpy(a_np)
    var result = qr(a)
    ref Q = result[0]
    var Qt = transpose(Q)
    var QtQ = matmul(Qt, Q)
    var I = eye[DType.float64](n)
    assert_matrices_close(
        QtQ,
        I,
        msg="Q^T Q == I for " + String(n) + "x" + String(n),
        atol=1e-9,
    )


# ===----------------------------------------------------------------------===#
# Transpose vs numpy
# ===----------------------------------------------------------------------===#


def test_transpose_random() raises:
    """Transpose(A) == A.T via numpy for various shapes."""
    var np = Python.import_module("numpy")
    _test_transpose(np, 1, 1)
    _test_transpose(np, 3, 5)
    _test_transpose(np, 7, 2)
    _test_transpose(np, 10, 10)
    _test_transpose(np, 1, 20)
    _test_transpose(np, 20, 1)


# ===----------------------------------------------------------------------===#
# Trace vs numpy
# ===----------------------------------------------------------------------===#


def test_trace_random() raises:
    """Trace(A) == np.trace(A) for random square matrices."""
    var np = Python.import_module("numpy")
    _test_trace(np, 1)
    _test_trace(np, 2)
    _test_trace(np, 5)
    _test_trace(np, 10)
    _test_trace(np, 20)


# ===----------------------------------------------------------------------===#
# LU vs numpy: PA == LU reconstruction
# ===----------------------------------------------------------------------===#


def test_lu_random_small() raises:
    """LU: PA == LU for random 4x4 matrices."""
    var np = Python.import_module("numpy")
    for _ in range(5):
        var a_np = np.random.rand(4, 4) + np.eye(4) * 0.1
        var a = matrix_from_numpy(a_np)
        var result = lu(a)
        ref L = result[0]
        ref U = result[1]
        ref piv = result[2]
        var PA = _permute_rows_mm(a, piv)
        var LU_result = matmul(L, U)
        assert_matrices_close(PA, LU_result, msg="LU 4x4", atol=1e-10)


def test_lu_random_medium() raises:
    """LU: PA == LU for random 10x10 matrix."""
    var np = Python.import_module("numpy")
    var a_np = np.random.rand(10, 10) + np.eye(10) * 0.1
    var a = matrix_from_numpy(a_np)
    var result = lu(a)
    ref L = result[0]
    ref U = result[1]
    ref piv = result[2]
    var PA = _permute_rows_mm(a, piv)
    var LU_result = matmul(L, U)
    assert_matrices_close(PA, LU_result, msg="LU 10x10", atol=1e-9)


# ===----------------------------------------------------------------------===#
# Cholesky vs numpy: A == L @ L^T
# ===----------------------------------------------------------------------===#


def test_cholesky_random() raises:
    """Cholesky: A == L @ L^T for random SPD matrices of various sizes."""
    var np = Python.import_module("numpy")
    _test_cholesky(np, 2)
    _test_cholesky(np, 3)
    _test_cholesky(np, 5)
    _test_cholesky(np, 8)


def test_cholesky_vs_numpy() raises:
    """Cholesky L matches np.linalg.cholesky(A) for a 5x5 SPD matrix."""
    var np = Python.import_module("numpy")
    var a_np = _make_spd(np, 5)
    var a = matrix_from_numpy(a_np)
    var L = cholesky(a)
    var L_np = np.linalg.cholesky(a_np)
    var L_expected = matrix_from_numpy(L_np)
    assert_matrices_close(
        L,
        L_expected,
        msg="Cholesky vs np.linalg.cholesky",
        atol=1e-9,
    )


# ===----------------------------------------------------------------------===#
# QR vs numpy: A == Q @ R reconstruction
# ===----------------------------------------------------------------------===#


def test_qr_random_square() raises:
    """QR: A == Q @ R for random square matrices."""
    var np = Python.import_module("numpy")
    _test_qr_square(np, 2)
    _test_qr_square(np, 3)
    _test_qr_square(np, 5)
    _test_qr_square(np, 8)


def test_qr_random_tall() raises:
    """QR: A == Q @ R for random tall matrices (m > n)."""
    var np = Python.import_module("numpy")
    _test_qr_tall(np, 6, 3)
    _test_qr_tall(np, 10, 4)
    _test_qr_tall(np, 20, 5)


def test_qr_orthogonality_random() raises:
    """QR: Q^T @ Q == I for random matrices."""
    var np = Python.import_module("numpy")
    _test_qr_orthogonality(np, 3)
    _test_qr_orthogonality(np, 5)
    _test_qr_orthogonality(np, 8)


# ===----------------------------------------------------------------------===#
# Test runner
# ===----------------------------------------------------------------------===#


def main() raises:
    testing.TestSuite.discover_tests[__functions_in_module()]().run()
