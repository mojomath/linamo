"""
Tests for view-based linalg operations (transpose, trace, lu, cholesky, qr,
det, solve, inv, lstsq on MatrixView inputs).
"""

import std.testing as testing
from linamo.types.matrix import Matrix
from linamo.routines.creation import matrix, eye
from linamo.routines.linalg import (
    transpose,
    trace,
    lu,
    cholesky,
    qr,
    det,
    solve,
    inv,
    lstsq,
)
from linamo.routines.math import matmul
from linamo.utils.test_utils import assert_matrices_close


# ===----------------------------------------------------------------------===#
# transpose on view
# ===----------------------------------------------------------------------===#


def test_transpose_view() raises:
    """Transpose a 2x2 sub-view of a 3x3 matrix."""
    var a = matrix[DType.float64](
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
    )
    var v = a[0:2, 0:2]  # [[1, 2], [4, 5]]
    var t = transpose(v)
    testing.assert_equal(t[0, 0], 1.0)
    testing.assert_equal(t[0, 1], 4.0)
    testing.assert_equal(t[1, 0], 2.0)
    testing.assert_equal(t[1, 1], 5.0)


def test_transpose_view_rectangular() raises:
    """Transpose a 2x3 sub-view."""
    var a = matrix[DType.float64](
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
    )
    var v = a[0:2, 0:3]  # [[1, 2, 3], [4, 5, 6]]
    var t = transpose(v)
    # Should be 3x2
    testing.assert_equal(t.nrows, 3)
    testing.assert_equal(t.ncols, 2)
    testing.assert_equal(t[0, 0], 1.0)
    testing.assert_equal(t[2, 1], 6.0)


# ===----------------------------------------------------------------------===#
# trace on view
# ===----------------------------------------------------------------------===#


def test_trace_view() raises:
    """Trace of a 2x2 sub-view."""
    var a = matrix[DType.float64](
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
    )
    var v = a[0:2, 0:2]  # [[1, 2], [4, 5]]
    var t = trace(v)
    testing.assert_equal(t, 6.0)  # 1 + 5


# ===----------------------------------------------------------------------===#
# lu on view
# ===----------------------------------------------------------------------===#


def test_lu_view() raises:
    """LU decomposition on a 2x2 sub-view, then verify PA=LU."""
    var a = matrix[DType.float64](
        [[4.0, 3.0, 99.0], [6.0, 3.0, 99.0], [99.0, 99.0, 99.0]]
    )
    var v = a[0:2, 0:2]  # [[4, 3], [6, 3]]
    var result = lu(v)
    ref L = result[0]
    ref U = result[1]
    ref piv = result[2]

    # Verify L is unit lower triangular
    testing.assert_equal(L[0, 0], 1.0)
    testing.assert_equal(L[1, 1], 1.0)

    # Verify U is upper triangular
    testing.assert_equal(U[1, 0], 0.0)

    # Verify PA = LU via reconstruction
    var LU = matmul(L, U)
    for i in range(2):
        for j in range(2):
            var src_row = piv[i]
            var diff = LU[i, j] - v[src_row, j]
            if diff < 0:
                diff = -diff
            testing.assert_true(diff < 1e-10, "PA = LU check")


# ===----------------------------------------------------------------------===#
# cholesky on view
# ===----------------------------------------------------------------------===#


def test_cholesky_view() raises:
    """Cholesky decomposition on a 2x2 sub-view of an SPD matrix."""
    # Build 3x3 with a 2x2 SPD top-left block: [[4, 2], [2, 5]]
    var a = matrix[DType.float64](
        [[4.0, 2.0, 0.0], [2.0, 5.0, 0.0], [0.0, 0.0, 1.0]]
    )
    var v = a[0:2, 0:2]
    var L = cholesky(v)
    # Verify L L^T = A_view
    var LLt = matmul(L, transpose(L))
    for i in range(2):
        for j in range(2):
            var d = LLt[i, j] - v[i, j]
            if d < 0:
                d = -d
            testing.assert_true(d < 1e-10, "cholesky LL^T=A check")


# ===----------------------------------------------------------------------===#
# qr on view
# ===----------------------------------------------------------------------===#


def test_qr_view() raises:
    """QR decomposition on a 2x2 sub-view."""
    var a = matrix[DType.float64](
        [[1.0, 2.0, 99.0], [3.0, 4.0, 99.0], [99.0, 99.0, 99.0]]
    )
    var v = a[0:2, 0:2]  # [[1, 2], [3, 4]]
    var result = qr(v)
    ref Q = result[0]
    _ = result[1]
    # Check Q is orthogonal: Q^T Q ≈ I
    var qtq = matmul(transpose(Q), Q)
    for i in range(2):
        for j in range(2):
            var expected: Float64 = 0.0
            if i == j:
                expected = 1.0
            var d = qtq[i, j] - expected
            if d < 0:
                d = -d
            testing.assert_true(d < 1e-10, "QR orthogonality check")


# ===----------------------------------------------------------------------===#
# det on view
# ===----------------------------------------------------------------------===#


def test_det_view() raises:
    """Determinant of a 2x2 sub-view."""
    var a = matrix[DType.float64](
        [[1.0, 2.0, 99.0], [3.0, 4.0, 99.0], [99.0, 99.0, 99.0]]
    )
    var v = a[0:2, 0:2]  # [[1, 2], [3, 4]], det = -2
    var d = det(v)
    var diff = d - (-2.0)
    if diff < 0:
        diff = -diff
    testing.assert_true(diff < 1e-10, "det of sub-view")


# ===----------------------------------------------------------------------===#
# solve on views (view×view, mat×view, view×mat)
# ===----------------------------------------------------------------------===#


def test_solve_view_view() raises:
    """Solve Ax = b where A and b are views."""
    var big_a = matrix[DType.float64](
        [[2.0, 1.0, 99.0], [5.0, 3.0, 99.0], [99.0, 99.0, 99.0]]
    )
    var big_b = matrix[DType.float64]([[5.0, 99.0], [13.0, 99.0], [99.0, 99.0]])
    var va = big_a[0:2, 0:2]
    var vb = big_b[0:2, 0:1]
    var x = solve(va, vb)
    var expected = matrix[DType.float64]([[2.0], [1.0]])
    assert_matrices_close(x, expected, msg="solve vv", atol=1e-10)


def test_solve_mat_view() raises:
    """Solve Ax = b where A is Matrix, b is view."""
    var A = matrix[DType.float64]([[2.0, 1.0], [5.0, 3.0]])
    var big_b = matrix[DType.float64]([[5.0, 99.0], [13.0, 99.0]])
    var vb = big_b[0:2, 0:1]
    var x = solve(A, vb)
    var expected = matrix[DType.float64]([[2.0], [1.0]])
    assert_matrices_close(x, expected, msg="solve mv", atol=1e-10)


def test_solve_view_mat() raises:
    """Solve Ax = b where A is view, b is Matrix."""
    var big_a = matrix[DType.float64](
        [[2.0, 1.0, 99.0], [5.0, 3.0, 99.0], [99.0, 99.0, 99.0]]
    )
    var va = big_a[0:2, 0:2]
    var b = matrix[DType.float64]([[5.0], [13.0]])
    var x = solve(va, b)
    var expected = matrix[DType.float64]([[2.0], [1.0]])
    assert_matrices_close(x, expected, msg="solve vm", atol=1e-10)


# ===----------------------------------------------------------------------===#
# inv on view
# ===----------------------------------------------------------------------===#


def test_inv_view() raises:
    """Inverse of a 2x2 sub-view."""
    var big = matrix[DType.float64](
        [[2.0, 1.0, 99.0], [5.0, 3.0, 99.0], [99.0, 99.0, 99.0]]
    )
    var v = big[0:2, 0:2]  # [[2, 1], [5, 3]]
    var inv_v = inv(v)
    # inv([[2,1],[5,3]]) = [[3,-1],[-5,2]]
    var expected = matrix[DType.float64]([[3.0, -1.0], [-5.0, 2.0]])
    assert_matrices_close(inv_v, expected, msg="inv of sub-view", atol=1e-10)


# ===----------------------------------------------------------------------===#
# lstsq on views
# ===----------------------------------------------------------------------===#


def test_lstsq_view_view() raises:
    """Least-squares on two views (exact system: square)."""
    var big_a = matrix[DType.float64](
        [[1.0, 1.0, 99.0], [1.0, 2.0, 99.0], [99.0, 99.0, 99.0]]
    )
    var big_b = matrix[DType.float64]([[3.0, 99.0], [5.0, 99.0], [99.0, 99.0]])
    var va = big_a[0:2, 0:2]
    var vb = big_b[0:2, 0:1]
    var x = lstsq(va, vb)
    var expected = matrix[DType.float64]([[1.0], [2.0]])
    assert_matrices_close(x, expected, msg="lstsq view exact", atol=1e-9)


def test_lstsq_overdetermined_view() raises:
    """Least-squares on an overdetermined system using views."""
    var A = matrix[DType.float64]([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
    var b = matrix[DType.float64]([[1.0], [2.0], [2.0]])
    var x = lstsq(A.view(), b.view())
    # Verify via reconstruction: Ax should be close to b
    var Ax = matmul(A, x)
    var ssr: Float64 = 0
    for i in range(3):
        var r = Ax[i, 0] - b[i, 0]
        ssr += r * r
    testing.assert_true(ssr < 1.0, "lstsq overdetermined residual small")


def main() raises:
    testing.TestSuite.discover_tests[__functions_in_module()]().run()
