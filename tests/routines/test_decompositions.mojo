"""
Tests for matrix decompositions: LU, Cholesky, QR.
"""

import std.testing as testing
import linamo as la
from linamo.types.matrix import Matrix
from linamo.routines.linalg import lu, cholesky, qr, transpose
from linamo.routines.math import matmul


# ===----------------------------------------------------------------------===#
# Helper: approximate equality for floating-point matrices
# ===----------------------------------------------------------------------===#


def assert_matrix_close(
    a: Matrix[DType.float64],
    b: Matrix[DType.float64],
    tol: Float64 = 1e-10,
) raises:
    """Assert that two matrices are element-wise close within tolerance."""
    testing.assert_equal(a.nrows(), b.nrows())
    testing.assert_equal(a.ncols(), b.ncols())
    for i in range(a.nrows()):
        for j in range(a.ncols()):
            var diff = a[i, j] - b[i, j]
            if diff < 0:
                diff = -diff
            testing.assert_true(
                diff < tol,
                msg=String(
                    "Mismatch at ("
                    + String(i)
                    + ","
                    + String(j)
                    + "): "
                    + String(a[i, j])
                    + " vs "
                    + String(b[i, j])
                ),
            )


# ===----------------------------------------------------------------------===#
# Helper: permute matrix rows by pivot vector
# ===----------------------------------------------------------------------===#


def permute_rows(
    mat: Matrix[DType.float64], piv: List[Int]
) raises -> Matrix[DType.float64]:
    """Return a new matrix with rows reordered according to piv."""
    var n = mat.nrows()
    var data = List[Float64](unsafe_uninit_length=n * mat.ncols())
    for i in range(n):
        var src = piv[i]
        for j in range(mat.ncols()):
            data[i * mat.ncols() + j] = mat[src, j]
    return Matrix[DType.float64](
        buffer=data^,
        nrows=n,
        ncols=mat.ncols(),
        row_stride=mat.ncols(),
        col_stride=1,
    )


# ===----------------------------------------------------------------------===#
# LU Decomposition Tests
# ===----------------------------------------------------------------------===#


def test_lu_basic() raises:
    """Test LU decomposition of a 3x3 matrix: PA = LU."""
    var a = la.matrix[DType.float64](
        [[2.0, 3.0, 1.0], [4.0, 7.0, 5.0], [6.0, 18.0, 22.0]]
    )
    var result = lu(a)
    ref L = result[0]
    ref U = result[1]
    ref piv = result[2]

    # Verify PA == LU.
    var PA = permute_rows(a, piv)
    var LU = matmul(L, U)
    assert_matrix_close(PA, LU)


def test_lu_identity() raises:
    """LU of identity should give L=I, U=I."""
    var a = la.eye[DType.float64](3)
    var result = lu(a)
    ref L = result[0]
    ref U = result[1]

    var I = la.eye[DType.float64](3)
    assert_matrix_close(L, I)
    assert_matrix_close(U, I)


def test_lu_2x2() raises:
    """Test LU on a simple 2x2 matrix."""
    var a = la.matrix[DType.float64]([[4.0, 3.0], [6.0, 3.0]])
    var result = lu(a)
    ref L = result[0]
    ref U = result[1]
    ref piv = result[2]

    var PA = permute_rows(a, piv)
    var LU = matmul(L, U)
    assert_matrix_close(PA, LU)


def test_lu_lower_triangular() raises:
    """L should be unit lower-triangular (ones on diagonal)."""
    var a = la.matrix[DType.float64](
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0]]
    )
    var result = lu(a)
    ref L = result[0]

    for i in range(L.nrows()):
        # Diagonal must be 1.
        testing.assert_equal(L[i, i], 1.0)
        # Above diagonal must be 0.
        for j in range(i + 1, L.ncols()):
            testing.assert_equal(L[i, j], 0.0)


def test_lu_upper_triangular() raises:
    """U should be upper-triangular (zeros below diagonal)."""
    var a = la.matrix[DType.float64](
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0]]
    )
    var result = lu(a)
    ref U = result[1]

    for i in range(U.nrows()):
        for j in range(i):
            testing.assert_equal(U[i, j], 0.0)


def test_lu_nonsquare_raises() raises:
    """LU should raise ValueError for non-square matrix."""
    var a = la.matrix[DType.float64]([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    var raised = False
    try:
        _ = lu(a)
    except:
        raised = True
    testing.assert_true(raised, msg="Expected ValueError for non-square")


def test_lu_col_major() raises:
    """LU should work on column-major matrices."""
    var a = la.matrix[DType.float64]([[2.0, 1.0], [5.0, 3.0]], order="F")
    var result = lu(a)
    ref L = result[0]
    ref U = result[1]
    ref piv = result[2]

    var PA = permute_rows(a, piv)
    var LU = matmul(L, U)
    assert_matrix_close(PA, LU)


# ===----------------------------------------------------------------------===#
# Cholesky Decomposition Tests
# ===----------------------------------------------------------------------===#


def test_cholesky_basic() raises:
    """Test Cholesky on a 3x3 positive-definite matrix: A = L L^T."""
    var a = la.matrix[DType.float64](
        [[4.0, 12.0, -16.0], [12.0, 37.0, -43.0], [-16.0, -43.0, 98.0]]
    )
    var L = cholesky(a)

    # Verify L is lower-triangular.
    for i in range(L.nrows()):
        for j in range(i + 1, L.ncols()):
            testing.assert_equal(L[i, j], 0.0)

    # Verify A == L @ L^T.
    var Lt = transpose(L)
    var LLt = matmul(L, Lt)
    assert_matrix_close(a, LLt)


def test_cholesky_identity() raises:
    """Cholesky of identity should give identity."""
    var I = la.eye[DType.float64](4)
    var L = cholesky(I)
    assert_matrix_close(L, I)


def test_cholesky_2x2() raises:
    """Test Cholesky on a simple 2x2 SPD matrix."""
    var a = la.matrix[DType.float64]([[25.0, 15.0], [15.0, 13.0]])
    var L = cholesky(a)

    var Lt = transpose(L)
    var LLt = matmul(L, Lt)
    assert_matrix_close(a, LLt)


def test_cholesky_not_positive_definite_raises() raises:
    """Cholesky should raise for non-positive-definite matrix."""
    var a = la.matrix[DType.float64](
        [[1.0, 2.0], [2.0, 1.0]]  # eigenvalues: 3 and -1
    )
    var raised = False
    try:
        _ = cholesky(a)
    except:
        raised = True
    testing.assert_true(
        raised, msg="Expected ValueError for non-positive-definite"
    )


def test_cholesky_nonsquare_raises() raises:
    """Cholesky should raise for non-square matrix."""
    var a = la.matrix[DType.float64]([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    var raised = False
    try:
        _ = cholesky(a)
    except:
        raised = True
    testing.assert_true(raised, msg="Expected ValueError for non-square")


def test_cholesky_col_major() raises:
    """Cholesky should work with column-major input."""
    var a = la.matrix[DType.float64](
        [[4.0, 12.0, -16.0], [12.0, 37.0, -43.0], [-16.0, -43.0, 98.0]],
        order="F",
    )
    var L = cholesky(a)
    var Lt = transpose(L)
    var LLt = matmul(L, Lt)
    assert_matrix_close(a, LLt)


# ===----------------------------------------------------------------------===#
# QR Decomposition Tests
# ===----------------------------------------------------------------------===#


def test_qr_basic() raises:
    """Test QR decomposition: A = QR."""
    var a = la.matrix[DType.float64](
        [[1.0, -1.0, 4.0], [1.0, 4.0, -2.0], [1.0, 4.0, 2.0]]
    )
    var result = qr(a)
    ref Q = result[0]
    ref R = result[1]

    # Verify A == Q @ R.
    var QR = matmul(Q, R)
    assert_matrix_close(a, QR, tol=1e-9)


def test_qr_orthogonal_q() raises:
    """Q should be orthogonal: Q^T Q = I."""
    var a = la.matrix[DType.float64]([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    var result = qr(a)
    ref Q = result[0]

    # Q^T @ Q should be identity (m x m).
    var Qt = transpose(Q)
    var QtQ = matmul(Qt, Q)
    var I = la.eye[DType.float64](Q.nrows())
    assert_matrix_close(QtQ, I, tol=1e-9)


def test_qr_upper_triangular_r() raises:
    """R should be upper-triangular."""
    var a = la.matrix[DType.float64](
        [[12.0, -51.0, 4.0], [6.0, 167.0, -68.0], [-4.0, 24.0, -41.0]]
    )
    var result = qr(a)
    ref R = result[1]

    for i in range(R.nrows()):
        for j in range(min(i, R.ncols())):
            var val = R[i, j]
            if val < 0:
                val = -val
            testing.assert_true(
                val < 1e-10,
                msg=String(
                    "R["
                    + String(i)
                    + ","
                    + String(j)
                    + "] = "
                    + String(R[i, j])
                    + " should be ~0"
                ),
            )


def test_qr_identity() raises:
    """QR of identity: Q @ R should equal I."""
    var I = la.eye[DType.float64](3)
    var result = qr(I)
    ref Q = result[0]
    ref R = result[1]

    var QR = matmul(Q, R)
    assert_matrix_close(QR, I, tol=1e-10)


def test_qr_rectangular() raises:
    """QR decomposition for a tall rectangular matrix (m > n)."""
    var a = la.matrix[DType.float64]([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    var result = qr(a)
    ref Q = result[0]
    ref R = result[1]

    testing.assert_equal(Q.nrows(), 3)
    testing.assert_equal(Q.ncols(), 3)
    testing.assert_equal(R.nrows(), 3)
    testing.assert_equal(R.ncols(), 2)

    var QR = matmul(Q, R)
    assert_matrix_close(a, QR, tol=1e-9)


def test_qr_m_less_than_n_raises() raises:
    """QR should raise when m < n."""
    var a = la.matrix[DType.float64]([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    var raised = False
    try:
        _ = qr(a)
    except:
        raised = True
    testing.assert_true(raised, msg="Expected ValueError for m < n")


def test_qr_col_major() raises:
    """QR should work with column-major input."""
    var a = la.matrix[DType.float64](
        [[1.0, -1.0, 4.0], [1.0, 4.0, -2.0], [1.0, 4.0, 2.0]], order="F"
    )
    var result = qr(a)
    ref Q = result[0]
    ref R = result[1]
    var QR = matmul(Q, R)
    assert_matrix_close(a, QR, tol=1e-9)


def main() raises:
    testing.TestSuite.discover_tests[__functions_in_module()]().run()
