"""
Tests for Phase 3 — Solvers & Inverse: det, solve, inv, lstsq.
"""

import std.testing as testing
from linamo.types.matrix import Matrix
from linamo.routines.creation import matrix, eye, zeros
from linamo.routines.linalg import det, solve, inv, lstsq, transpose
from linamo.routines.math import matmul
from linamo.utils.test_utils import assert_matrices_close


# ===----------------------------------------------------------------------===#
# det() tests
# ===----------------------------------------------------------------------===#


def test_det_identity() raises:
    """Det of identity matrix is 1."""
    var I = eye[DType.float64](3)
    var d = det(I)
    testing.assert_true(
        (d - 1.0) < 1e-12 and (d - 1.0) > -1e-12,
        msg="det(I) should be 1, got " + String(d),
    )


def test_det_2x2() raises:
    """Det of [[1,2],[3,4]] = 1*4 - 2*3 = -2."""
    var A = matrix[DType.float64]([[1.0, 2.0], [3.0, 4.0]])
    var d = det(A)
    var diff = d - (-2.0)
    if diff < 0:
        diff = -diff
    testing.assert_true(
        diff < 1e-10,
        msg="det([[1,2],[3,4]]) should be -2, got " + String(d),
    )


def test_det_3x3() raises:
    """Det of a known 3x3 matrix."""
    # [[2,1,1],[1,3,2],[1,0,0]] -> det = 2*(0) - 1*(0-2) + 1*(0-3) = 0+2-3 = -1
    var A = matrix[DType.float64](
        [[2.0, 1.0, 1.0], [1.0, 3.0, 2.0], [1.0, 0.0, 0.0]]
    )
    var d = det(A)
    var diff = d - (-1.0)
    if diff < 0:
        diff = -diff
    testing.assert_true(
        diff < 1e-10,
        msg="det should be -1, got " + String(d),
    )


def test_det_singular() raises:
    """Det of a singular matrix is 0."""
    var A = matrix[DType.float64]([[1.0, 2.0], [2.0, 4.0]])
    var d = det(A)
    if d < 0:
        d = -d
    testing.assert_true(
        d < 1e-10,
        msg="det of singular matrix should be ~0, got " + String(d),
    )


def test_det_1x1() raises:
    """Det of 1x1 matrix [[v]] = v."""
    var A = matrix[DType.float64]([[7.0]])
    var d = det(A)
    var diff = d - 7.0
    if diff < 0:
        diff = -diff
    testing.assert_true(
        diff < 1e-12,
        msg="det([[7]]) should be 7, got " + String(d),
    )


def test_det_nonsquare_raises() raises:
    """Det of non-square matrix raises ValueError."""
    var A = matrix[DType.float64]([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    var raised = False
    try:
        _ = det(A)
    except:
        raised = True
    testing.assert_true(raised, msg="det should raise for non-square matrix")


# ===----------------------------------------------------------------------===#
# solve() tests
# ===----------------------------------------------------------------------===#


def test_solve_2x2() raises:
    """Solve [[2,1],[1,3]] x = [[5],[7]] -> x = [[8/5],[9/5]]."""
    var A = matrix[DType.float64]([[2.0, 1.0], [1.0, 3.0]])
    var b = matrix[DType.float64]([[5.0], [7.0]])
    var x = solve(A, b)
    # Expected: x = [8/5, 9/5] = [1.6, 1.8]
    var diff0 = x[0, 0] - 1.6
    if diff0 < 0:
        diff0 = -diff0
    var diff1 = x[1, 0] - 1.8
    if diff1 < 0:
        diff1 = -diff1
    testing.assert_true(
        diff0 < 1e-10 and diff1 < 1e-10,
        msg="solve 2x2 failed: got ["
        + String(x[0, 0])
        + ", "
        + String(x[1, 0])
        + "]",
    )


def test_solve_identity() raises:
    """Solve I @ x = b -> x = b."""
    var I = eye[DType.float64](3)
    var b = matrix[DType.float64]([[1.0], [2.0], [3.0]])
    var x = solve(I, b)
    assert_matrices_close(x, b, msg="solve(I, b) should equal b", atol=1e-12)


def test_solve_3x3() raises:
    """Solve a known 3x3 system and verify Ax = b."""
    var A = matrix[DType.float64](
        [[2.0, 1.0, -1.0], [-3.0, -1.0, 2.0], [-2.0, 1.0, 2.0]]
    )
    var b = matrix[DType.float64]([[8.0], [-11.0], [-3.0]])
    var x = solve(A, b)
    # Verify: Ax should equal b
    var Ax = matmul(A, x)
    assert_matrices_close(Ax, b, msg="A @ solve(A,b) should equal b", atol=1e-9)


def test_solve_multiple_rhs() raises:
    """Solve A @ X = B for multiple right-hand sides (n x k)."""
    var A = matrix[DType.float64]([[4.0, 1.0], [1.0, 3.0]])
    var B = matrix[DType.float64]([[5.0, 9.0], [7.0, 4.0]])
    var X = solve(A, B)
    # Verify: A @ X should equal B
    var AX = matmul(A, X)
    assert_matrices_close(AX, B, msg="A @ solve(A,B) should equal B", atol=1e-9)


def test_solve_dimension_mismatch_raises() raises:
    """Solve with mismatched dimensions raises."""
    var A = matrix[DType.float64]([[1.0, 2.0], [3.0, 4.0]])
    var b = matrix[DType.float64]([[1.0], [2.0], [3.0]])
    var raised = False
    try:
        _ = solve(A, b)
    except:
        raised = True
    testing.assert_true(raised, msg="solve should raise for dim mismatch")


def test_solve_nonsquare_raises() raises:
    """Solve with non-square A raises."""
    var A = matrix[DType.float64]([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    var b = matrix[DType.float64]([[1.0], [2.0]])
    var raised = False
    try:
        _ = solve(A, b)
    except:
        raised = True
    testing.assert_true(raised, msg="solve should raise for non-square A")


# ===----------------------------------------------------------------------===#
# inv() tests
# ===----------------------------------------------------------------------===#


def test_inv_2x2() raises:
    """Inv of [[2,1],[1,3]] and verify A @ A^{-1} = I."""
    var A = matrix[DType.float64]([[2.0, 1.0], [1.0, 3.0]])
    var A_inv = inv(A)
    var product = matmul(A, A_inv)
    var I = eye[DType.float64](2)
    assert_matrices_close(product, I, msg="A @ inv(A) should be I", atol=1e-10)


def test_inv_identity() raises:
    """Inv of identity is identity."""
    var I = eye[DType.float64](4)
    var I_inv = inv(I)
    assert_matrices_close(I_inv, I, msg="inv(I) should be I", atol=1e-12)


def test_inv_3x3() raises:
    """Inv of a 3x3 matrix, verify round-trip."""
    var A = matrix[DType.float64](
        [[2.0, 1.0, -1.0], [-3.0, -1.0, 2.0], [-2.0, 1.0, 2.0]]
    )
    var A_inv = inv(A)
    var product = matmul(A, A_inv)
    var I = eye[DType.float64](3)
    assert_matrices_close(product, I, msg="A @ inv(A) should be I", atol=1e-9)


def test_inv_double_inverse() raises:
    """Inv(inv(A)) should equal A."""
    var A = matrix[DType.float64]([[4.0, 7.0], [2.0, 6.0]])
    var A_inv_inv = inv(inv(A))
    assert_matrices_close(
        A_inv_inv,
        A,
        msg="inv(inv(A)) should equal A",
        atol=1e-9,
    )


def test_inv_nonsquare_raises() raises:
    """Inv of non-square matrix raises."""
    var A = matrix[DType.float64]([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    var raised = False
    try:
        _ = inv(A)
    except:
        raised = True
    testing.assert_true(raised, msg="inv should raise for non-square matrix")


# ===----------------------------------------------------------------------===#
# lstsq() tests
# ===----------------------------------------------------------------------===#


def test_lstsq_exact_square() raises:
    """Least squares on a square system should give the exact solution."""
    var A = matrix[DType.float64]([[2.0, 1.0], [1.0, 3.0]])
    var b = matrix[DType.float64]([[5.0], [7.0]])
    var x = lstsq(A, b)
    # Should match solve exactly
    var x_exact = solve(A, b)
    assert_matrices_close(x, x_exact, msg="lstsq exact", atol=1e-9)


def test_lstsq_overdetermined() raises:
    """Least squares on a simple overdetermined system."""
    # y = 2x + 1. Points: (1,3), (2,5), (3,7). Exact fit.
    # A = [[1,1],[1,2],[1,3]], b = [[3],[5],[7]]
    var A = matrix[DType.float64]([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
    var b = matrix[DType.float64]([[3.0], [5.0], [7.0]])
    var x = lstsq(A, b)
    # Expected: x = [1, 2] (intercept=1, slope=2)
    var diff0 = x[0, 0] - 1.0
    if diff0 < 0:
        diff0 = -diff0
    var diff1 = x[1, 0] - 2.0
    if diff1 < 0:
        diff1 = -diff1
    testing.assert_true(
        diff0 < 1e-9 and diff1 < 1e-9,
        msg="lstsq overdetermined: got ["
        + String(x[0, 0])
        + ", "
        + String(x[1, 0])
        + "] expected [1, 2]",
    )


def test_lstsq_noisy_overdetermined() raises:
    """Least squares minimizes ||Ax - b||: verify normal equations."""
    # A^T A x = A^T b should hold for the lstsq solution.
    var A = matrix[DType.float64](
        [
            [1.0, 1.0],
            [1.0, 2.0],
            [1.0, 3.0],
            [1.0, 4.0],
            [1.0, 5.0],
        ]
    )
    var b = matrix[DType.float64]([[2.1], [3.9], [6.2], [7.8], [10.1]])
    var x = lstsq(A, b)
    # Verify normal equations: A^T A x = A^T b
    var AtA = matmul(transpose(A), A)
    var Atb = matmul(transpose(A), b)
    var AtAx = matmul(AtA, x)
    assert_matrices_close(
        AtAx,
        Atb,
        msg="Normal equations A^T A x = A^T b",
        atol=1e-9,
    )


def test_lstsq_multiple_rhs() raises:
    """Least squares with multiple right-hand sides."""
    var A = matrix[DType.float64]([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
    var B = matrix[DType.float64]([[3.0, 1.0], [5.0, 2.0], [7.0, 3.0]])
    var X = lstsq(A, B)
    testing.assert_equal(X.nrows, 2)
    testing.assert_equal(X.ncols, 2)
    # Verify: A^T A X = A^T B (normal equations)
    var AtA = matmul(transpose(A), A)
    var AtB = matmul(transpose(A), B)
    var AtAX = matmul(AtA, X)
    assert_matrices_close(
        AtAX,
        AtB,
        msg="Normal equations for multiple RHS",
        atol=1e-9,
    )


def test_lstsq_m_less_than_n_raises() raises:
    """Lstsq with m < n raises."""
    var A = matrix[DType.float64]([[1.0, 2.0, 3.0]])
    var b = matrix[DType.float64]([[1.0]])
    var raised = False
    try:
        _ = lstsq(A, b)
    except:
        raised = True
    testing.assert_true(raised, msg="lstsq should raise for m < n")


def test_lstsq_dim_mismatch_raises() raises:
    """Lstsq with mismatched A and b rows raises."""
    var A = matrix[DType.float64]([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
    var b = matrix[DType.float64]([[1.0], [2.0]])
    var raised = False
    try:
        _ = lstsq(A, b)
    except:
        raised = True
    testing.assert_true(raised, msg="lstsq should raise for dim mismatch")


# ===----------------------------------------------------------------------===#
# Test runner
# ===----------------------------------------------------------------------===#


def main() raises:
    testing.TestSuite.discover_tests[__functions_in_module()]().run()
