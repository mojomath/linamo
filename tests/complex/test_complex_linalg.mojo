"""
Tests for linear algebra over complex elements.

Two different kinds of work sit behind these.

`lu`, `det`, `solve`, `inv` and `matrix_power` needed no new algorithm, only a
new way to rank a pivot. They share one body parameterised on that ranking,
because pivoting was the only step where a complex element could not follow
the real one: it used to go through `Comparable`, and there is no order on the
complex plane. A real element is ranked by `-x if x < zero else x`, a complex
one by `|z|`, which is a real number and does have an order.

`cholesky`, `qr` and `lstsq` needed a real algorithm change, and it is always
the same change: the transpose in them is a *conjugate* transpose. `A = L L^H`
rather than `L L^T`; a `Q` that is unitary rather than orthogonal; a residual
made orthogonal under the conjugate inner product. Under the plain transpose
none of those quantities is what its name says, so these three are separate
implementations rather than a shared body with a parameter.

The last section checks that the element types that always had an order still
reach the answers they reached before the pivot ranking moved.
"""

import std.testing as testing
import linamo as la
from linamo import CFloat64, Dec128


def _a() raises -> la.Matrix[CFloat64]:
    """Returns `[[1+i, 2], [3, 4-i]]`, whose determinant is `-1+3i`."""
    return la.matrix[CFloat64](
        [
            [CFloat64(1, 1), CFloat64(2, 0)],
            [CFloat64(3, 0), CFloat64(4, -1)],
        ]
    )


def _assert_close(
    got: CFloat64, want: CFloat64, msg: String = "", tol: Float64 = 1e-12
) raises:
    """Asserts two complex numbers agree to within `tol` in both components.

    Elimination over `Float64` components rounds, so an exact comparison would
    be testing the rounding rather than the algorithm.
    """
    testing.assert_almost_equal(got.re(), want.re(), atol=tol, msg=msg)
    testing.assert_almost_equal(got.im(), want.im(), atol=tol, msg=msg)


def _assert_matrices_close(
    got: la.Matrix[CFloat64], want: la.Matrix[CFloat64], msg: String = ""
) raises:
    """Asserts two matrices agree entry by entry to within `_assert_close`."""
    testing.assert_equal(got.nrows(), want.nrows(), msg + " (rows)")
    testing.assert_equal(got.ncols(), want.ncols(), msg + " (cols)")
    for i in range(got.nrows()):
        for j in range(got.ncols()):
            _assert_close(got[i, j], want[i, j], msg)


# ===----------------------------------------------------------------------===#
# LU
# ===----------------------------------------------------------------------===#


def test_lu_reconstructs_the_matrix() raises:
    """`L @ U` equals the rows of A taken in the order `piv` gives."""
    var a = _a()
    var factored = la.lu(a)
    ref L = factored[0]
    ref U = factored[1]
    ref piv = factored[2]
    var product = L @ U
    for i in range(2):
        for j in range(2):
            _assert_close(product[i, j], a[piv[i], j], "L @ U")


def test_lu_is_unit_lower_triangular() raises:
    """L has `1+0i` on its diagonal and zeros above it."""
    var factored = la.lu(_a())
    ref L = factored[0]
    testing.assert_equal(L[0, 0], CFloat64.one())
    testing.assert_equal(L[1, 1], CFloat64.one())
    testing.assert_equal(L[0, 1], CFloat64.zero())


def test_lu_is_upper_triangular() raises:
    """U has zeros below the diagonal."""
    var factored = la.lu(_a())
    ref U = factored[1]
    testing.assert_equal(U[1, 0], CFloat64.zero())


def test_pivoting_ranks_by_magnitude_not_by_component() raises:
    """The row with the larger `|z|` is chosen, even when its real part is
    smaller.

    Column 0 holds `1+0i` and `0+3i`. Ranked by magnitude the second wins
    (`3 > 1`); ranked by real part, or by any ordering that looked only at
    `re`, the first would. The permutation records which happened.
    """
    var m = la.matrix[CFloat64](
        [
            [CFloat64(1, 0), CFloat64(2, 0)],
            [CFloat64(0, 3), CFloat64(1, 0)],
        ]
    )
    var factored = la.lu(m)
    ref piv = factored[2]
    testing.assert_equal(piv[0], 1, "the larger magnitude is pivoted up")


# ===----------------------------------------------------------------------===#
# Determinant
# ===----------------------------------------------------------------------===#


def test_det_is_the_complex_determinant() raises:
    """`(1+i)(4-i) - 2*3 == 5+3i - 6 == -1+3i`."""
    _assert_close(la.det(_a()), CFloat64(-1, 3), "det")


def test_det_of_the_identity_is_one() raises:
    _assert_close(la.det(la.eye[CFloat64](3)), CFloat64.one(), "det(I)")


def test_det_of_a_singular_matrix_is_zero() raises:
    """The second row is `(1+i)` times the first, so the rows are dependent."""
    var m = la.matrix[CFloat64](
        [
            [CFloat64(1, 0), CFloat64(2, 0)],
            [CFloat64(1, 1), CFloat64(2, 2)],
        ]
    )
    _assert_close(la.det(m), CFloat64.zero(), "det of a singular matrix")


def test_det_of_a_diagonal_matrix_multiplies_the_diagonal() raises:
    """`(2i)(3i) == -6`, which a real-only determinant could not produce."""
    var m = la.matrix[CFloat64](
        [
            [CFloat64(0, 2), CFloat64.zero()],
            [CFloat64.zero(), CFloat64(0, 3)],
        ]
    )
    _assert_close(la.det(m), CFloat64(-6, 0), "det of a diagonal matrix")


# ===----------------------------------------------------------------------===#
# Solve and inverse
# ===----------------------------------------------------------------------===#


def test_solve_satisfies_the_system() raises:
    """`A @ x == b` for the x that `solve` returns."""
    var a = _a()
    var b = la.matrix[CFloat64]([[CFloat64(1, 0)], [CFloat64(0, 1)]])
    var x = la.solve(a, b)
    var check = a @ x
    for i in range(2):
        _assert_close(check[i, 0], b[i, 0], "A @ x")


def test_solve_handles_several_right_hand_sides() raises:
    """A matrix right-hand side is solved column by column."""
    var a = _a()
    var b = la.eye[CFloat64](2)
    var x = la.solve(a, b)
    var check = a @ x
    for i in range(2):
        for j in range(2):
            _assert_close(check[i, j], b[i, j], "A @ X")


def test_inv_is_a_two_sided_inverse() raises:
    """`A @ inv(A)` and `inv(A) @ A` are both the identity."""
    var a = _a()
    var inverse = la.inv(a)
    var I = la.eye[CFloat64](2)
    var left = inverse @ a
    var right = a @ inverse
    for i in range(2):
        for j in range(2):
            _assert_close(left[i, j], I[i, j], "inv(A) @ A")
            _assert_close(right[i, j], I[i, j], "A @ inv(A)")


def test_inv_of_i_times_identity_is_minus_i() raises:
    """`1/i == -i`, so the inverse is not the conjugate."""
    var m = la.matrix[CFloat64](
        [
            [CFloat64(0, 1), CFloat64.zero()],
            [CFloat64.zero(), CFloat64(0, 1)],
        ]
    )
    var inverse = la.inv(m)
    _assert_close(inverse[0, 0], CFloat64(0, -1), "1/i")
    _assert_close(inverse[1, 1], CFloat64(0, -1), "1/i")


def test_solve_on_a_singular_matrix_raises() raises:
    var m = la.matrix[CFloat64](
        [
            [CFloat64(1, 0), CFloat64(2, 0)],
            [CFloat64(1, 1), CFloat64(2, 2)],
        ]
    )
    var b = la.matrix[CFloat64]([[CFloat64.one()], [CFloat64.one()]])
    with testing.assert_raises():
        _ = la.solve(m, b)


# ===----------------------------------------------------------------------===#
# Matrix power
# ===----------------------------------------------------------------------===#


def test_matrix_power_agrees_with_repeated_multiplication() raises:
    var a = _a()
    var squared = la.matrix_power(a, 2)
    var by_hand = a @ a
    for i in range(2):
        for j in range(2):
            testing.assert_equal(squared[i, j], by_hand[i, j])


def test_matrix_power_zero_is_the_identity() raises:
    var powered = la.matrix_power(_a(), 0)
    var I = la.eye[CFloat64](2)
    for i in range(2):
        for j in range(2):
            testing.assert_equal(powered[i, j], I[i, j])


def test_a_negative_power_inverts_first() raises:
    """`matrix_power(A, -1)` is `inv(A)`."""
    var a = _a()
    var powered = la.matrix_power(a, -1)
    var inverse = la.inv(a)
    for i in range(2):
        for j in range(2):
            _assert_close(powered[i, j], inverse[i, j], "A ** -1")


def test_the_power_operator_reaches_the_complex_overload() raises:
    """`A ** n` agrees with `matrix_power(A, n)`.

    The operator carries its own constraint, and the one written for a real
    element asks for `Comparable`, which a complex number is not. Without a
    second overload `matrix_power(a, -1)` would work while `a ** -1` failed to
    compile --- which is what happened before this test existed.
    """
    var a = _a()
    _assert_matrices_close(a**2, la.matrix_power(a, 2), "A ** 2")
    _assert_matrices_close(a**-1, la.matrix_power(a, -1), "A ** -1")
    _assert_matrices_close(a**0, la.eye[CFloat64](2), "A ** 0")


def test_the_power_operator_works_on_a_view() raises:
    """The same operator on `MatrixView`, which carries its own copy."""
    var a = _a()
    _assert_matrices_close(a.view() ** 2, a @ a, "view ** 2")


# ===----------------------------------------------------------------------===#
# Cholesky
# ===----------------------------------------------------------------------===#
# `A = L L^H`, the Hermitian analogue of `A = L L^T`. The test matrix is
# `[[4, 1-i], [1+i, 3]]`: Hermitian, and positive-definite because its leading
# minors are `4` and `4*3 - |1-i|^2 == 10`.


def _hermitian() raises -> la.Matrix[CFloat64]:
    return la.from_string[CFloat64]("[[4, 1-1i], [1+1i, 3]]")


def test_cholesky_reconstructs_the_matrix() raises:
    """`L @ conj_transpose(L)` is A --- with the plain transpose it is not."""
    var a = _hermitian()
    var L = la.cholesky(a)
    _assert_matrices_close(L @ la.conj_transpose(L), a, "L @ L^H")


def test_cholesky_is_lower_triangular() raises:
    var L = la.cholesky(_hermitian())
    testing.assert_equal(L[0, 1], CFloat64.zero())


def test_the_cholesky_diagonal_is_real_and_positive() raises:
    """The property that makes the factor exist: `L[i,i]^2` is a real energy.

    `L[0,0]` is `sqrt(4) == 2` and `L[1,1]` is `sqrt(3 - |0.5+0.5i|^2*2)`,
    which is `sqrt(2.5)`.
    """
    var L = la.cholesky(_hermitian())
    testing.assert_equal(L[0, 0].im(), 0.0)
    testing.assert_equal(L[1, 1].im(), 0.0)
    testing.assert_true(L[0, 0].re() > 0.0)
    testing.assert_true(L[1, 1].re() > 0.0)
    _assert_close(L[0, 0], CFloat64(2, 0), "L[0,0]")
    _assert_close(L[1, 0], CFloat64(0.5, 0.5), "L[1,0]")


def test_cholesky_rejects_a_non_positive_definite_matrix() raises:
    var m = la.from_string[CFloat64]("[[1, 2], [2, 1]]")
    with testing.assert_raises():
        _ = la.cholesky(m)


def test_cholesky_rejects_a_non_square_matrix() raises:
    var m = la.from_string[CFloat64]("[[1, 0, 0], [0, 1, 0]]")
    with testing.assert_raises():
        _ = la.cholesky(m)


# ===----------------------------------------------------------------------===#
# QR
# ===----------------------------------------------------------------------===#
# Householder over the complex field. The reflector's sign becomes a phase:
# where the real routine picks `-1` or `+1` to point away from `x_k`, this one
# picks `x_k / |x_k|`, the unit number pointing the same way.


def _tall() raises -> la.Matrix[CFloat64]:
    return la.from_string[CFloat64]("[[1+2i, 3-1i], [2i, 4], [1, -1i]]")


def test_qr_reconstructs_the_matrix() raises:
    var a = _tall()
    var factored = la.qr(a)
    _assert_matrices_close(factored[0] @ factored[1], a, "Q @ R")


def test_q_is_unitary() raises:
    """`conj_transpose(Q) @ Q` is the identity.

    Unitary is what orthogonal becomes over the complex field; `transpose(Q)
    @ Q` is not the identity here, which is the whole reason the conjugate
    transpose had to exist first.
    """
    var factored = la.qr(_tall())
    ref Q = factored[0]
    _assert_matrices_close(
        la.conj_transpose(Q) @ Q, la.eye[CFloat64](3), "Q^H @ Q"
    )


def test_r_is_upper_triangular() raises:
    var factored = la.qr(_tall())
    ref R = factored[1]
    testing.assert_equal(R[1, 0], CFloat64.zero())
    testing.assert_equal(R[2, 0], CFloat64.zero())
    testing.assert_equal(R[2, 1], CFloat64.zero())


def test_qr_handles_a_zero_column() raises:
    """A column that is already zero has no reflector to build."""
    var m = la.from_string[CFloat64]("[[0, 1], [0, 2i], [0, 1+1i]]")
    var factored = la.qr(m)
    _assert_matrices_close(factored[0] @ factored[1], m, "Q @ R")


def test_qr_rejects_a_wide_matrix() raises:
    var m = la.from_string[CFloat64]("[[1, 2, 3], [4, 5, 6]]")
    with testing.assert_raises():
        _ = la.qr(m)


# ===----------------------------------------------------------------------===#
# Least squares
# ===----------------------------------------------------------------------===#


def test_lstsq_recovers_an_exact_solution() raises:
    """When the system is consistent, the least squares answer is the answer."""
    var a = la.from_string[CFloat64]("[[1, 0], [0, 1], [1, 1]]")
    var x_true = la.from_string[CFloat64]("[[1+1i], [2]]")
    var b = a @ x_true
    _assert_matrices_close(la.lstsq(a, b), x_true, "lstsq")


def test_the_lstsq_residual_is_orthogonal_to_the_column_space() raises:
    """`conj_transpose(A) @ (Ax - b)` is zero, which defines the solution.

    Orthogonality here is with respect to the conjugate inner product. Under
    the plain transpose this quantity is not zero, and the answer would not
    minimise anything.
    """
    var a = la.from_string[CFloat64]("[[1, 0], [0, 1], [1, 1]]")
    var b = la.from_string[CFloat64]("[[1+1i], [2], [0]]")
    var x = la.lstsq(a, b)
    var residual = la.conj_transpose(a) @ ((a @ x) - b)
    for i in range(residual.nrows()):
        _assert_close(residual[i, 0], CFloat64.zero(), "A^H (Ax - b)", 1e-10)


def test_lstsq_handles_several_right_hand_sides() raises:
    var a = la.from_string[CFloat64]("[[1, 0], [0, 1], [1, 1]]")
    var x_true = la.from_string[CFloat64]("[[1+1i, 2i], [2, 1-1i]]")
    var b = a @ x_true
    _assert_matrices_close(la.lstsq(a, b), x_true, "lstsq, 2 rhs")


def test_lstsq_rejects_mismatched_shapes() raises:
    var a = la.from_string[CFloat64]("[[1, 0], [0, 1], [1, 1]]")
    var b = la.from_string[CFloat64]("[[1], [2]]")
    with testing.assert_raises():
        _ = la.lstsq(a, b)


# ===----------------------------------------------------------------------===#
# The ordered element types are unchanged
# ===----------------------------------------------------------------------===#
# Pivoting moved from a `Comparable` bound to a parameter, so these check that
# the element types that always had an order still reach the same answers
# through the new path.


def test_a_decimal_determinant_is_still_exact() raises:
    """`det([[4, 7], [2, 6]]) == 10`, exactly, in decimal."""
    var m = la.from_string[Dec128]("[[4, 7], [2, 6]]")
    testing.assert_equal(la.det(m), Dec128(10))


def test_a_decimal_inverse_is_still_exact() raises:
    """The inverse of `[[4, 7], [2, 6]]` terminates in decimal."""
    var m = la.from_string[Dec128]("[[4, 7], [2, 6]]")
    var inverse = la.inv(m)
    testing.assert_equal(inverse[0, 0], Dec128.from_string("0.6"))
    testing.assert_equal(inverse[0, 1], Dec128.from_string("-0.7"))
    testing.assert_equal(inverse[1, 0], Dec128.from_string("-0.2"))
    testing.assert_equal(inverse[1, 1], Dec128.from_string("0.4"))


def test_decimal_pivoting_still_picks_the_largest_magnitude() raises:
    """A negative entry of larger magnitude still wins the pivot."""
    var m = la.from_string[Dec128]("[[1, 2], [-9, 3]]")
    var factored = la.lu(m)
    ref piv = factored[2]
    testing.assert_equal(piv[0], 1, "|-9| > |1|")


def main() raises:
    testing.TestSuite.discover_tests[__functions_in_module()]().run()
