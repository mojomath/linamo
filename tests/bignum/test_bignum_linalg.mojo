"""
Tests for elimination over decimo's number types.

`lu`, `det`, `solve` and `inv` each carry a `Numeric` overload beside the
scalar one, so a `Matrix[Dec128]` or `Matrix[BDec]` decomposes and inverts
through the same names a `Matrix[Float64]` uses. What the element type buys is
the arithmetic underneath: a decimal quotient rounded to the type's precision
rather than to 53 bits.

`BigInt` reaches the same routines --- nothing in the bound excludes it --- but
its `/` truncates toward zero and an integer matrix has no integer inverse in
general, so the answers are only meaningful in the special cases where no
truncation happens. The last section pins both sides of that down.
"""

import std.testing as testing
import linamo as la
from linamo import BDec, BInt, Dec128


def _a() raises -> la.Matrix[Dec128]:
    """Returns `[[4, 7], [2, 6]]`, whose inverse is exact in decimal."""
    return la.from_string[Dec128]("[[4, 7], [2, 6]]")


# ===----------------------------------------------------------------------===#
# LU and determinant
# ===----------------------------------------------------------------------===#


def test_lu_reconstructs_the_matrix() raises:
    """L @ U equals the rows of A taken in the order `piv` gives."""
    var a = _a()
    var factored = la.lu(a)
    ref L = factored[0]
    ref U = factored[1]
    ref piv = factored[2]
    var product = L @ U
    for i in range(2):
        for j in range(2):
            testing.assert_equal(product[i, j], a[piv[i], j])


def test_lu_is_unit_lower_triangular() raises:
    """L has ones on its diagonal and zeros above it."""
    var factored = la.lu(_a())
    ref L = factored[0]
    testing.assert_equal(L[0, 0], Dec128("1"))
    testing.assert_equal(L[1, 1], Dec128("1"))
    testing.assert_equal(L[0, 1], Dec128("0"))


def test_det_is_exact_in_decimal() raises:
    """`det([[4, 7], [2, 6]])` is 10, sign of the pivoting included."""
    testing.assert_equal(la.det(_a()), Dec128("10"))


def test_det_of_a_singular_matrix_is_zero() raises:
    """A column with no non-zero pivot leaves a zero on U's diagonal."""
    var a = la.from_string[Dec128]("[[1, 2], [2, 4]]")
    testing.assert_equal(la.det(a), Dec128("0"))


# ===----------------------------------------------------------------------===#
# Solve and inverse
# ===----------------------------------------------------------------------===#


def test_solve_a_single_right_hand_side() raises:
    """Ax = b for a column vector b."""
    var a = _a()
    var b = la.from_string[Dec128]("[[1], [1]]")
    var x = la.solve(a, b)
    testing.assert_equal(x.nrows(), 2)
    testing.assert_equal(x.ncols(), 1)
    # 4x + 7y = 1, 2x + 6y = 1 gives x = -0.1, y = 0.2.
    testing.assert_equal(x[0, 0], Dec128("-0.1"))
    testing.assert_equal(x[1, 0], Dec128("0.2"))


def test_inv_is_exact_in_decimal() raises:
    """The inverse of `[[4, 7], [2, 6]]` lands on terminating decimals."""
    var x = la.inv(_a())
    testing.assert_equal(x[0, 0], Dec128("0.6"))
    testing.assert_equal(x[0, 1], Dec128("-0.7"))
    testing.assert_equal(x[1, 0], Dec128("-0.2"))
    testing.assert_equal(x[1, 1], Dec128("0.4"))


def test_inv_times_the_matrix_is_the_identity() raises:
    """The round trip closes exactly, which a binary float need not do."""
    var a = _a()
    var product = la.inv(a) @ a
    testing.assert_equal(product[0, 0], Dec128("1"))
    testing.assert_equal(product[0, 1], Dec128("0"))
    testing.assert_equal(product[1, 0], Dec128("0"))
    testing.assert_equal(product[1, 1], Dec128("1"))


def test_inv_of_a_singular_matrix_raises() raises:
    """No pivot means no inverse, and the message says which matrix."""
    var a = la.from_string[Dec128]("[[1, 2], [2, 4]]")
    var raised = False
    try:
        _ = la.inv(a)
    except e:
        raised = True
        testing.assert_true("singular" in String(e), "the message says why")
    testing.assert_true(raised, "inv of a singular matrix")


def test_solve_rejects_a_mismatched_right_hand_side() raises:
    """The right-hand side must have as many rows as A."""
    var a = _a()
    var b = la.from_string[Dec128]("[[1], [1], [1]]")
    var raised = False
    try:
        _ = la.solve(a, b)
    except:
        raised = True
    testing.assert_true(raised, "solve with the wrong number of rows")


def test_the_routines_take_views() raises:
    """A slice of a larger matrix inverts without being copied out first."""
    var big = la.from_string[Dec128]("[[4, 7, 0], [2, 6, 0], [0, 0, 9]]")
    var x = la.inv(big[0:2, 0:2])
    testing.assert_equal(x[0, 0], Dec128("0.6"))


def test_bigdecimal_inverts_too() raises:
    """`BDec` is arbitrary-precision, and reaches the same overloads."""
    var a = la.from_string[BDec]("[[4, 7], [2, 6]]")
    var x = la.inv(a)
    testing.assert_equal(x[0, 0], BDec("0.6"))
    testing.assert_equal(x[1, 1], BDec("0.4"))


# ===----------------------------------------------------------------------===#
# Negative matrix powers
# ===----------------------------------------------------------------------===#


def test_pow_minus_one_is_the_inverse() raises:
    """`A ** -1` is `inv(A)` for an arbitrary-precision element, as for SIMD."""
    var a = _a()
    var x = a**-1
    testing.assert_equal(x[0, 0], Dec128("0.6"))
    testing.assert_equal(x[1, 1], Dec128("0.4"))


def test_pow_minus_two_inverts_then_squares() raises:
    """`A ** -2` is `inv(A) @ inv(A)`."""
    var a = _a()
    var x = a**-2
    var expected = la.inv(a) @ la.inv(a)
    testing.assert_equal(x[0, 0], expected[0, 0])
    testing.assert_equal(x[1, 1], expected[1, 1])


def test_pow_minus_one_of_a_singular_matrix_raises() raises:
    """The inverse is reached through `inv`, so it raises where `inv` does."""
    var a = la.from_string[Dec128]("[[1, 2], [2, 4]]")
    var raised = False
    try:
        _ = a**-1
    except:
        raised = True
    testing.assert_true(raised, "a singular matrix to a negative power")


# ===----------------------------------------------------------------------===#
# `BigInt` reaches these routines, and mostly should not
# ===----------------------------------------------------------------------===#


def test_bigint_inverts_a_unimodular_matrix_correctly() raises:
    """Where the elimination never truncates, the integer answer is right."""
    var a = la.matrix[BInt]([[1, 1], [0, 1]])
    var x = la.inv(a)
    testing.assert_equal(String(x[0, 0]), "1")
    testing.assert_equal(String(x[0, 1]), "-1")
    testing.assert_equal(String(x[1, 0]), "0")
    testing.assert_equal(String(x[1, 1]), "1")


def test_bigint_truncation_makes_the_general_answer_wrong() raises:
    """`BigInt` division truncates, so elimination over it is not linear algebra.

    `[[1, 2], [3, 4]]` has determinant -2. Partial pivoting divides 1 by 3,
    gets 0 rather than a third, and the factorisation that follows describes a
    different matrix. Nothing raises; the number is simply not the answer. A
    decimal element type is what these routines are for.
    """
    var a = la.matrix[BInt]([[1, 2], [3, 4]])
    testing.assert_true(
        la.det(a) != BInt(-2), "truncated elimination misses the determinant"
    )


def main() raises:
    testing.TestSuite.discover_tests[__functions_in_module()]().run()
