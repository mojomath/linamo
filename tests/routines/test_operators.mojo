"""
Tests for the Phase 5.2 operator surface on `Matrix`.

- in-place operators (`+=`, `-=`, `*=`, `/=`, `//=`, `%=`)
- `__mul__` and `__pow__`, both of which are linear-algebra operations
- the element-wise `mul`, `div` and `pow` methods that stand beside them
- `__floordiv__`, `__mod__`
- reflected scalar operators (`__radd__`, `__rsub__`, `__rmul__`,
  `__rtruediv__`)
- comparison operators returning `Matrix[Scalar[DType.bool]]`

Each group is exercised against a matrix, a view, and a scalar right-hand
side where the operator accepts one, and the non-contiguous (strided) path is
covered separately from the C-contiguous SIMD fast path.
"""

import std.testing as testing
import linamo as la
from linamo.types.matrix import Matrix


# ===----------------------------------------------------------------------===#
# In-place operators
# ===----------------------------------------------------------------------===#


def test_iadd_matrix() raises:
    """`+=` with another matrix."""
    var a = la.matrix[Float64]([[1.0, 2.0], [3.0, 4.0]])
    var b = la.matrix[Float64]([[10.0, 20.0], [30.0, 40.0]])
    a += b
    testing.assert_equal(a[0, 0], 11.0)
    testing.assert_equal(a[0, 1], 22.0)
    testing.assert_equal(a[1, 0], 33.0)
    testing.assert_equal(a[1, 1], 44.0)


def test_isub_matrix() raises:
    """`-=` with another matrix."""
    var a = la.matrix[Float64]([[10.0, 20.0], [30.0, 40.0]])
    var b = la.matrix[Float64]([[1.0, 2.0], [3.0, 4.0]])
    a -= b
    testing.assert_equal(a[0, 0], 9.0)
    testing.assert_equal(a[1, 1], 36.0)


def test_itruediv_scalar() raises:
    """`/=` with a scalar. A matrix on the right does not compile: `*=` and
    `/=` would have to mean a product that cannot be written in place."""
    var a = la.matrix[Float64]([[10.0, 20.0], [30.0, 40.0]])
    a /= 5.0
    testing.assert_equal(a[0, 0], 2.0)
    testing.assert_equal(a[1, 1], 8.0)


def test_ifloordiv_matrix() raises:
    """`//=` with another matrix."""
    var a = la.matrix[Float64]([[7.0, 9.0], [11.0, 13.0]])
    var b = la.matrix[Float64]([[2.0, 2.0], [3.0, 5.0]])
    a //= b
    testing.assert_equal(a[0, 0], 3.0)
    testing.assert_equal(a[0, 1], 4.0)
    testing.assert_equal(a[1, 0], 3.0)
    testing.assert_equal(a[1, 1], 2.0)


def test_imod_matrix() raises:
    """`%=` with another matrix."""
    var a = la.matrix[Float64]([[7.0, 9.0], [11.0, 13.0]])
    var b = la.matrix[Float64]([[2.0, 2.0], [3.0, 5.0]])
    a %= b
    testing.assert_equal(a[0, 0], 1.0)
    testing.assert_equal(a[0, 1], 1.0)
    testing.assert_equal(a[1, 0], 2.0)
    testing.assert_equal(a[1, 1], 3.0)


def test_iadd_scalar() raises:
    """`+=` with a scalar."""
    var a = la.matrix[Float64]([[1.0, 2.0], [3.0, 4.0]])
    a += 10.0
    testing.assert_equal(a[0, 0], 11.0)
    testing.assert_equal(a[1, 1], 14.0)


def test_imul_scalar() raises:
    """`*=` with a scalar."""
    var a = la.matrix[Float64]([[1.0, 2.0], [3.0, 4.0]])
    a *= 3.0
    testing.assert_equal(a[0, 0], 3.0)
    testing.assert_equal(a[1, 1], 12.0)


def test_iadd_view() raises:
    """`+=` with a matrix view."""
    var a = la.matrix[Float64]([[1.0, 2.0], [3.0, 4.0]])
    var big = la.matrix[Float64](
        [[10.0, 20.0, 99.0], [30.0, 40.0, 99.0], [99.0, 99.0, 99.0]]
    )
    a += big[0:2, 0:2]
    testing.assert_equal(a[0, 0], 11.0)
    testing.assert_equal(a[1, 1], 44.0)


def test_iadd_preserves_shape_mismatch_raises() raises:
    """`+=` on mismatched shapes raises rather than truncating."""
    var a = la.matrix[Float64]([[1.0, 2.0], [3.0, 4.0]])
    var b = la.matrix[Float64]([[1.0, 2.0, 3.0]])
    var raised = False
    try:
        a += b
    except:
        raised = True
    testing.assert_true(raised, "expected a shape-mismatch error")


def test_inplace_on_strided_matrix() raises:
    """In-place ops write through the target's own strides.

    A column-major matrix is not C-contiguous, so this exercises the
    stride-aware fallback rather than the SIMD fast path, and the layout must
    survive the operation.
    """
    # Column-major 2x2: row_stride 1, col_stride 2.
    var a = Matrix[Float64](2, 2, 1, 2)
    a[0, 0] = 1.0
    a[0, 1] = 2.0
    a[1, 0] = 3.0
    a[1, 1] = 4.0
    testing.assert_true(a.is_f_contiguous(), "expected column-major layout")

    a += 10.0
    testing.assert_equal(a[0, 0], 11.0)
    testing.assert_equal(a[0, 1], 12.0)
    testing.assert_equal(a[1, 0], 13.0)
    testing.assert_equal(a[1, 1], 14.0)
    testing.assert_true(
        a.is_f_contiguous(), "in-place op must not change the layout"
    )
    testing.assert_equal(a.row_stride(), 1)
    testing.assert_equal(a.col_stride(), 2)


def test_inplace_matrix_operand_on_strided_target() raises:
    """`+=` with a matrix operand when the target is not C-contiguous.

    Covers the stride-aware branch of the binary in-place core, which is a
    different path from the scalar one above.
    """
    var a = Matrix[Float64](2, 2, 1, 2)  # column-major
    a[0, 0] = 1.0
    a[0, 1] = 2.0
    a[1, 0] = 3.0
    a[1, 1] = 4.0
    var b = la.matrix[Float64]([[10.0, 20.0], [30.0, 40.0]])
    a += b
    testing.assert_equal(a[0, 0], 11.0)
    testing.assert_equal(a[0, 1], 22.0)
    testing.assert_equal(a[1, 0], 33.0)
    testing.assert_equal(a[1, 1], 44.0)
    testing.assert_equal(a.col_stride(), 2)


# ===----------------------------------------------------------------------===#
# floordiv, mod, pow
# ===----------------------------------------------------------------------===#


def test_floordiv_matrix() raises:
    """`//` between two matrices."""
    var a = la.matrix[Float64]([[7.0, 9.0], [11.0, 13.0]])
    var b = la.matrix[Float64]([[2.0, 2.0], [3.0, 5.0]])
    var c = a // b
    testing.assert_equal(c[0, 0], 3.0)
    testing.assert_equal(c[1, 1], 2.0)


def test_mod_matrix() raises:
    """`%` between two matrices."""
    var a = la.matrix[Float64]([[7.0, 9.0], [11.0, 13.0]])
    var b = la.matrix[Float64]([[2.0, 2.0], [3.0, 5.0]])
    var c = a % b
    testing.assert_equal(c[0, 0], 1.0)
    testing.assert_equal(c[1, 1], 3.0)


def test_mul_is_matmul() raises:
    """`*` is the matrix product, the same operation as `@`."""
    var a = la.matrix[Float64]([[1.0, 2.0], [3.0, 4.0]])
    var b = la.matrix[Float64]([[5.0, 6.0], [7.0, 8.0]])
    var c = a * b
    testing.assert_equal(c[0, 0], 19.0)
    testing.assert_equal(c[0, 1], 22.0)
    testing.assert_equal(c[1, 0], 43.0)
    testing.assert_equal(c[1, 1], 50.0)
    var d = a @ b
    testing.assert_equal(c[0, 0], d[0, 0])
    testing.assert_equal(c[1, 1], d[1, 1])


def test_mul_respects_the_order_of_the_operands() raises:
    """The matrix product does not commute, so `a * b` is not `b * a`.

    This is the one behaviour that separates the new `*` from the old one and
    would go unnoticed on symmetric operands, so it is asserted directly.
    """
    var a = la.matrix[Float64]([[1.0, 2.0], [3.0, 4.0]])
    var b = la.matrix[Float64]([[5.0, 6.0], [7.0, 8.0]])
    testing.assert_equal((a * b)[0, 0], 19.0)
    testing.assert_equal((b * a)[0, 0], 23.0)


def test_mul_rejects_a_mismatched_inner_dimension() raises:
    """`*` checks what `@` checks: the inner dimensions must agree."""
    var a = la.matrix[Float64]([[1.0, 2.0, 3.0]])
    var b = la.matrix[Float64]([[1.0, 2.0, 3.0]])
    with testing.assert_raises():
        _ = a * b


def test_elementwise_mul_and_div_methods() raises:
    """`mul` and `div` are the element-wise pair `*` and `/` used to be."""
    var a = la.matrix[Float64]([[1.0, 2.0], [3.0, 4.0]])
    var b = la.matrix[Float64]([[2.0, 3.0], [4.0, 5.0]])
    var c = a.mul(b)
    testing.assert_equal(c[0, 0], 2.0)
    testing.assert_equal(c[1, 1], 20.0)
    var d = b.div(a)
    testing.assert_equal(d[0, 0], 2.0)
    testing.assert_equal(d[1, 1], 1.25)


def test_elementwise_methods_take_a_scalar_too() raises:
    """`a.mul(2.0)` is `a * 2.0`; the scalar operand never changed meaning."""
    var a = la.matrix[Float64]([[1.0, 2.0], [3.0, 4.0]])
    testing.assert_equal(a.mul(2.0)[1, 1], 8.0)
    testing.assert_equal(a.div(2.0)[1, 1], 2.0)
    testing.assert_equal(a.pow(2.0)[1, 1], 16.0)


def test_pow_elementwise_method() raises:
    """`pow` raises each element to the matching element of the operand."""
    var a = la.matrix[Float64]([[2.0, 3.0], [4.0, 5.0]])
    var b = la.matrix[Float64]([[2.0, 2.0], [3.0, 0.0]])
    var c = a.pow(b)
    testing.assert_equal(c[0, 0], 4.0)
    testing.assert_equal(c[0, 1], 9.0)
    testing.assert_equal(c[1, 0], 64.0)
    testing.assert_equal(c[1, 1], 1.0)


def test_pow_is_repeated_multiplication() raises:
    """`a ** 3` is `a @ a @ a`, not each element cubed."""
    var a = la.matrix[Float64]([[1.0, 2.0], [3.0, 4.0]])
    var c = a**3
    var expected = a @ a @ a
    testing.assert_equal(c[0, 0], expected[0, 0])
    testing.assert_equal(c[0, 1], expected[0, 1])
    testing.assert_equal(c[1, 0], expected[1, 0])
    testing.assert_equal(c[1, 1], expected[1, 1])
    testing.assert_equal(c[0, 0], 37.0)


def test_pow_one_returns_the_matrix() raises:
    """The base case of the squaring loop."""
    var a = la.matrix[Float64]([[1.0, 2.0], [3.0, 4.0]])
    var c = a**1
    testing.assert_equal(c[0, 0], 1.0)
    testing.assert_equal(c[1, 1], 4.0)


def test_pow_zero_is_the_identity() raises:
    """`a ** 0` is the identity of the product, as for any other power."""
    var a = la.matrix[Float64]([[1.0, 2.0], [3.0, 4.0]])
    var c = a**0
    testing.assert_equal(c[0, 0], 1.0)
    testing.assert_equal(c[0, 1], 0.0)
    testing.assert_equal(c[1, 0], 0.0)
    testing.assert_equal(c[1, 1], 1.0)


def test_pow_minus_one_is_the_inverse() raises:
    """A negative exponent inverts first, so `a ** -1 @ a` is the identity."""
    var a = la.matrix[Float64]([[4.0, 7.0], [2.0, 6.0]])
    var c = a**-1
    testing.assert_almost_equal(c[0, 0], 0.6)
    testing.assert_almost_equal(c[0, 1], -0.7)
    testing.assert_almost_equal(c[1, 0], -0.2)
    testing.assert_almost_equal(c[1, 1], 0.4)
    var back = c @ a
    testing.assert_almost_equal(back[0, 0], 1.0)
    testing.assert_almost_equal(back[0, 1], 0.0)


def test_pow_rejects_a_non_square_matrix() raises:
    """A rectangular matrix cannot be multiplied by itself."""
    var a = la.matrix[Float64]([[1.0, 2.0, 3.0]])
    with testing.assert_raises():
        _ = a**2


def test_floordiv_view() raises:
    """`//` with a matrix view on the right."""
    var a = la.matrix[Float64]([[7.0, 9.0], [11.0, 13.0]])
    var big = la.matrix[Float64](
        [[2.0, 2.0, 99.0], [3.0, 5.0, 99.0], [99.0, 99.0, 99.0]]
    )
    var c = a // big[0:2, 0:2]
    testing.assert_equal(c[0, 0], 3.0)
    testing.assert_equal(c[1, 1], 2.0)


def test_mod_scalar() raises:
    """`%` with a scalar."""
    var a = la.matrix[Float64]([[7.0, 9.0], [11.0, 13.0]])
    var c = a % 5.0
    testing.assert_equal(c[0, 0], 2.0)
    testing.assert_equal(c[1, 1], 3.0)


# ===----------------------------------------------------------------------===#
# Reflected scalar operators
# ===----------------------------------------------------------------------===#


def test_radd() raises:
    """`scalar + matrix`."""
    var a = la.matrix[Float64]([[1.0, 2.0], [3.0, 4.0]])
    var c = 10.0 + a
    testing.assert_equal(c[0, 0], 11.0)
    testing.assert_equal(c[1, 1], 14.0)


def test_rmul() raises:
    """`scalar * matrix`."""
    var a = la.matrix[Float64]([[1.0, 2.0], [3.0, 4.0]])
    var c = 3.0 * a
    testing.assert_equal(c[0, 0], 3.0)
    testing.assert_equal(c[1, 1], 12.0)


def test_rsub_is_not_commutative() raises:
    """`scalar - matrix` subtracts the matrix, not the other way round."""
    var a = la.matrix[Float64]([[1.0, 2.0], [3.0, 4.0]])
    var c = 10.0 - a
    testing.assert_equal(c[0, 0], 9.0)
    testing.assert_equal(c[0, 1], 8.0)
    testing.assert_equal(c[1, 0], 7.0)
    testing.assert_equal(c[1, 1], 6.0)


def test_rtruediv_is_not_commutative() raises:
    """`scalar / matrix` divides the scalar by each element."""
    var a = la.matrix[Float64]([[1.0, 2.0], [4.0, 5.0]])
    var c = 20.0 / a
    testing.assert_equal(c[0, 0], 20.0)
    testing.assert_equal(c[0, 1], 10.0)
    testing.assert_equal(c[1, 0], 5.0)
    testing.assert_equal(c[1, 1], 4.0)


# ===----------------------------------------------------------------------===#
# Comparison operators
# ===----------------------------------------------------------------------===#


def test_greater_matrix() raises:
    """`>` returns an element-wise bool mask."""
    var a = la.matrix[Float64]([[1.0, 5.0], [3.0, 4.0]])
    var b = la.matrix[Float64]([[2.0, 2.0], [3.0, 9.0]])
    var m = a > b
    testing.assert_equal(m[0, 0], False)
    testing.assert_equal(m[0, 1], True)
    testing.assert_equal(m[1, 0], False)
    testing.assert_equal(m[1, 1], False)


def test_greater_equal_matrix() raises:
    """`>=` includes the equal case."""
    var a = la.matrix[Float64]([[1.0, 5.0], [3.0, 4.0]])
    var b = la.matrix[Float64]([[2.0, 2.0], [3.0, 9.0]])
    var m = a >= b
    testing.assert_equal(m[0, 0], False)
    testing.assert_equal(m[1, 0], True)


def test_less_and_less_equal() raises:
    """`<` and `<=`."""
    var a = la.matrix[Float64]([[1.0, 5.0], [3.0, 4.0]])
    var b = la.matrix[Float64]([[2.0, 2.0], [3.0, 9.0]])
    var lt = a < b
    testing.assert_equal(lt[0, 0], True)
    testing.assert_equal(lt[1, 0], False)
    var le = a <= b
    testing.assert_equal(le[1, 0], True)


def test_equal_returns_mask_not_bool() raises:
    """`==` is element-wise, matching NumPy rather than returning one Bool."""
    var a = la.matrix[Float64]([[1.0, 2.0], [3.0, 4.0]])
    var b = la.matrix[Float64]([[1.0, 9.0], [3.0, 9.0]])
    var m = a == b
    testing.assert_equal(m[0, 0], True)
    testing.assert_equal(m[0, 1], False)
    testing.assert_equal(m[1, 0], True)
    testing.assert_equal(m[1, 1], False)
    # The mask has the same shape as the operands.
    testing.assert_equal(m.nrows(), 2)
    testing.assert_equal(m.ncols(), 2)


def test_not_equal() raises:
    """`!=` is the complement of `==`."""
    var a = la.matrix[Float64]([[1.0, 2.0], [3.0, 4.0]])
    var b = la.matrix[Float64]([[1.0, 9.0], [3.0, 9.0]])
    var m = a != b
    testing.assert_equal(m[0, 0], False)
    testing.assert_equal(m[0, 1], True)


def test_compare_scalar() raises:
    """Comparison against a scalar right-hand side."""
    var a = la.matrix[Float64]([[1.0, 2.0], [3.0, 4.0]])
    var m = a > 2.0
    testing.assert_equal(m[0, 0], False)
    testing.assert_equal(m[0, 1], False)
    testing.assert_equal(m[1, 0], True)
    testing.assert_equal(m[1, 1], True)


def test_compare_view() raises:
    """Comparison against a matrix view."""
    var a = la.matrix[Float64]([[1.0, 5.0], [3.0, 4.0]])
    var big = la.matrix[Float64](
        [[2.0, 2.0, 99.0], [3.0, 9.0, 99.0], [99.0, 99.0, 99.0]]
    )
    var m = a > big[0:2, 0:2]
    testing.assert_equal(m[0, 1], True)
    testing.assert_equal(m[1, 1], False)


def test_compare_strided_operand() raises:
    """Comparison falls back to the stride-aware path for non-contiguous data.
    """
    var a = Matrix[Float64](2, 2, 1, 2)  # column-major
    a[0, 0] = 1.0
    a[0, 1] = 5.0
    a[1, 0] = 3.0
    a[1, 1] = 4.0
    testing.assert_true(not a.is_c_contiguous(), "expected non-C-contiguous")
    var m = a > 2.0
    testing.assert_equal(m[0, 0], False)
    testing.assert_equal(m[0, 1], True)
    testing.assert_equal(m[1, 0], True)
    testing.assert_equal(m[1, 1], True)


def test_compare_shape_mismatch_raises() raises:
    """Comparing mismatched shapes raises."""
    var a = la.matrix[Float64]([[1.0, 2.0], [3.0, 4.0]])
    var b = la.matrix[Float64]([[1.0, 2.0, 3.0]])
    var raised = False
    try:
        var _m = a > b
    except:
        raised = True
    testing.assert_true(raised, "expected a shape-mismatch error")


# ===----------------------------------------------------------------------===#
# MatrixView operator parity
# ===----------------------------------------------------------------------===#
# Views get every operator except the in-place ones, which cannot be defined
# on a type generic over `origin`.


def test_view_scalar_arithmetic() raises:
    """Scalar right-hand side on a view."""
    var big = la.matrix[Float64](
        [[1.0, 2.0, 99.0], [3.0, 4.0, 99.0], [99.0, 99.0, 99.0]]
    )
    var v = big[0:2, 0:2]
    var c = v * 10.0
    testing.assert_equal(c[0, 0], 10.0)
    testing.assert_equal(c[1, 1], 40.0)


def test_view_floordiv_and_pow() raises:
    """`//` and `**` on a view, plus the element-wise `pow` beside them."""
    var big = la.matrix[Float64](
        [[7.0, 9.0, 99.0], [11.0, 13.0, 99.0], [99.0, 99.0, 99.0]]
    )
    var v = big[0:2, 0:2]
    var fd = v // 2.0
    testing.assert_equal(fd[0, 0], 3.0)
    testing.assert_equal(fd[1, 1], 6.0)
    var pw = v.pow(2.0)
    testing.assert_equal(pw[0, 0], 49.0)
    var sq = v**2
    testing.assert_equal(sq[0, 0], 7.0 * 7.0 + 9.0 * 11.0)


def test_view_reflected_scalar() raises:
    """`scalar - view` on a view."""
    var big = la.matrix[Float64](
        [[1.0, 2.0, 99.0], [3.0, 4.0, 99.0], [99.0, 99.0, 99.0]]
    )
    var v = big[0:2, 0:2]
    var c = 10.0 - v
    testing.assert_equal(c[0, 0], 9.0)
    testing.assert_equal(c[1, 1], 6.0)


def test_view_comparison() raises:
    """Comparison operators on a view, including the strided path."""
    var big = la.matrix[Float64](
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
    )
    # Strided view: every other column.
    var v = big[0:3:2, 0:3:2]
    testing.assert_equal(v.nrows(), 2)
    testing.assert_equal(v.ncols(), 2)
    var m = v > 3.0
    testing.assert_equal(m[0, 0], False)  # 1.0
    testing.assert_equal(m[0, 1], False)  # 3.0
    testing.assert_equal(m[1, 0], True)  # 7.0
    testing.assert_equal(m[1, 1], True)  # 9.0


def test_view_equality_mask_against_matrix() raises:
    """`view == matrix` returns a mask."""
    var big = la.matrix[Float64](
        [[1.0, 2.0, 99.0], [3.0, 4.0, 99.0], [99.0, 99.0, 99.0]]
    )
    var v = big[0:2, 0:2]
    var b = la.matrix[Float64]([[1.0, 9.0], [3.0, 9.0]])
    var m = v == b
    testing.assert_equal(m[0, 0], True)
    testing.assert_equal(m[0, 1], False)
    testing.assert_equal(m[1, 0], True)


def main() raises:
    testing.TestSuite.discover_tests[__functions_in_module()]().run()
