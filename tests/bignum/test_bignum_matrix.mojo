"""
Tests for matrices whose elements are decimo's numbers.

`BigInt`, `BigDecimal` and `Decimal128` conform to `decimo.Numeric`, so a
matrix of them is an ordinary `Matrix` with the ordinary operators. What
separates them from `Float64` is only where the arithmetic comes from: a
`Numeric` element carries its own, and the scalar SIMD kernels are bypassed.
The tests below are therefore written the way a user would write the code ---
`a + b`, `a @ b`, `la.eye[BInt](3)` --- and pass through the `Numeric`
overloads without naming them.
"""

import std.testing as testing
import linamo as la
from linamo import BInt, Dec128


def _a() raises -> la.Matrix[BInt]:
    """Returns the 2x2 matrix `[[1, 2], [3, 4]]`."""
    return la.matrix[BInt]([[1, 2], [3, 4]])


# ===----------------------------------------------------------------------===#
# Storage and element access
# ===----------------------------------------------------------------------===#


def test_matrix_holds_bigint() raises:
    """A `Matrix` stores a heap-owning element like any other."""
    var a = _a()
    testing.assert_equal(a.nrows(), 2)
    testing.assert_equal(a.ncols(), 2)
    testing.assert_equal(String(a[0, 0]), "1")
    testing.assert_equal(String(a[1, 1]), "4")


def test_elements_are_independent_after_copy() raises:
    """Copying a matrix copies its elements, not a shared buffer."""
    var a = _a()
    var b = a.copy()
    b[0, 0] = BInt(99)
    testing.assert_equal(String(a[0, 0]), "1")
    testing.assert_equal(String(b[0, 0]), "99")


def test_element_larger_than_any_dtype() raises:
    """The reason for all of this: an element no `DType` can hold."""
    var factorial = BInt.one()
    for k in range(1, 31):
        factorial = factorial * BInt(k)
    var m = la.diag([factorial.copy(), factorial.copy()])
    # 30! = 265252859812191058636308480000000, well past 2^64.
    testing.assert_equal(String(m[0, 0]), "265252859812191058636308480000000")
    testing.assert_equal(
        String(la.trace(m)), "530505719624382117272616960000000"
    )


# ===----------------------------------------------------------------------===#
# Routines that only move or compare elements
# ===----------------------------------------------------------------------===#
# These were generic over the element type before `Numeric` existed and stay
# that way: nothing here adds or multiplies anything.


def test_transpose_is_generic() raises:
    """`transpose` only moves elements, as a routine and as a method."""
    var t = la.transpose(_a())
    testing.assert_equal(String(t[0, 1]), "3")
    testing.assert_equal(String(t[1, 0]), "2")
    var u = _a().transpose()
    testing.assert_equal(String(u[0, 1]), "3")


def test_slicing_yields_a_view() raises:
    """Slicing a `BInt` matrix borrows rather than copies."""
    var a = _a()
    var row = a[1:2, :]
    testing.assert_equal(row.nrows(), 1)
    testing.assert_equal(row.ncols(), 2)
    testing.assert_equal(String(row[0, 0]), "3")


def test_reshape_and_flatten_are_generic() raises:
    """Reshaping preserves the elements under a new shape."""
    var r = la.reshape(_a(), 4, 1)
    testing.assert_equal(r.nrows(), 4)
    testing.assert_equal(String(r[2, 0]), "3")
    var f = la.flatten(_a())
    testing.assert_equal(f.ncols(), 4)
    testing.assert_equal(String(f[0, 3]), "4")


def test_contiguous_f_order_is_generic() raises:
    """A layout change moves elements without touching their values."""
    var f = la.contiguous(_a(), "F")
    testing.assert_true(f.is_f_contiguous())
    testing.assert_equal(String(f[0, 1]), "2")
    testing.assert_equal(String(f[1, 0]), "3")


def test_sort_rides_the_stdlib_comparable() raises:
    """`sort` asks only for `Comparable`, which `BInt` already declares."""
    var m = la.matrix[BInt]([[30, 10, 20]])
    var s = la.sort(m, 1)
    testing.assert_equal(String(s[0, 0]), "10")
    testing.assert_equal(String(s[0, 1]), "20")
    testing.assert_equal(String(s[0, 2]), "30")


def test_argsort_and_argmax_are_generic() raises:
    """The index routines compare elements and return `Int64` positions."""
    var m = la.matrix[BInt]([[30, 10, 20]])
    var order = la.argsort(m, 1)
    testing.assert_equal(order[0, 0], 1)
    testing.assert_equal(order[0, 2], 0)
    testing.assert_equal(la.argmax(m), 0)
    testing.assert_equal(la.argmin(m), 1)


# ===----------------------------------------------------------------------===#
# Arithmetic through the operators
# ===----------------------------------------------------------------------===#


def test_add_and_sub() raises:
    """Element-wise addition and subtraction."""
    var a = _a()
    var s = a + a
    testing.assert_equal(String(s[0, 0]), "2")
    testing.assert_equal(String(s[1, 1]), "8")
    var d = s - a
    testing.assert_equal(String(d[1, 1]), "4")


def test_mul_is_element_wise() raises:
    """`*` is the Hadamard product; `@` is matrix multiplication."""
    var p = _a() * _a()
    testing.assert_equal(String(p[0, 1]), "4")
    testing.assert_equal(String(p[1, 0]), "9")


def test_div_truncates_on_an_integral_element() raises:
    """`/` on a `BInt` truncates toward zero, as `Int` does."""
    var q = _a() / la.matrix[BInt]([[2, 3], [2, 3]])
    testing.assert_equal(String(q[0, 0]), "0")
    testing.assert_equal(String(q[1, 0]), "1")


def test_neg_and_scalar_operands() raises:
    """Negation, and a value on either side of the operator."""
    var n = -_a()
    testing.assert_equal(String(n[0, 0]), "-1")
    testing.assert_equal(String((_a() + BInt(10))[1, 1]), "14")
    testing.assert_equal(String((_a() * BInt(3))[1, 1]), "12")
    testing.assert_equal(String((BInt(10) - _a())[0, 0]), "9")
    testing.assert_equal(String((BInt(2) * _a())[1, 1]), "8")


def test_matmul() raises:
    """`[[1, 2], [3, 4]]` squared is `[[7, 10], [15, 22]]`."""
    var p = _a() @ _a()
    testing.assert_equal(String(p[0, 0]), "7")
    testing.assert_equal(String(p[0, 1]), "10")
    testing.assert_equal(String(p[1, 0]), "15")
    testing.assert_equal(String(p[1, 1]), "22")


def test_matmul_against_identity() raises:
    """Multiplying by the identity leaves a matrix unchanged."""
    var p = _a() @ la.identity[BInt](2)
    testing.assert_equal(String(p[0, 1]), "2")
    testing.assert_equal(String(p[1, 0]), "3")


def test_matmul_shape_mismatch_raises() raises:
    """Mismatched inner dimensions are a `ValueError`, not a wrong answer."""
    var wide = la.matrix[BInt]([[1, 2, 3]])
    with testing.assert_raises():
        var _p = wide @ wide


def test_add_shape_mismatch_raises() raises:
    """Element-wise operations check their shapes."""
    var wide = la.matrix[BInt]([[1, 2, 3]])
    with testing.assert_raises():
        var _s = _a() + wide


def test_sum_and_trace() raises:
    """The whole-matrix sum and the diagonal sum."""
    testing.assert_equal(String(la.sum(_a())), "10")
    testing.assert_equal(String(la.trace(_a())), "5")


def test_trace_requires_square() raises:
    """A trace is defined for square matrices only."""
    var wide = la.matrix[BInt]([[1, 2, 3]])
    with testing.assert_raises():
        var _t = la.trace(wide)


def test_zeros_ones_and_eye() raises:
    """The creation routines ask `Numeric` for a zero and a one."""
    var z = la.zeros[BInt](2, 3)
    testing.assert_equal(z.size(), 6)
    testing.assert_equal(String(z[1, 2]), "0")
    var o = la.ones[BInt](2, 2)
    testing.assert_equal(String(o[0, 1]), "1")
    var i = la.eye[BInt](3)
    testing.assert_equal(String(i[1, 1]), "1")
    testing.assert_equal(String(i[1, 2]), "0")


def test_views_are_accepted_as_operands() raises:
    """A `MatrixView` operand works wherever a `Matrix` does."""
    var a = _a()
    var top = a[0:1, :]
    var s = top + top
    testing.assert_equal(s.nrows(), 1)
    testing.assert_equal(String(s[0, 1]), "4")
    var t = top.transpose()
    testing.assert_equal(String(t[1, 0]), "2")


# ===----------------------------------------------------------------------===#
# The other two `Numeric` element types
# ===----------------------------------------------------------------------===#


def test_decimal128_matrix() raises:
    """`Decimal128` is exact where a binary float is not: 0.1 + 0.2 == 0.3."""
    var a = la.matrix[Dec128]([[Dec128("0.1"), Dec128("0.2")]])
    var b = la.matrix[Dec128]([[Dec128("0.2"), Dec128("0.1")]])
    testing.assert_equal(String((a + b)[0, 0]), "0.3")
    testing.assert_equal(String(la.sum(a)), "0.3")


def test_bigdecimal_matrix() raises:
    """`la.Decimal` is decimo's arbitrary-precision decimal."""
    var a = la.matrix[la.Decimal]([[la.Decimal("1.5"), la.Decimal("2.5")]])
    testing.assert_equal(String((a * la.Decimal("2"))[0, 1]), "5.0")


def main() raises:
    testing.TestSuite.discover_tests[__functions_in_module()]().run()
