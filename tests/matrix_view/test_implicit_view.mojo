"""
Tests for the implicit `Matrix` -> `MatrixView` conversion.

Every named routine in the library takes `MatrixView` operands and nothing
else. A `Matrix` argument still works because `MatrixView` has an `@implicit`
constructor from `Matrix`, so the compiler inserts the conversion. That is what
replaced the four hand-written overloads (view x view, mat x mat, mat x view,
view x mat) that each binary operation used to carry. The operator dunders on
both types were collapsed the same way and are covered here too: the
conversion fires under `a + b` sugar, not only in a direct call.

Two properties have to hold for that to be safe, and both are tested here:

1. The conversion produces a *read-only* view, so passing the same matrix
   twice is two immutable borrows and compiles. Were it mutable, `add(a, a)`
   would be rejected as two exclusive borrows of one matrix.
2. The conversion therefore cannot reach anything in `routines.mutation`,
   whose signatures are pinned to `Origin[mut=True]`. A caller who never
   imports that module still cannot obtain a mutable view.

Property 2 is a compile-time guarantee, so it cannot be asserted at runtime;
`test_conversion_is_read_only` checks the `mut` parameter instead, which is
the same fact in a form a test can see.
"""

import std.testing as testing
import linamo as la
from linamo.routines.math import add, matmul, scalar_add
from linamo.routines.logic import greater


# ===----------------------------------------------------------------------===#
# One signature, four call shapes
# ===----------------------------------------------------------------------===#


def test_add_accepts_all_four_combinations() raises:
    """`add` has a single view x view signature; all four shapes call it."""
    var a = la.matrix[DType.float64]([[1.0, 2.0], [3.0, 4.0]])
    var b = la.matrix[DType.float64]([[10.0, 20.0], [30.0, 40.0]])

    var mm = add(a, b)
    var mv = add(a, b.view())
    var vm = add(a.view(), b)
    var vv = add(a.view(), b.view())

    for c in [mm^, mv^, vm^, vv^]:
        testing.assert_equal(c[0, 0], 11.0)
        testing.assert_equal(c[1, 1], 44.0)


def test_matmul_accepts_all_four_combinations() raises:
    """The same holds for the SIMD matmul entry point."""
    var a = la.matrix[DType.float64]([[1.0, 2.0], [3.0, 4.0]])
    var b = la.matrix[DType.float64]([[5.0, 6.0], [7.0, 8.0]])

    var mm = matmul(a, b)
    var mv = matmul(a, b.view())
    var vm = matmul(a.view(), b)
    var vv = matmul(a.view(), b.view())

    for c in [mm^, mv^, vm^, vv^]:
        testing.assert_equal(c[0, 0], 19.0)
        testing.assert_equal(c[1, 1], 50.0)


def test_comparison_accepts_a_matrix() raises:
    """Comparisons in `routines.logic` collapsed the same way."""
    var a = la.matrix[DType.float64]([[1.0, 5.0], [3.0, 4.0]])
    var b = la.matrix[DType.float64]([[2.0, 2.0], [3.0, 9.0]])
    var m = greater(a, b)
    testing.assert_equal(m[0, 0], False)
    testing.assert_equal(m[0, 1], True)
    testing.assert_equal(m[1, 0], False)


def test_scalar_routine_accepts_a_matrix() raises:
    """Scalar forms dropped their `Matrix` overload too."""
    var a = la.matrix[DType.float64]([[1.0, 2.0], [3.0, 4.0]])
    var c = scalar_add(a, 10.0)
    testing.assert_equal(c[0, 0], 11.0)
    testing.assert_equal(c[1, 1], 14.0)


# ===----------------------------------------------------------------------===#
# The conversion is read-only
# ===----------------------------------------------------------------------===#


def test_conversion_is_read_only() raises:
    """An implicitly built view is immutable even when the matrix is a `var`.

    This is what keeps `routines.mutation` the only door to a mutable view: a
    signature pinned to `Origin[mut=True]` cannot be satisfied by this
    conversion, so `fill(m, ...)` is a compile error rather than a silent
    write.
    """
    var a = la.matrix[DType.float64]([[1.0, 2.0], [3.0, 4.0]])
    var v = la.MatrixView(a)
    testing.assert_false(v.mut)
    testing.assert_equal(v[0, 0], 1.0)


def test_same_matrix_twice() raises:
    """Passing one matrix into both operands must compile and run.

    Two immutable borrows of the same buffer are fine. Were the conversion
    mutable, this call would be two exclusive borrows and would be rejected.
    """
    var a = la.matrix[DType.float64]([[1.0, 2.0], [3.0, 4.0]])
    var c = add(a, a)
    testing.assert_equal(c[0, 0], 2.0)
    testing.assert_equal(c[1, 1], 8.0)


def test_mixed_matrix_and_own_slice() raises:
    """A matrix and a slice of itself in one call, after the collapse."""
    var a = la.matrix[DType.float64]([[1.0, 2.0], [3.0, 4.0]])
    var c = add(a[0:2, 0:2], a)
    testing.assert_equal(c[0, 0], 2.0)
    testing.assert_equal(c[1, 1], 8.0)


# ===----------------------------------------------------------------------===#
# The same collapse, one level up: operator sugar
# ===----------------------------------------------------------------------===#
# The dunders on `Matrix` and `MatrixView` declare their right-hand operand as
# a `MatrixView` and have no `Matrix`-operand twin. These tests exist mostly to
# be compiled: if a needed overload were missing, the file would not build.


def test_arithmetic_operators_accept_all_four_combinations() raises:
    """`+ - * / @` resolve for every mix of `Matrix` and `MatrixView`."""
    var a = la.matrix[DType.float64]([[1.0, 2.0], [3.0, 4.0]])
    var b = la.matrix[DType.float64]([[10.0, 20.0], [30.0, 40.0]])

    for c in [a + b, a + b.view(), a.view() + b, a.view() + b.view()]:
        testing.assert_equal(c[0, 0], 11.0)
    for c in [a - b, a - b.view(), a.view() - b, a.view() - b.view()]:
        testing.assert_equal(c[0, 0], -9.0)
    for c in [a * b, a * b.view(), a.view() * b, a.view() * b.view()]:
        testing.assert_equal(c[0, 0], 10.0)
    for c in [b / a, b / a.view(), b.view() / a, b.view() / a.view()]:
        testing.assert_equal(c[0, 0], 10.0)
    for c in [a @ b, a @ b.view(), a.view() @ b, a.view() @ b.view()]:
        testing.assert_equal(c[0, 0], 70.0)


def test_floordiv_mod_pow_operators_accept_all_four_combinations() raises:
    """`// % **` resolve for every mix of `Matrix` and `MatrixView`."""
    var a = la.matrix[DType.float64]([[2.0, 2.0], [2.0, 2.0]])
    var b = la.matrix[DType.float64]([[7.0, 7.0], [7.0, 7.0]])

    for c in [b // a, b // a.view(), b.view() // a, b.view() // a.view()]:
        testing.assert_equal(c[0, 0], 3.0)
    for c in [b % a, b % a.view(), b.view() % a, b.view() % a.view()]:
        testing.assert_equal(c[0, 0], 1.0)
    for c in [a**a, a ** a.view(), a.view() ** a, a.view() ** a.view()]:
        testing.assert_equal(c[0, 0], 4.0)


def test_comparison_operators_accept_all_four_combinations() raises:
    """The six comparisons resolve for every mix, returning a bool mask."""
    var a = la.matrix[DType.float64]([[1.0, 2.0], [3.0, 4.0]])
    var b = la.matrix[DType.float64]([[10.0, 20.0], [30.0, 40.0]])

    for c in [a < b, a < b.view(), a.view() < b, a.view() < b.view()]:
        testing.assert_equal(c[0, 0], True)
    for c in [a <= b, a <= b.view(), a.view() <= b, a.view() <= b.view()]:
        testing.assert_equal(c[0, 0], True)
    for c in [a > b, a > b.view(), a.view() > b, a.view() > b.view()]:
        testing.assert_equal(c[0, 0], False)
    for c in [a >= b, a >= b.view(), a.view() >= b, a.view() >= b.view()]:
        testing.assert_equal(c[0, 0], False)
    for c in [a == b, a == b.view(), a.view() == b, a.view() == b.view()]:
        testing.assert_equal(c[0, 0], False)
    for c in [a != b, a != b.view(), a.view() != b, a.view() != b.view()]:
        testing.assert_equal(c[0, 0], True)


def test_inplace_operators_accept_a_matrix_and_a_view() raises:
    """The in-place operators take either operand type on the right."""
    var b = la.matrix[DType.float64]([[10.0, 20.0], [30.0, 40.0]])

    var c = la.matrix[DType.float64]([[1.0, 2.0], [3.0, 4.0]])
    c += b
    c += b.view()
    testing.assert_equal(c[0, 0], 21.0)

    c -= b
    c -= b.view()
    testing.assert_equal(c[0, 0], 1.0)

    c *= b
    c /= b.view()
    testing.assert_equal(c[0, 0], 1.0)

    var d = la.matrix[DType.float64]([[7.0, 7.0], [7.0, 7.0]])
    var two = la.matrix[DType.float64]([[2.0, 2.0], [2.0, 2.0]])
    d //= two.view()
    testing.assert_equal(d[0, 0], 3.0)
    d %= two
    testing.assert_equal(d[0, 0], 1.0)


def test_scalar_operators_are_not_shadowed() raises:
    """A scalar does not convert to a matrix, so it keeps its own overloads.

    Collapsing the matrix-operand forms must not disturb these, nor the
    reflected forms that put the scalar on the left.
    """
    var a = la.matrix[DType.float64]([[1.0, 2.0], [3.0, 4.0]])

    testing.assert_equal((a + 1.0)[0, 0], 2.0)
    testing.assert_equal((a.view() + 1.0)[0, 0], 2.0)
    testing.assert_equal((1.0 + a)[0, 0], 2.0)
    testing.assert_equal((1.0 - a)[0, 0], 0.0)
    testing.assert_equal((2.0 / a)[0, 0], 2.0)
    testing.assert_equal((a < 2.0)[0, 0], True)
    testing.assert_equal((a.view() < 2.0)[0, 0], True)


def test_operator_with_the_same_matrix_on_both_sides() raises:
    """`a + a` stays legal: the conversion yields a read-only view."""
    var a = la.matrix[DType.float64]([[1.0, 2.0], [3.0, 4.0]])
    testing.assert_equal((a + a)[0, 0], 2.0)
    testing.assert_equal((a + a[0:2, 0:2])[1, 1], 8.0)


def main() raises:
    var suite = testing.TestSuite.discover_tests[__functions_in_module()]()
    suite^.run()
