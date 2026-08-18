"""
Tests for the implicit `Matrix` -> `MatrixView` conversion.

Every named routine in the library takes `MatrixView` operands and nothing
else. A `Matrix` argument still works because `MatrixView` has an `@implicit`
constructor from `Matrix`, so the compiler inserts the conversion. That is what
replaced the four hand-written overloads (view x view, mat x mat, mat x view,
view x mat) that each binary operation used to carry.

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


def main() raises:
    var suite = testing.TestSuite.discover_tests[__functions_in_module()]()
    suite^.run()
