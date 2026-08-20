"""
Tests for matrices of arbitrary-precision elements.

Two halves, and the split is the point of `linamo.decimo` existing:

- The routines that only *move* or *compare* elements come from core Linamo.
  They are generic over the element type already and need no import from
  `linamo.decimo` at all.
- The arithmetic comes from `linamo.decimo`, because a `BigInt` has no dtype
  and no vector instruction, so it cannot go through the scalar kernels.
"""

import std.testing as testing
import linamo as la
import linamo.decimo as lad
from decimo import BigInt


def _a() raises -> la.Matrix[BigInt]:
    """Returns the 2x2 matrix `[[1, 2], [3, 4]]`."""
    return la.matrix[BigInt](
        [
            [BigInt(1), BigInt(2)],
            [BigInt(3), BigInt(4)],
        ]
    )


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
    b[0, 0] = BigInt(99)
    testing.assert_equal(String(a[0, 0]), "1")
    testing.assert_equal(String(b[0, 0]), "99")


def test_element_larger_than_any_dtype() raises:
    """The reason for all of this: an element no `DType` can hold."""
    var factorial = BigInt.one()
    for k in range(1, 31):
        factorial = factorial * BigInt(k)
    var m = lad.diag([factorial.copy(), factorial.copy()])
    # 30! = 265252859812191058636308480000000, well past 2^64.
    testing.assert_equal(String(m[0, 0]), "265252859812191058636308480000000")
    testing.assert_equal(
        String(lad.trace(m)), "530505719624382117272616960000000"
    )


# ===----------------------------------------------------------------------===#
# Routines that need no arithmetic, and so need no `linamo.decimo`
# ===----------------------------------------------------------------------===#


def test_transpose_is_generic() raises:
    """`transpose` only moves elements, so core Linamo already has it."""
    var t = la.transpose(_a())
    testing.assert_equal(String(t[0, 1]), "3")
    testing.assert_equal(String(t[1, 0]), "2")


def test_slicing_yields_a_view() raises:
    """Slicing a `BigInt` matrix borrows rather than copies."""
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
    """`sort` asks only for `Comparable`, which `BigInt` already declares."""
    var m = la.matrix[BigInt](
        [
            [BigInt(30), BigInt(10), BigInt(20)],
        ]
    )
    var s = la.sort(m, 1)
    testing.assert_equal(String(s[0, 0]), "10")
    testing.assert_equal(String(s[0, 1]), "20")
    testing.assert_equal(String(s[0, 2]), "30")


def test_argsort_and_argmax_are_generic() raises:
    """The index routines compare elements and return `int64` positions."""
    var m = la.matrix[BigInt](
        [
            [BigInt(30), BigInt(10), BigInt(20)],
        ]
    )
    var order = la.argsort(m, 1)
    testing.assert_equal(order[0, 0], 1)
    testing.assert_equal(order[0, 2], 0)
    testing.assert_equal(la.argmax(m), 0)
    testing.assert_equal(la.argmin(m), 1)


# ===----------------------------------------------------------------------===#
# Arithmetic, from `linamo.decimo`
# ===----------------------------------------------------------------------===#


def test_add_and_sub() raises:
    """Element-wise addition and subtraction."""
    var a = _a()
    var s = lad.add(a, a)
    testing.assert_equal(String(s[0, 0]), "2")
    testing.assert_equal(String(s[1, 1]), "8")
    var d = lad.sub(s, a)
    testing.assert_equal(String(d[1, 1]), "4")


def test_mul_is_element_wise() raises:
    """`mul` is the Hadamard product, not matrix multiplication."""
    var p = lad.mul(_a(), _a())
    testing.assert_equal(String(p[0, 1]), "4")
    testing.assert_equal(String(p[1, 0]), "9")


def test_neg_and_scalar_ops() raises:
    """Negation, and the two scalar-operand forms."""
    var n = lad.neg(_a())
    testing.assert_equal(String(n[0, 0]), "-1")
    var s = lad.scalar_add(_a(), BigInt(10))
    testing.assert_equal(String(s[1, 1]), "14")
    var t = lad.scalar_mul(_a(), BigInt(3))
    testing.assert_equal(String(t[1, 1]), "12")


def test_matmul() raises:
    """`[[1, 2], [3, 4]]` squared is `[[7, 10], [15, 22]]`."""
    var p = lad.matmul(_a(), _a())
    testing.assert_equal(String(p[0, 0]), "7")
    testing.assert_equal(String(p[0, 1]), "10")
    testing.assert_equal(String(p[1, 0]), "15")
    testing.assert_equal(String(p[1, 1]), "22")


def test_matmul_against_identity() raises:
    """Multiplying by the identity leaves a matrix unchanged."""
    var p = lad.matmul(_a(), lad.identity[BigInt](2))
    testing.assert_equal(String(p[0, 1]), "2")
    testing.assert_equal(String(p[1, 0]), "3")


def test_matmul_shape_mismatch_raises() raises:
    """Mismatched inner dimensions are a `ValueError`, not a wrong answer."""
    var wide = la.matrix[BigInt]([[BigInt(1), BigInt(2), BigInt(3)]])
    with testing.assert_raises():
        var _p = lad.matmul(wide, wide)


def test_add_shape_mismatch_raises() raises:
    """Element-wise operations check their shapes."""
    var wide = la.matrix[BigInt]([[BigInt(1), BigInt(2), BigInt(3)]])
    with testing.assert_raises():
        var _s = lad.add(_a(), wide)


def test_total_and_trace() raises:
    """The whole-matrix sum and the diagonal sum."""
    testing.assert_equal(String(lad.total(_a())), "10")
    testing.assert_equal(String(lad.trace(_a())), "5")


def test_trace_requires_square() raises:
    """A trace is defined for square matrices only."""
    var wide = la.matrix[BigInt]([[BigInt(1), BigInt(2), BigInt(3)]])
    with testing.assert_raises():
        var _t = lad.trace(wide)


def test_zeros_ones_and_eye() raises:
    """The creation routines ask `Numeric` for a zero and a one."""
    var z = lad.zeros[BigInt](2, 3)
    testing.assert_equal(z.size(), 6)
    testing.assert_equal(String(z[1, 2]), "0")
    var o = lad.ones[BigInt](2, 2)
    testing.assert_equal(String(o[0, 1]), "1")
    var i = lad.eye[BigInt](3)
    testing.assert_equal(String(i[1, 1]), "1")
    testing.assert_equal(String(i[1, 2]), "0")


def test_views_are_accepted_as_operands() raises:
    """A `MatrixView` operand works wherever a `Matrix` does."""
    var a = _a()
    var top = a[0:1, :]
    var s = lad.add(top, top)
    testing.assert_equal(s.nrows(), 1)
    testing.assert_equal(String(s[0, 1]), "4")


def main() raises:
    testing.TestSuite.discover_tests[__functions_in_module()]().run()
