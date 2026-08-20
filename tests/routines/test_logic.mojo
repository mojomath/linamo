"""
Tests for the Phase 5.6 approximate comparison and logical connectives.

Three things get separate attention, because each fails on its own:

*The tolerance formula*, `|a - b| <= atol + rtol * |b|`, which is asymmetric
in its operands. A test that only ever compares a value with itself cannot
tell that formula apart from a symmetric one.

*The non-finite cases*, where NumPy's rules are not the ones the formula would
give: `inf` is close to `inf` although `inf - inf` is NaN, and a NaN is close
to nothing at all unless `equal_nan` says otherwise.

*Strided operands*, since every routine funnels through views and the
contiguous fast path is a different body of code from the strided one.
"""

import std.testing as testing
import linamo as la
from linamo.routines.logic import (
    all,
    allclose,
    isclose,
    logical_and,
    logical_not,
    logical_or,
    logical_xor,
    scalar_allclose,
    scalar_isclose,
    scalar_logical_and,
    scalar_logical_or,
    scalar_logical_xor,
)


def _nan() raises -> Float64:
    return Float64("nan")


def _inf() raises -> Float64:
    return Float64("inf")


# ===----------------------------------------------------------------------===#
# isclose
# ===----------------------------------------------------------------------===#


def test_isclose_within_default_tolerance() raises:
    """Test that a difference under the default rtol reads as close."""
    var a = la.matrix[Float64]([[1.0, 2.0], [3.0, 4.0]])
    var b = la.matrix[Float64]([[1.0, 2.000000001], [3.0, 4.0]])
    testing.assert_true(all(isclose(a, b)))


def test_isclose_outside_default_tolerance() raises:
    """Test that a difference over the default rtol reads as distant."""
    var a = la.matrix[Float64]([[1.0, 2.0]])
    var b = la.matrix[Float64]([[1.0, 2.5]])
    var mask = isclose(a, b)
    testing.assert_true(mask[0, 0])
    testing.assert_false(mask[0, 1])


def test_isclose_custom_tolerances() raises:
    """Test that a widened rtol admits a difference the default rejects."""
    var a = la.matrix[Float64]([[1.0, 2.0]])
    var b = la.matrix[Float64]([[1.0, 2.5]])
    testing.assert_true(all(isclose(a, b, rtol=0.3)))
    testing.assert_false(all(isclose(a, b, rtol=0.1)))


def test_isclose_atol_decides_at_zero() raises:
    """Test that atol, not rtol, settles a comparison against zero.

    `rtol * |b|` vanishes when the reference is zero, so without an absolute
    tolerance nothing but an exact zero would ever be close to zero.
    """
    var a = la.matrix[Float64]([[1e-9]])
    var b = la.matrix[Float64]([[0.0]])
    testing.assert_true(all(isclose(a, b)))
    testing.assert_false(all(isclose(a, b, atol=0.0)))


def test_isclose_is_asymmetric_in_its_operands() raises:
    """Test that the second operand is the reference, as in NumPy.

    With atol at zero the test is `|a - b| <= rtol * |b|`, so the operand
    order decides the answer whenever the two differ in magnitude.
    """
    var small = la.matrix[Float64]([[1e-8]])
    var zero = la.matrix[Float64]([[0.0]])
    testing.assert_false(all(isclose(small, zero, rtol=1.0, atol=0.0)))
    testing.assert_true(all(isclose(zero, small, rtol=1.0, atol=0.0)))


def test_isclose_nan_is_close_to_nothing() raises:
    """Test that a NaN is not close to a NaN, nor to a number, by default."""
    var a = la.matrix[Float64]([[_nan(), _nan()]])
    var b = la.matrix[Float64]([[_nan(), 1.0]])
    var mask = isclose(a, b)
    testing.assert_false(mask[0, 0])
    testing.assert_false(mask[0, 1])


def test_isclose_equal_nan() raises:
    """Test that equal_nan makes a NaN close to a NaN, and only to a NaN."""
    var a = la.matrix[Float64]([[_nan(), _nan()]])
    var b = la.matrix[Float64]([[_nan(), 1.0]])
    var mask = isclose(a, b, equal_nan=True)
    testing.assert_true(mask[0, 0])
    testing.assert_false(mask[0, 1])


def test_isclose_infinities() raises:
    """Test that equal infinities are close and opposite ones are not.

    The tolerance formula cannot answer this: `inf - inf` is a NaN, which
    compares false against every tolerance, so the non-finite case is decided
    by equality instead.
    """
    var a = la.matrix[Float64]([[_inf(), _inf(), _inf()]])
    var b = la.matrix[Float64]([[_inf(), -_inf(), 1e308]])
    var mask = isclose(a, b)
    testing.assert_true(mask[0, 0])
    testing.assert_false(mask[0, 1])
    testing.assert_false(mask[0, 2])


def test_isclose_strided_operands() raises:
    """Test that the strided path agrees with the contiguous one."""
    var a = la.matrix[Float64](
        [[1.0, 9.0, 2.0], [9.0, 9.0, 9.0], [3.0, 9.0, 4.5]]
    )
    var b = la.matrix[Float64](
        [[1.0, 0.0, 2.0], [0.0, 0.0, 0.0], [3.0, 0.0, 4.0]]
    )
    var mask = isclose(a[0:3:2, 0:3:2], b[0:3:2, 0:3:2])
    testing.assert_equal(mask.nrows(), 2)
    testing.assert_equal(mask.ncols(), 2)
    testing.assert_true(mask[0, 0])
    testing.assert_true(mask[0, 1])
    testing.assert_true(mask[1, 0])
    testing.assert_false(mask[1, 1])


def test_isclose_result_is_c_contiguous() raises:
    """Test that the mask is a freshly allocated C-contiguous matrix."""
    var a = la.matrix[Float64]([[1.0, 2.0], [3.0, 4.0]], "F")
    var mask = isclose(a, a)
    testing.assert_true(mask.is_c_contiguous())


def test_isclose_shape_mismatch_raises() raises:
    """Test that operands of different shapes raise."""
    var a = la.matrix[Float64]([[1.0, 2.0]])
    var b = la.matrix[Float64]([[1.0]])
    var raised = False
    try:
        var _m = isclose(a, b)
    except:
        raised = True
    testing.assert_true(raised, "A shape mismatch should raise")


# ===----------------------------------------------------------------------===#
# allclose
# ===----------------------------------------------------------------------===#


def test_allclose_true_and_false() raises:
    """Test the reduction against a matrix that agrees and one that does not."""
    var a = la.matrix[Float64]([[1.0, 2.0], [3.0, 4.0]])
    var b = la.matrix[Float64]([[1.0, 2.000000001], [3.0, 4.0]])
    var c = la.matrix[Float64]([[1.0, 2.0], [3.0, 4.5]])
    testing.assert_true(allclose(a, b))
    testing.assert_false(allclose(a, c))


def test_allclose_agrees_with_the_mask() raises:
    """Test that allclose is `all(isclose(...))` on every argument form."""
    var a = la.matrix[Float64]([[1.0, _nan()], [_inf(), 4.0]])
    var b = la.matrix[Float64]([[1.0, _nan()], [_inf(), 4.0]])
    testing.assert_equal(allclose(a, b), all(isclose(a, b)))
    testing.assert_equal(
        allclose(a, b, equal_nan=True), all(isclose(a, b, equal_nan=True))
    )


def test_allclose_strided_operands() raises:
    """Test that the walk follows strides rather than the buffer."""
    var a = la.matrix[Float64]([[1.0, 9.0], [2.0, 9.0]])
    var b = la.matrix[Float64]([[1.0, 0.0], [2.0, 0.0]])
    testing.assert_true(allclose(a[:, 0:1], b[:, 0:1]))


def test_allclose_shape_mismatch_raises() raises:
    """Test that operands of different shapes raise."""
    var a = la.matrix[Float64]([[1.0, 2.0]])
    var b = la.matrix[Float64]([[1.0]])
    var raised = False
    try:
        var _b = allclose(a, b)
    except:
        raised = True
    testing.assert_true(raised, "A shape mismatch should raise")


# ===----------------------------------------------------------------------===#
# scalar_isclose / scalar_allclose
# ===----------------------------------------------------------------------===#


def test_scalar_isclose() raises:
    """Test the mask of a matrix against a single reference value."""
    var m = la.matrix[Float64]([[2.0, 2.000000001], [2.5, -2.0]])
    var mask = scalar_isclose(m, 2.0)
    testing.assert_true(mask[0, 0])
    testing.assert_true(mask[0, 1])
    testing.assert_false(mask[1, 0])
    testing.assert_false(mask[1, 1])


def test_scalar_isclose_at_zero() raises:
    """Test that atol carries the comparison against a zero reference."""
    var m = la.matrix[Float64]([[0.0, 1e-9, 1e-3]])
    var mask = scalar_isclose(m, 0.0)
    testing.assert_true(mask[0, 0])
    testing.assert_true(mask[0, 1])
    testing.assert_false(mask[0, 2])


def test_scalar_isclose_strided() raises:
    """Test the strided path of the scalar form."""
    var m = la.matrix[Float64]([[2.0, 9.0], [2.5, 9.0]])
    var mask = scalar_isclose(m[:, 0:1], 2.0)
    testing.assert_equal(mask.ncols(), 1)
    testing.assert_true(mask[0, 0])
    testing.assert_false(mask[1, 0])


def test_scalar_allclose() raises:
    """Test the reduced scalar form."""
    var uniform = la.matrix[Float64]([[3.0, 3.0], [3.0, 3.0]])
    var mixed = la.matrix[Float64]([[3.0, 3.0], [3.0, 3.5]])
    testing.assert_true(scalar_allclose(uniform, 3.0))
    testing.assert_false(scalar_allclose(mixed, 3.0))


# ===----------------------------------------------------------------------===#
# Logical connectives
# ===----------------------------------------------------------------------===#


def test_logical_and_on_masks() raises:
    """Test the conjunction of two boolean masks."""
    var m = la.matrix[Float64]([[1.0, 2.0], [3.0, 4.0]])
    var mask = logical_and(m > 1.0, m < 4.0)
    testing.assert_false(mask[0, 0])
    testing.assert_true(mask[0, 1])
    testing.assert_true(mask[1, 0])
    testing.assert_false(mask[1, 1])


def test_logical_and_reads_truthiness() raises:
    """Test that a numeric operand is read for truthiness, not bitwise.

    `2 & 1` is zero, so a bitwise implementation would disagree here.
    """
    var a = la.matrix[Float64]([[2.0, 0.0], [2.0, 0.0]])
    var b = la.matrix[Float64]([[1.0, 1.0], [0.0, 0.0]])
    var mask = logical_and(a, b)
    testing.assert_true(mask[0, 0])
    testing.assert_false(mask[0, 1])
    testing.assert_false(mask[1, 0])
    testing.assert_false(mask[1, 1])


def test_logical_or() raises:
    """Test the disjunction."""
    var a = la.matrix[Float64]([[2.0, 0.0], [2.0, 0.0]])
    var b = la.matrix[Float64]([[1.0, 1.0], [0.0, 0.0]])
    var mask = logical_or(a, b)
    testing.assert_true(mask[0, 0])
    testing.assert_true(mask[0, 1])
    testing.assert_true(mask[1, 0])
    testing.assert_false(mask[1, 1])


def test_logical_xor() raises:
    """Test the exclusive disjunction."""
    var a = la.matrix[Float64]([[2.0, 0.0], [2.0, 0.0]])
    var b = la.matrix[Float64]([[1.0, 1.0], [0.0, 0.0]])
    var mask = logical_xor(a, b)
    testing.assert_false(mask[0, 0])
    testing.assert_true(mask[0, 1])
    testing.assert_true(mask[1, 0])
    testing.assert_false(mask[1, 1])


def test_logical_not() raises:
    """Test that negation is a test against zero."""
    var m = la.matrix[Float64]([[0.0, 1.0], [-2.0, 0.0]])
    var mask = logical_not(m)
    testing.assert_true(mask[0, 0])
    testing.assert_false(mask[0, 1])
    testing.assert_false(mask[1, 0])
    testing.assert_true(mask[1, 1])


def test_logical_not_of_a_mask() raises:
    """Test negation of a boolean operand, the common case."""
    var m = la.matrix[Float64]([[1.0, 5.0]])
    var mask = logical_not(m > 2.0)
    testing.assert_true(mask[0, 0])
    testing.assert_false(mask[0, 1])


def test_logical_on_an_integer_dtype() raises:
    """Test that the connectives accept any dtype, unlike isclose."""
    var a = la.matrix[Int32]([[2, 0]])
    var b = la.matrix[Int32]([[1, 3]])
    var mask = logical_and(a, b)
    testing.assert_true(mask[0, 0])
    testing.assert_false(mask[0, 1])
    testing.assert_true(logical_not(a)[0, 1])


def test_logical_strided_operands() raises:
    """Test that the connectives follow strides."""
    var a = la.matrix[Float64]([[1.0, 9.0], [0.0, 9.0]])
    var b = la.matrix[Float64]([[1.0, 9.0], [1.0, 9.0]])
    var mask = logical_and(a[:, 0:1], b[:, 0:1])
    testing.assert_equal(mask.ncols(), 1)
    testing.assert_true(mask[0, 0])
    testing.assert_false(mask[1, 0])


def test_logical_shape_mismatch_raises() raises:
    """Test that operands of different shapes raise."""
    var a = la.matrix[Float64]([[1.0, 2.0]])
    var b = la.matrix[Float64]([[1.0]])
    var raised = False
    try:
        var _m = logical_or(a, b)
    except:
        raised = True
    testing.assert_true(raised, "A shape mismatch should raise")


def test_scalar_logical_forms() raises:
    """Test the connectives against a single value."""
    var m = la.matrix[Float64]([[1.0, 0.0]])
    var conj = scalar_logical_and(m, 1.0)
    testing.assert_true(conj[0, 0])
    testing.assert_false(conj[0, 1])

    var disj = scalar_logical_or(m, 0.0)
    testing.assert_true(disj[0, 0])
    testing.assert_false(disj[0, 1])

    var excl = scalar_logical_xor(m, 1.0)
    testing.assert_false(excl[0, 0])
    testing.assert_true(excl[0, 1])


def main() raises:
    testing.TestSuite.discover_tests[__functions_in_module()]().run()
