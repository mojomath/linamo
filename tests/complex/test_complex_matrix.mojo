"""
Tests for matrices whose elements are complex numbers.

`Complex` conforms to `Numeric`, and that one declaration is the whole of what
Linamo does for complex matrices: the creation routines, the operators, the
matrix product and the reductions were all written against `Numeric` rather
than against a `DType`, so they take a `Matrix[CFloat64]` without a line
written for it. These tests are the evidence for that claim --- each one calls
a routine that predates complex support entirely.

Elimination is in `test_complex_linalg.mojo`, which needed one change to the
library rather than none: partial pivoting used to rank candidates through
`Comparable`, and there is no ordering on the complex plane.
"""

import std.testing as testing
import linamo as la
from linamo import BInt, CFloat64


def _a() raises -> la.Matrix[CFloat64]:
    """Returns `[[1+2i, 3-i], [i, 2]]`."""
    return la.matrix[CFloat64](
        [
            [CFloat64(1, 2), CFloat64(3, -1)],
            [CFloat64(0, 1), CFloat64(2, 0)],
        ]
    )


# ===----------------------------------------------------------------------===#
# Creation
# ===----------------------------------------------------------------------===#


def test_zeros_fills_with_the_additive_identity() raises:
    """`zeros` reaches `T.zero()`, so it needs nothing complex-specific."""
    var m = la.zeros[CFloat64](2, 3)
    testing.assert_equal(m.nrows(), 2)
    testing.assert_equal(m.ncols(), 3)
    for i in range(2):
        for j in range(3):
            testing.assert_equal(m[i, j], CFloat64.zero())


def test_eye_puts_one_on_the_diagonal() raises:
    """And `1+0i` is what `one()` means here."""
    var m = la.eye[CFloat64](3)
    for i in range(3):
        for j in range(3):
            var expected = CFloat64.one() if i == j else CFloat64.zero()
            testing.assert_equal(m[i, j], expected)


def test_full_takes_a_complex_fill_value() raises:
    var m = la.full[CFloat64](2, 2, CFloat64(1, -1))
    for i in range(2):
        for j in range(2):
            testing.assert_equal(m[i, j], CFloat64(1, -1))


def test_the_shape_constructor_accepts_a_numeric_element() raises:
    """`Matrix[CFloat64](2, 2, 2, 1)` zero-fills as the scalar form does."""
    var m = la.Matrix[CFloat64](2, 2, 2, 1)
    testing.assert_equal(m[0, 0], CFloat64.zero())
    testing.assert_equal(m[1, 1], CFloat64.zero())


# ===----------------------------------------------------------------------===#
# Element-wise arithmetic
# ===----------------------------------------------------------------------===#


def test_addition_is_element_wise() raises:
    var a = _a()
    var sum = a + a
    testing.assert_equal(sum[0, 0], CFloat64(2, 4))
    testing.assert_equal(sum[0, 1], CFloat64(6, -2))
    testing.assert_equal(sum[1, 0], CFloat64(0, 2))
    testing.assert_equal(sum[1, 1], CFloat64(4, 0))


def test_subtraction_of_a_matrix_from_itself_is_zero() raises:
    var a = _a()
    var diff = a - a
    for i in range(2):
        for j in range(2):
            testing.assert_equal(diff[i, j], CFloat64.zero())


def test_mul_is_the_hadamard_product() raises:
    """`.mul()` squares each entry; `(1+2i)^2 == -3+4i`."""
    var a = _a()
    var hadamard = a.mul(a)
    testing.assert_equal(hadamard[0, 0], CFloat64(-3, 4))
    testing.assert_equal(hadamard[0, 1], CFloat64(8, -6))
    testing.assert_equal(hadamard[1, 0], CFloat64(-1, 0))
    testing.assert_equal(hadamard[1, 1], CFloat64(4, 0))


def test_negation_negates_every_entry() raises:
    var a = _a()
    var negated = -a
    for i in range(2):
        for j in range(2):
            testing.assert_equal(negated[i, j], -a[i, j])


# ===----------------------------------------------------------------------===#
# Matrix product
# ===----------------------------------------------------------------------===#


def test_matmul_uses_complex_arithmetic() raises:
    """Verified by hand: the `[0,0]` entry is `(1+2i)^2 + (3-i)i == -2+7i`."""
    var a = _a()
    var product = a @ a
    testing.assert_equal(product[0, 0], CFloat64(-2, 7))
    testing.assert_equal(product[0, 1], CFloat64(11, 3))
    testing.assert_equal(product[1, 0], CFloat64(-2, 3))
    testing.assert_equal(product[1, 1], CFloat64(5, 3))


def test_star_is_the_matrix_product_here_too() raises:
    """`*` between two matrices is `@`, as it is for every element type."""
    var a = _a()
    testing.assert_equal((a * a)[0, 0], (a @ a)[0, 0])


def test_identity_is_the_multiplicative_identity() raises:
    var a = _a()
    var product = a @ la.eye[CFloat64](2)
    for i in range(2):
        for j in range(2):
            testing.assert_equal(product[i, j], a[i, j])


# ===----------------------------------------------------------------------===#
# Reductions and manipulation
# ===----------------------------------------------------------------------===#


def test_trace_sums_the_diagonal() raises:
    """`(1+2i) + (2+0i) == 3+2i`."""
    testing.assert_equal(la.trace(_a()), CFloat64(3, 2))


def test_sum_adds_every_entry() raises:
    """`(1+2i) + (3-i) + i + 2 == 6+2i`."""
    testing.assert_equal(la.sum(_a()), CFloat64(6, 2))


def test_transpose_is_not_conjugated() raises:
    """Plain transpose, so `[0,1]` and `[1,0]` swap with their signs intact.

    The conjugate transpose a complex algorithm actually wants does not exist
    yet; this test records that `transpose` is the ordinary one.
    """
    var t = la.transpose(_a())
    testing.assert_equal(t[0, 1], CFloat64(0, 1))
    testing.assert_equal(t[1, 0], CFloat64(3, -1))


def test_reshape_moves_no_arithmetic() raises:
    """Structural routines never touch the element type."""
    var flat = la.reshape(_a(), 1, 4)
    testing.assert_equal(flat.nrows(), 1)
    testing.assert_equal(flat.ncols(), 4)
    testing.assert_equal(flat[0, 0], CFloat64(1, 2))


# ===----------------------------------------------------------------------===#
# from_string
# ===----------------------------------------------------------------------===#


def test_from_string_reads_a_complex_literal() raises:
    var m = la.from_string[CFloat64]("[[1+2i, 3-1i], [2i, -i]]")
    testing.assert_equal(m[0, 0], CFloat64(1, 2))
    testing.assert_equal(m[0, 1], CFloat64(3, -1))
    testing.assert_equal(m[1, 0], CFloat64(0, 2))
    testing.assert_equal(m[1, 1], CFloat64(0, -1))


def test_from_string_mixes_real_and_imaginary_cells() raises:
    """A mostly-real matrix does not have to spell `+0i` everywhere."""
    var m = la.from_string[CFloat64]("[[1, 2i], [-i, 3.5-0.5i]]")
    testing.assert_equal(m[0, 0], CFloat64(1, 0))
    testing.assert_equal(m[0, 1], CFloat64(0, 2))
    testing.assert_equal(m[1, 1], CFloat64(3.5, -0.5))


def test_the_printed_matrix_parses_back() raises:
    """The body of `String(m)` is a literal `from_string` accepts.

    The printer puts no spaces inside a cell, which is what keeps this closed:
    the matrix tokenizer splits on whitespace as well as on commas, so a cell
    written `1 + 2i` would arrive as three cells.
    """
    var m = _a()
    var body = String(String(m).split("\n", 1)[1])
    var back = la.from_string[CFloat64](body)
    for i in range(2):
        for j in range(2):
            testing.assert_equal(back[i, j], m[i, j])


# ===----------------------------------------------------------------------===#
# Conjugate transpose
# ===----------------------------------------------------------------------===#


def test_conj_transpose_transposes_and_conjugates() raises:
    var h = la.conj_transpose(_a())
    testing.assert_equal(h[0, 0], CFloat64(1, -2))
    testing.assert_equal(h[0, 1], CFloat64(0, -1))
    testing.assert_equal(h[1, 0], CFloat64(3, 1))
    testing.assert_equal(h[1, 1], CFloat64(2, 0))


def test_conj_transpose_differs_from_transpose() raises:
    """The distinction the name exists for."""
    var a = _a()
    testing.assert_equal(la.transpose(a)[0, 1], CFloat64(0, 1))
    testing.assert_equal(la.conj_transpose(a)[0, 1], CFloat64(0, -1))


def test_the_method_agrees_with_the_routine() raises:
    var a = _a()
    var by_method = a.conj_transpose()
    var by_routine = la.conj_transpose(a)
    for i in range(2):
        for j in range(2):
            testing.assert_equal(by_method[i, j], by_routine[i, j])


def test_a_times_its_conjugate_transpose_is_hermitian() raises:
    """The property Cholesky and QR need: the diagonal is real and the
    off-diagonal entries are conjugate pairs.

    `A @ transpose(A)` has neither, which is why the plain transpose cannot
    stand in for this one over complex elements.
    """
    var a = _a()
    var h = a @ la.conj_transpose(a)
    testing.assert_equal(h[0, 0].im(), 0.0, "the diagonal is real")
    testing.assert_equal(h[1, 1].im(), 0.0, "the diagonal is real")
    testing.assert_true(h[0, 0].re() > 0.0, "and non-negative")
    testing.assert_equal(h[0, 1], h[1, 0].conj(), "off-diagonals are conjugate")


def test_conj_transpose_is_a_plain_transpose_on_a_real_element() raises:
    """A real element type is its own conjugate, so the two agree.

    This is why `conj_transpose` is offered for every element type rather than
    for complex alone: an algorithm written in terms of it stays correct here.
    """
    var f = la.matrix[Float64]([[1.0, 2.0], [3.0, 4.0]])
    testing.assert_equal(la.conj_transpose(f)[0, 1], la.transpose(f)[0, 1])
    var b = la.from_string[BInt]("[[1, 2], [3, 4]]")
    testing.assert_equal(la.conj_transpose(b)[0, 1], la.transpose(b)[0, 1])


# ===----------------------------------------------------------------------===#
# Formatting
# ===----------------------------------------------------------------------===#


def test_the_header_names_the_component_dtype() raises:
    """`Matrix[Complex[float64]]`, not `Matrix[Complex]`.

    A parameterised element keeps its parameters, so a 32-bit complex matrix
    and a 64-bit one do not print the same header.
    """
    var text = String(_a())
    testing.assert_equal(text.split("\n")[0], "Matrix[Complex[float64]] 2x2")


def test_every_entry_shows_both_components() raises:
    """Including the ones whose imaginary part is zero."""
    var text = String(_a())
    testing.assert_true("2.0+0.0i" in text, "a real-valued entry keeps its 0i")
    testing.assert_true("3.0-1.0i" in text, "a negative part uses one sign")


def main() raises:
    testing.TestSuite.discover_tests[__functions_in_module()]().run()
