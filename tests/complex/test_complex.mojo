"""
Tests for the `Complex` element type.

The arithmetic under test here is the stdlib's --- `Complex` forwards every
operator to `std.complex.ComplexSIMD`. What these tests pin down is the part
Linamo owns: that the forwarding is faithful, that the two places the wrapper
deliberately departs from the stdlib do what they claim (division by zero
raises rather than returning a NaN, and formatting always writes both
components), and that `zero()`, `one()` and `sqrt()` --- which the stdlib has
no equivalent of --- are correct.

`test_complex_matrix.mojo` covers what the conformance buys: a `Matrix` of
these.
"""

import std.testing as testing
from linamo.types.complex import Complex, CFloat32, CFloat64, ComplexFloat64
from std.complex import ComplexScalar


# ===----------------------------------------------------------------------===#
# Components and construction
# ===----------------------------------------------------------------------===#


def test_components_round_trip() raises:
    """`re()` and `im()` return what the constructor was given."""
    var z = CFloat64(3, -4)
    testing.assert_equal(z.re(), 3.0)
    testing.assert_equal(z.im(), -4.0)


def test_imaginary_part_defaults_to_zero() raises:
    """One argument makes a real number, not an uninitialised one."""
    var z = CFloat64(2.5)
    testing.assert_equal(z.re(), 2.5)
    testing.assert_equal(z.im(), 0.0)


def test_wraps_the_stdlib_type_implicitly() raises:
    """A `ComplexScalar[d]` converts without being named."""
    var inner = ComplexScalar[DType.float64](1, 2)
    var z: CFloat64 = inner
    testing.assert_equal(z.re(), 1.0)
    testing.assert_equal(z.im(), 2.0)


def test_std_returns_the_wrapped_value() raises:
    """`std()` is the way back out to `std.complex`."""
    var z = CFloat64(1, 2)
    testing.assert_equal(z.std().re, 1.0)
    testing.assert_equal(z.std().im, 2.0)


def test_the_long_and_short_names_are_the_same_type() raises:
    """`CFloat64` and `ComplexFloat64` are one type, not two."""
    var z: ComplexFloat64 = CFloat64(1, 1)
    testing.assert_equal(z.re(), 1.0)


# ===----------------------------------------------------------------------===#
# Numeric
# ===----------------------------------------------------------------------===#


def test_zero_and_one_are_the_identities() raises:
    """`zero()` leaves a value unchanged under `+`, `one()` under `*`."""
    var z = CFloat64(3, -4)
    testing.assert_equal(z + CFloat64.zero(), z)
    testing.assert_equal(z * CFloat64.one(), z)


def test_addition_and_subtraction_are_componentwise() raises:
    """The real and imaginary parts add and subtract independently."""
    var a = CFloat64(1, 2)
    var b = CFloat64(3, -5)
    testing.assert_equal(a + b, CFloat64(4, -3))
    testing.assert_equal(a - b, CFloat64(-2, 7))


def test_multiplication_is_complex_not_componentwise() raises:
    """`(1+2i)(3-i) == 5+5i`, which the component-wise product is not."""
    var product = CFloat64(1, 2) * CFloat64(3, -1)
    testing.assert_equal(product, CFloat64(5, 5))


def test_i_squared_is_minus_one() raises:
    """The defining property of the imaginary unit."""
    var i = CFloat64(0, 1)
    testing.assert_equal(i * i, CFloat64(-1, 0))


def test_division_inverts_multiplication() raises:
    """`(a*b)/b == a`, and the quotient is the conjugate over the norm."""
    var a = CFloat64(1, 2)
    var b = CFloat64(3, -1)
    testing.assert_equal((a * b) / b, a)
    # 1/(3+4i) == (3-4i)/25
    testing.assert_equal(CFloat64.one() / CFloat64(3, 4), CFloat64(0.12, -0.16))


def test_division_by_zero_raises() raises:
    """Where the stdlib yields a NaN, `Numeric` promises an error.

    This is one of the two places the wrapper does not simply forward.
    """
    with testing.assert_raises():
        _ = CFloat64(1, 1) / CFloat64.zero()


def test_negation_flips_both_components() raises:
    """`-z` is the additive inverse, not the conjugate."""
    var z = CFloat64(3, -4)
    testing.assert_equal(-z, CFloat64(-3, 4))
    testing.assert_equal(z + -z, CFloat64.zero())


# ===----------------------------------------------------------------------===#
# Magnitude and conjugate
# ===----------------------------------------------------------------------===#


def test_norm_is_the_hypotenuse() raises:
    """`|3+4i| == 5`. This is the value elimination will pivot on.

    Spelled `norm()` and not `abs()`: `Absable` requires the result to be the
    input's own type, and a magnitude is real.
    """
    testing.assert_equal(CFloat64(3, -4).norm(), 5.0)


def test_squared_norm_skips_the_root() raises:
    """`squared_norm()` orders magnitudes exactly as `norm()` does."""
    testing.assert_equal(CFloat64(3, -4).squared_norm(), 25.0)


def test_conjugate_flips_only_the_imaginary_part() raises:
    """And `z * conj(z)` is the squared magnitude, with no imaginary part."""
    var z = CFloat64(3, -4)
    testing.assert_equal(z.conj(), CFloat64(3, 4))
    testing.assert_equal(z * z.conj(), CFloat64(25, 0))


# ===----------------------------------------------------------------------===#
# Rootable
# ===----------------------------------------------------------------------===#


def test_sqrt_squares_back() raises:
    """`sqrt(3+4i) == 2+i`, exactly."""
    testing.assert_equal(CFloat64(3, 4).sqrt(), CFloat64(2, 1))


def test_sqrt_of_minus_one_is_the_principal_root() raises:
    """`i`, not `-i`: the root with the non-negative real part."""
    testing.assert_equal(CFloat64(-1, 0).sqrt(), CFloat64(0, 1))


def test_sqrt_keeps_the_sign_of_the_imaginary_part() raises:
    """A root below the real axis stays below it."""
    testing.assert_equal(CFloat64(3, -4).sqrt(), CFloat64(2, -1))


def test_sqrt_of_zero_is_zero() raises:
    """The one input the magnitude formula cannot be divided through."""
    testing.assert_equal(CFloat64.zero().sqrt(), CFloat64.zero())


# ===----------------------------------------------------------------------===#
# Formatting
# ===----------------------------------------------------------------------===#
# The other place the wrapper does not forward. The stdlib omits a zero
# imaginary part and writes the sign as part of the number, so its output for
# a column of values takes three different shapes; these tests pin the one
# shape a matrix needs.


def test_a_positive_imaginary_part_is_written_with_a_plus() raises:
    testing.assert_equal(String(CFloat64(1, 2)), "1.0+2.0i")


def test_a_negative_imaginary_part_is_written_with_a_minus() raises:
    """Not `1.0 + -2.0i`, which is what forwarding would have given."""
    testing.assert_equal(String(CFloat64(1, -2)), "1.0-2.0i")


def test_a_zero_imaginary_part_is_still_written() raises:
    """Otherwise a real-valued entry is indistinguishable from a real."""
    testing.assert_equal(String(CFloat64(2, 0)), "2.0+0.0i")


# ===----------------------------------------------------------------------===#
# Parsable
# ===----------------------------------------------------------------------===#
# The literal syntax is `a+bi`, which is what `write_to` emits, so parsing the
# printed form returns the value. Both parts may be left out where they are
# implied.


def test_parses_both_parts() raises:
    testing.assert_equal(CFloat64.from_string("1+2i"), CFloat64(1, 2))
    testing.assert_equal(CFloat64.from_string("3-1i"), CFloat64(3, -1))
    testing.assert_equal(CFloat64.from_string("-1+2i"), CFloat64(-1, 2))
    testing.assert_equal(CFloat64.from_string("-1-2i"), CFloat64(-1, -2))


def test_parses_a_bare_real() raises:
    """No trailing `i`, so the whole token is the real part."""
    testing.assert_equal(CFloat64.from_string("3"), CFloat64(3, 0))
    testing.assert_equal(CFloat64.from_string("-2.5"), CFloat64(-2.5, 0))


def test_parses_a_bare_imaginary() raises:
    """A coefficient with no real part in front of it."""
    testing.assert_equal(CFloat64.from_string("2i"), CFloat64(0, 2))
    testing.assert_equal(CFloat64.from_string("-3.5i"), CFloat64(0, -3.5))


def test_an_omitted_coefficient_is_one() raises:
    """`i` and `-i` carry no digits and mean one and minus one."""
    testing.assert_equal(CFloat64.from_string("i"), CFloat64(0, 1))
    testing.assert_equal(CFloat64.from_string("+i"), CFloat64(0, 1))
    testing.assert_equal(CFloat64.from_string("-i"), CFloat64(0, -1))
    testing.assert_equal(CFloat64.from_string("1-i"), CFloat64(1, -1))


def test_an_exponent_sign_does_not_split_the_parts() raises:
    """`1e-3+2i` has three signs and only the third divides it."""
    testing.assert_equal(CFloat64.from_string("1e-3+2i"), CFloat64(0.001, 2))
    testing.assert_equal(CFloat64.from_string("2e+3i"), CFloat64(0, 2000))
    testing.assert_equal(CFloat64.from_string("1E2-1i"), CFloat64(100, -1))


def test_the_printed_form_parses_back() raises:
    """`from_string(String(z)) == z`, which is the point of the syntax."""
    for z in [CFloat64(1, 2), CFloat64(3, -1), CFloat64(2, 0), CFloat64(0, -1)]:
        testing.assert_equal(CFloat64.from_string(String(z)), z)


def test_an_empty_literal_raises() raises:
    with testing.assert_raises():
        _ = CFloat64.from_string("")


def test_a_non_number_raises() raises:
    with testing.assert_raises():
        _ = CFloat64.from_string("oops")


# ===----------------------------------------------------------------------===#
# Component width
# ===----------------------------------------------------------------------===#


def test_the_component_dtype_is_carried() raises:
    """`CFloat32` is a pair of `Float32`, and rounds like one."""
    var narrow = CFloat32(1, 2)
    testing.assert_equal(narrow.re(), Float32(1))
    var wide = CFloat64(1, 2)
    testing.assert_equal(wide.re(), Float64(1))


def main() raises:
    testing.TestSuite.discover_tests[__functions_in_module()]().run()
