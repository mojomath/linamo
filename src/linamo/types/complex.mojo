"""
This module defines the `Complex` element type, a complex number that a matrix
can hold.
"""

import std.math
from std.complex import ComplexSIMD, ComplexScalar

from decimo import Numeric, Parsable, Rootable

from linamo.errors import ValueError
from linamo.traits.conjugable import Conjugable


# [Mojo Miji]
# This type carries no arithmetic of its own. Every operator below forwards to
# `std.complex.ComplexSIMD`, which is where complex multiplication, division and
# the magnitude actually live and where they should stay.
#
# What the wrapper exists for is a single line the stdlib could not have
# written. `ComplexSIMD` declares `(Equatable, TrivialRegisterPassable,
# Writable, _Expable)`; it does not declare `Numeric`, because `Numeric` is
# decimo's trait and was written afterwards. Mojo's conformance is nominal ---
# declared where the struct is --- so no amount of code in Linamo can add it to
# a type in the stdlib. The same rule that forced `Numeric` into decimo rather
# than here forces this newtype.
#
# The trade is worth making precisely once. Sixty-four places in this library
# are gated on `conforms_to(T, Numeric)`: twenty-eight routines and thirty-six
# methods on `Matrix` and `MatrixView`. Declaring the trait once opens all of
# them, and opens every routine added to that tier later. The alternative ---
# naming `ComplexScalar[d]` in an overload of each --- also works, and needs no
# wrapper, but it needs sixty-four complex twins now and one more with every
# routine after.
struct Complex[d: DType](
    Conjugable,
    Copyable,
    Deinitable,
    Equatable,
    Movable,
    Numeric,
    Parsable,
    Rootable,
    Writable,
):
    """A complex number with `d`-typed real and imaginary parts.

    The element type of a complex matrix. `Matrix[ComplexFloat64]` is an
    ordinary matrix with the ordinary operators: what changes is where the
    arithmetic comes from.

    Parameters:
        d: The dtype of the real and imaginary components.
    """

    comptime Inner = ComplexScalar[Self.d]
    """The stdlib type this one forwards to."""

    comptime Real = Scalar[Self.d]
    """The type of the components, and of the magnitude."""

    var _v: Self.Inner
    """The wrapped value."""

    # ===-----------------------------------------------------------------===#
    # Initialization
    # ===-----------------------------------------------------------------===#

    def __init__(out self, re: Self.Real, im: Self.Real = 0):
        """Initializes a complex number from its components.

        Args:
            re: The real part.
            im: The imaginary part. Defaults to zero, so a real number is
                spelled with one argument.
        """
        # [Mojo Miji]
        # `comptime assert` is only legal inside a function, so the constraint
        # on `d` is stated in the two constructors rather than once on the
        # struct. Every value of this type is built by one of them, `zero()`
        # and `one()` included, so nothing slips past.
        comptime assert Self.d.is_floating_point(), (
            "Complex requires a floating-point component dtype: the quotient"
            " of two Gaussian integers is not a Gaussian integer, so"
            " `__truediv__` could not honour `Numeric`."
        )
        self._v = Self.Inner(re, im)

    @implicit
    def __init__(out self, v: Self.Inner):
        """Initializes a complex number from the stdlib's own complex type.

        Implicit, so a `ComplexFloat64` from `std.complex` may be passed
        wherever this type is expected.

        Args:
            v: The value to wrap.
        """
        comptime assert (
            Self.d.is_floating_point()
        ), "Complex requires a floating-point component dtype."
        self._v = v

    # ===-----------------------------------------------------------------===#
    # Retrieve attributes
    # ===-----------------------------------------------------------------===#

    @always_inline
    def re(self) -> Self.Real:
        """Returns the real part."""
        return self._v.re

    @always_inline
    def im(self) -> Self.Real:
        """Returns the imaginary part."""
        return self._v.im

    @always_inline
    def std(self) -> Self.Inner:
        """Returns the wrapped `std.complex` value.

        The way back out to the stdlib, for the functions this type does not
        re-expose.

        Returns:
            The underlying `ComplexScalar[d]`.
        """
        return self._v

    # ===-----------------------------------------------------------------===#
    # Complex-specific operations
    # ===-----------------------------------------------------------------===#

    @always_inline
    def conj(self) -> Self:
        """Returns the complex conjugate.

        Returns:
            `re - im*i`.
        """
        return Self(self._v.conj())

    @always_inline
    def norm(self) -> Self.Real:
        """Returns the magnitude.

        This is the value elimination pivots on: `Complex` has no ordering ---
        there is none on the complex plane --- but its magnitude is a real
        number and does.

        Returns:
            `sqrt(re*re + im*im)`.
        """
        return self._v.norm()

    @always_inline
    def squared_norm(self) -> Self.Real:
        """Returns the squared magnitude.

        Comparing two squared magnitudes orders them exactly as comparing the
        magnitudes does, and skips a square root.

        Returns:
            `re*re + im*im`.
        """
        return self._v.squared_norm()

    # [Mojo Miji]
    # No `__abs__`, so no `abs(z)`. The stdlib's `Absable` requires the result
    # to be `Self` --- "the absolute value operation always returns the same
    # type as the input" --- and the magnitude of a complex number is real.
    # Conforming would mean returning `5.0+0.0i` for `|3+4i|`, which is a
    # complex number standing where a real one belongs. `norm()` says the true
    # thing instead, and the name is the one `std.complex` already uses.

    # ===-----------------------------------------------------------------===#
    # Numeric
    # ===-----------------------------------------------------------------===#

    @staticmethod
    @always_inline
    def zero() -> Self:
        """Returns the additive identity, `0 + 0i`."""
        return Self(0, 0)

    @staticmethod
    @always_inline
    def one() -> Self:
        """Returns the multiplicative identity, `1 + 0i`."""
        return Self(1, 0)

    @always_inline
    def __neg__(self) raises -> Self:
        """Returns the additive inverse.

        Returns:
            `-re - im*i`.
        """
        return Self(-self._v)

    @always_inline
    def __add__(self, other: Self) raises -> Self:
        """Returns the sum of `self` and `other`.

        Args:
            other: The value to add to `self`.

        Returns:
            The sum.
        """
        return Self(self._v + other._v)

    @always_inline
    def __sub__(self, other: Self) raises -> Self:
        """Returns `other` subtracted from `self`.

        Args:
            other: The value to subtract.

        Returns:
            The difference.
        """
        return Self(self._v - other._v)

    @always_inline
    def __mul__(self, other: Self) raises -> Self:
        """Returns the product of `self` and `other`.

        Args:
            other: The value to multiply `self` by.

        Returns:
            The product.
        """
        return Self(self._v * other._v)

    def __truediv__(self, other: Self) raises -> Self:
        """Returns `self` divided by `other`.

        Args:
            other: The divisor.

        Returns:
            The quotient.

        Raises:
            ValueError: If `other` is zero.
        """
        # [Mojo Miji]
        # The stdlib divides by `squared_norm()` without checking it, which
        # yields an infinity or a NaN rather than an error. `Numeric` promises
        # a raise, and every other element type in this library raises here, so
        # the check belongs on this side of the forward.
        if other._v.squared_norm() == 0:
            raise ValueError(
                function="Complex.__truediv__()",
                message="Division by zero.",
            )
        return Self(self._v / other._v)

    # ===-----------------------------------------------------------------===#
    # Rootable
    # ===-----------------------------------------------------------------===#

    def sqrt(self) raises -> Self:
        """Returns the principal square root.

        The root whose real part is non-negative; on the negative real axis,
        the one with positive imaginary part. So `sqrt(-1) == 1i`, not `-1i`.

        Returns:
            The principal square root of `self`.
        """
        # [Mojo Miji]
        # Computed from the magnitude rather than from polar form: `r` is
        # already non-negative, so both halves below are square roots of
        # non-negative reals, and neither `atan2` nor a quadrant fix-up is
        # needed. Writing it through `abs(im)` and restoring the sign at the
        # end keeps the imaginary part's sign right without a branch on the
        # quadrant.
        var re = self._v.re
        var im = self._v.im
        if re == 0 and im == 0:
            return Self.zero()

        var r = self.norm()
        var root_re = std.math.sqrt((r + re) * 0.5)
        var root_im = std.math.sqrt((r - re) * 0.5)
        return Self(root_re, root_im if im >= 0 else -root_im)

    # ===-----------------------------------------------------------------===#
    # Parsable
    # ===-----------------------------------------------------------------===#

    @staticmethod
    def from_string(value: StringSlice) raises -> Self:
        """Parses a complex literal written as `a+bi`.

        The spelling `write_to` produces, so `from_string(String(z))` returns
        `z`. Both parts may be omitted where they are implied:

        | text     | value    |
        |----------|----------|
        | `1+2i`   | `1+2i`   |
        | `3`      | `3+0i`   |
        | `2i`     | `0+2i`   |
        | `-i`     | `0-1i`   |
        | `1e-3+2i`| `0.001+2i` |

        The literal must contain no spaces. `from_string` on a matrix splits
        its text on whitespace as well as on commas, so `1 + 2i` would arrive
        here as three separate cells rather than one; the matrix printer emits
        no spaces inside a cell, which is what keeps the round trip closed.

        Args:
            value: The text to parse.

        Returns:
            The parsed value.

        Raises:
            Error: If the text is not a complex literal.
        """
        comptime PLUS = Byte(ord("+"))
        comptime MINUS = Byte(ord("-"))
        comptime E_LOWER = Byte(ord("e"))
        comptime E_UPPER = Byte(ord("E"))
        comptime I_LOWER = Byte(ord("i"))
        comptime I_UPPER = Byte(ord("I"))

        var text = String(value.strip())
        var n = text.byte_length()
        if n == 0:
            raise Error("empty complex literal")

        var bytes = text.as_bytes()

        # No trailing `i`, so the whole token is a real number.
        if bytes[n - 1] != I_LOWER and bytes[n - 1] != I_UPPER:
            return Self(Self.Real(atof(text)))

        # [Mojo Miji]
        # With the `i` dropped, what is left is either a lone coefficient
        # (`2`, `-`, `` for `2i`, `-i`, `i`) or a real part followed by a
        # signed one. The sign that separates them is the last `+` or `-`
        # that is neither the leading sign nor an exponent's --- `1e-3+2i`
        # has three of those characters and only the third divides it.
        var body = String(text[byte = 0 : n - 1])
        var m = body.byte_length()
        var split = -1
        var k = m - 1
        while k >= 1:
            var c = bytes[k]
            if (c == PLUS or c == MINUS) and (
                bytes[k - 1] != E_LOWER and bytes[k - 1] != E_UPPER
            ):
                split = k
                break
            k -= 1

        if split == -1:
            return Self(Self.Real(0), Self._coefficient(body))
        return Self(
            Self.Real(atof(String(body[byte=0:split]))),
            Self._coefficient(String(body[byte=split:])),
        )

    @staticmethod
    def _coefficient(text: String) raises -> Self.Real:
        """Reads the coefficient of the imaginary part.

        `i` and `-i` carry no digits, so an empty coefficient, `+`, and `-`
        stand for one and minus one rather than for a missing number.

        Args:
            text: The coefficient's text, sign included.

        Returns:
            The coefficient.

        Raises:
            Error: If the text is neither empty, a sign, nor a number.
        """
        if text.byte_length() == 0 or text == "+":
            return Self.Real(1)
        if text == "-":
            return Self.Real(-1)
        return Self.Real(atof(text))

    # ===-----------------------------------------------------------------===#
    # Equatable
    # ===-----------------------------------------------------------------===#

    @always_inline
    def __eq__(self, other: Self) -> Bool:
        """Returns whether both components are equal.

        Args:
            other: The value to compare against.

        Returns:
            True if the real and imaginary parts both match.
        """
        return self._v == other._v

    @always_inline
    def __ne__(self, other: Self) -> Bool:
        """Returns whether the two values differ.

        Args:
            other: The value to compare against.

        Returns:
            True if either component differs.
        """
        return not (self._v == other._v)

    # ===-----------------------------------------------------------------===#
    # Formatting
    # ===-----------------------------------------------------------------===#

    def write_to(self, mut writer: Some[Writer]):
        """Writes the value as `a+bi`.

        Args:
            writer: The object to write to.
        """
        # [Mojo Miji]
        # Not forwarded. The stdlib omits the imaginary part when it is zero
        # and writes the sign as part of the number, so a row of its output
        # reads `1.0 + 2.0i`, `3.0 + -1.0i`, `2.0`. Down a matrix column that
        # is three different shapes for the same type, and the last one is
        # indistinguishable from a real matrix. Every element here gets both
        # parts and one sign character.
        var im = self._v.im
        writer.write(self._v.re)
        writer.write("-" if im < 0 else "+")
        writer.write(abs(im), "i")


# ===----------------------------------------------------------------------===#
# Aliases
# ===----------------------------------------------------------------------===#
# [Mojo Miji]
# Named by component width, as the stdlib's `ComplexFloat64` is and as Mojo
# names every other scalar: `ComplexFloat64` is a pair of `Float64`. NumPy's
# `complex128` counts the bits of the pair instead, which reads as a wider
# component than it is. The short forms exist because an element type is
# written once per matrix and often twice per line.

comptime ComplexFloat32 = Complex[DType.float32]
"""A complex number with 32-bit components."""

comptime ComplexFloat64 = Complex[DType.float64]
"""A complex number with 64-bit components."""

comptime CFloat32 = ComplexFloat32
"""Short name for `ComplexFloat32`."""

comptime CFloat64 = ComplexFloat64
"""Short name for `ComplexFloat64`."""
