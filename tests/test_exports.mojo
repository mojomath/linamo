"""
Tests that the public surface is reachable as `la.<name>`.

Every other test file imports the routine it exercises from its module path,
which is why none of them can see this class of break: a routine can be
correct, tested and still unreachable for a user who wrote only
`import linamo as la`. These tests call each name through the package alias
and nothing else, so a name dropped from `linamo/__init__.mojo` fails here
even though its own tests still pass.

The assertions are deliberately thin. What is under test is that the name
resolves and the call type-checks; the behaviour behind it belongs to the
module's own test file.
"""

import std.testing as testing
import linamo as la


# ===----------------------------------------------------------------------===#
# Closeness
# ===----------------------------------------------------------------------===#
# No operator spells these, so the package alias is the only spelling a user
# has for them short of a module-path import.


def test_isclose_and_allclose_are_exported() raises:
    """Test that the approximate comparisons resolve as `la.<name>`."""
    var a = la.matrix[Float64]([[0.1, 0.2], [0.3, 0.4]])
    var b = (a * 3.0) / 3.0

    testing.assert_false(la.all(a == b), "the round trip should lose bits")
    testing.assert_true(la.allclose(a, b))
    testing.assert_true(la.all(la.isclose(a, b)))


def test_scalar_closeness_forms_are_exported() raises:
    """Test that the closeness forms taking a single value resolve."""
    var zeros = la.zeros[Float64](2, 2)

    testing.assert_true(la.scalar_allclose(zeros, 0.0))
    testing.assert_true(la.all(la.scalar_isclose(zeros, 0.0)))


# ===----------------------------------------------------------------------===#
# Logical connectives
# ===----------------------------------------------------------------------===#


def test_logical_connectives_are_exported() raises:
    """Test that the binary connectives resolve as `la.<name>`."""
    var t = la.matrix[Float64]([[1.0, 0.0]])
    var u = la.matrix[Float64]([[1.0, 1.0]])

    testing.assert_true(la.logical_and(t, u)[0, 0])
    testing.assert_false(la.logical_and(t, u)[0, 1])
    testing.assert_true(la.logical_or(t, u)[0, 1])
    testing.assert_true(la.logical_xor(t, u)[0, 1])


def test_logical_not_is_exported() raises:
    """Test that the unary connective resolves as `la.<name>`."""
    var t = la.matrix[Float64]([[1.0, 0.0]])

    testing.assert_false(la.logical_not(t)[0, 0])
    testing.assert_true(la.logical_not(t)[0, 1])


def test_scalar_logical_forms_are_exported() raises:
    """Test that the connectives taking a single value resolve."""
    var t = la.matrix[Float64]([[1.0, 0.0]])

    testing.assert_true(la.scalar_logical_and(t, 1.0)[0, 0])
    testing.assert_true(la.scalar_logical_or(t, 0.0)[0, 0])
    testing.assert_true(la.scalar_logical_xor(t, 1.0)[0, 1])


# ===----------------------------------------------------------------------===#
# Boolean reductions
# ===----------------------------------------------------------------------===#


def test_all_and_any_are_exported() raises:
    """Test that the mask reductions resolve as `la.<name>`."""
    var m = la.matrix[Float64]([[1.0, -1.0]])

    testing.assert_false(la.all(m > 0.0))
    testing.assert_true(la.any(m > 0.0))


# ===----------------------------------------------------------------------===#
# The element-wise three
# ===----------------------------------------------------------------------===#
# `*` is the matrix product, which leaves these without an operator. Their
# siblings `add` and `sub` are not exported on purpose --- `+` and `-` spell
# them --- so this pins the rule from the side that is easy to get wrong.


def test_elementwise_routines_are_exported() raises:
    """Test that the routines no operator spells resolve as `la.<name>`."""
    var a = la.matrix[Float64]([[2.0, 3.0]])
    var b = la.matrix[Float64]([[4.0, 5.0]])

    testing.assert_equal(la.mul(a, b)[0, 0], 8.0)
    testing.assert_equal(la.div(b, a)[0, 0], 2.0)
    testing.assert_equal(la.pow(a, b)[0, 0], 16.0)


def main() raises:
    testing.TestSuite.discover_tests[__functions_in_module()]().run()
