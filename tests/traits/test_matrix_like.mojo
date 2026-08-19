"""
Keeps `linamo.traits.matrix_like` compiling.

No type conforms to `MatrixLike` and nothing in the library is generic over it,
so nothing else imports the module and the compiler would never look at it. The
file is kept because the trait is a reasonable idea that is merely unused --- a
later version could give it the read-only algorithms `Matrix` and `MatrixView`
duplicate today --- and an unchecked module rots. Importing it here is the
whole test: a syntax or signature error in the trait fails the suite.
"""

import std.testing as testing
from linamo.traits.matrix_like import MatrixLike


def test_matrix_like_module_compiles() raises:
    """The import above is the assertion; this body only proves it ran."""
    testing.assert_true(True)


def main() raises:
    var suite = testing.TestSuite.discover_tests[__functions_in_module()]()
    suite^.run()
