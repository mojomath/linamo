"""
Tests for Matrix string representation and writing.
"""

import std.testing as testing
import matmojo as mm


def test_matrix_str_basic() raises:
    """Test __str__ produces expected tab-separated format.

    Note: since Mojo 1.0.0 removed `Stringable`, `String(mat)` routes through
    `write_to` (the rich representation).  `__str__` -- the bare tab-separated
    grid -- must now be called explicitly.
    """
    var mat = mm.matrix[DType.float64]([[1.0, 2.0], [3.0, 4.0]])
    var s = mat.__str__()
    # __str__ produces tab-separated values, rows separated by newlines
    testing.assert_true("1.0" in s, "String should contain element 1.0")
    testing.assert_true("4.0" in s, "String should contain element 4.0")


def test_matrix_write_to_includes_metadata() raises:
    """Test write_to includes dtype and shape info."""
    var mat = mm.matrix[DType.float64]([[1.0, 2.0], [3.0, 4.0]])
    # print() calls write_to, which includes metadata
    var s = String("")
    mat.write_to(s)
    testing.assert_true("Matrix" in s, "write_to should include 'Matrix'")
    testing.assert_true("float64" in s, "write_to should include dtype")
    testing.assert_true("2x2" in s, "write_to should include shape")


def test_matrix_str_single_row() raises:
    """Test __str__ for a single-row matrix."""
    var mat = mm.matrix[DType.int64]([[10, 20, 30]])
    var s = mat.__str__()
    testing.assert_true("10" in s, "Should contain 10")
    testing.assert_true("20" in s, "Should contain 20")
    testing.assert_true("30" in s, "Should contain 30")
    # Single row should have no newlines
    testing.assert_true(
        "\n" not in s,
        "Single row string should not contain newline",
    )


def main() raises:
    testing.TestSuite.discover_tests[__functions_in_module()]().run()
