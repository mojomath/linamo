"""
Tests for the printed layout of a matrix.

These assert on exact lines rather than on substrings. Alignment is the point
of the module, and a substring check cannot see a column that has slipped by
one space.
"""

import std.testing as testing
import linamo as la


def test_decimal_points_line_up() raises:
    """A column pads either side of the dot, so the dots stand in one line."""
    var m = la.matrix[Float64](
        [[1457.2, 9.5, 1589.62], [3.25, 1626.8, 12.5], [1648.0, 1726.0, 1804.0]]
    )
    var lines = m.__str__().split("\n")
    testing.assert_equal(lines[0], "[[ 1457.2      9.5  1589.62 ]")
    testing.assert_equal(lines[1], " [    3.25  1626.8    12.5  ]")
    testing.assert_equal(lines[2], " [ 1648.0   1726.0  1804.0  ]]")


def test_rows_are_all_the_same_width() raises:
    """Padding the last column too is what keeps the right bracket straight."""
    var m = la.matrix[Float64]([[1.0, 22.0], [333.0, 4.0]])
    var lines = m.__str__().split("\n")
    testing.assert_equal(
        len(lines[1].codepoints()),
        len(lines[0].codepoints()) + 1,
        "only the closing `]]` should make the last line longer",
    )


def test_header_hides_the_layout_of_a_plain_matrix() raises:
    """A freshly built matrix has nothing to say about its strides."""
    var m = la.matrix[Float64]([[1.0, 2.0], [3.0, 4.0]])
    var text = String("")
    m.write_to(text)
    testing.assert_equal(text.split("\n")[0], "Matrix[float64] 2x2")


def test_header_shows_the_layout_of_a_strided_view() raises:
    """A view onto part of a matrix is exactly when the layout matters."""
    var m = la.matrix[Float64](
        [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
        ]
    )
    var v = m[1:3, 1:3]
    var text = String("")
    v.write_to(text)
    testing.assert_equal(
        text.split("\n")[0], "MatrixView[float64] 2x2, strides (4, 1), offset 5"
    )


def test_print_does_not_end_with_a_blank_line() raises:
    """The grid stops at the last bracket; `print` adds the one newline."""
    var m = la.matrix[Float64]([[1.0, 2.0], [3.0, 4.0]])
    var text = String("")
    m.write_to(text)
    testing.assert_true(
        text.endswith("]]"), "write_to should end at the closing bracket"
    )


def test_a_long_fraction_is_trimmed_and_marked() raises:
    """Digits after the point are cut; the mark says the reading is abridged."""
    var m = la.from_string[la.BDec]("[[1.0, 3.0]]")
    var q = m[0:1, 0:1].div(m[0:1, 1:2])
    var text = q.__str__()
    testing.assert_equal(text, "[[ 0.33333333… ]]")


def test_the_integer_part_is_never_trimmed() raises:
    """A trimmed magnitude would be a wrong reading, not a short one."""
    var m = la.from_string[la.BInt](
        "[[170141183460469231731687303715884105728]]"
    )
    testing.assert_equal(
        m.__str__(), "[[ 170141183460469231731687303715884105728 ]]"
    )


def test_a_large_matrix_elides_rows_and_columns() raises:
    """Past the element threshold both dimensions keep their edges only."""
    var m = la.zeros[Float64](40, 40)
    var lines = m.__str__().split("\n")
    testing.assert_equal(len(lines), 7, "three rows, the mark, three rows")
    testing.assert_equal(lines[3], " ...")
    testing.assert_true(
        "..." in lines[0], "the row should be elided as well as the column"
    )


def test_a_small_matrix_is_printed_whole() raises:
    """Under the threshold and inside the line, nothing is left out."""
    var m = la.zeros[Float64](8, 8)
    var lines = m.__str__().split("\n")
    testing.assert_equal(len(lines), 8)
    testing.assert_true("..." not in lines[0], "8x8 fits and should print all")


def test_wide_elements_keep_the_minimum_column_count() raises:
    """The floor answers a wide element type with a row worth reading.

    Six forty-digit integers cannot be made to fit a line however few are
    shown, so the width rule bottoms out at `MIN_COLS_SHOWN` rather than at
    the one column it would otherwise reach.
    """
    var digits = String("1234567890123456789012345678901234567890")
    var row = String("[")
    for i in range(6):
        row += ", " if i > 0 else ""
        row += digits
    row += "]"
    var m = la.from_string[la.BInt](String("[", row, ", ", row, "]"))
    var lines = m.__str__().split("\n")
    testing.assert_true(
        "..." in lines[0], "six forty-digit columns cannot fit a line"
    )
    testing.assert_equal(
        lines[0].count(digits), 3, "the floor keeps three columns"
    )


def test_two_columns_are_never_split_by_a_mark() raises:
    """Below the floor every column is shown, so there is nothing to elide."""
    var m = la.from_string[la.BInt](
        "[[8320987112741390144276341183223364380754172606361245952449277696409600000000000000,"
        " 0], [0, 1]]"
    )
    var lines = m.__str__().split("\n")
    testing.assert_true(
        "..." not in lines[0],
        "a 2x2 shows both columns however wide they are",
    )


def test_a_single_row_has_no_newline() raises:
    """One row is one line, brackets and all."""
    var m = la.matrix[Int64]([[10, 20, 30]])
    var text = m.__str__()
    testing.assert_equal(text, "[[ 10  20  30 ]]")


def test_a_static_matrix_prints_like_the_others() raises:
    """`StaticMatrix` shares the layout and differs only in its header."""
    var m = la.smatrix[2, 2, Float64]([[1.0, 2.0], [3.0, 4.0]])
    var text = String("")
    m.write_to(text)
    testing.assert_equal(text.split("\n")[0], "StaticMatrix[float64] 2x2")
    testing.assert_equal(text.split("\n")[1], "[[ 1.0  2.0 ]")


def main() raises:
    testing.TestSuite.discover_tests[__functions_in_module()]().run()
