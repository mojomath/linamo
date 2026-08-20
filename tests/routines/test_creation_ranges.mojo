"""
Tests for the Phase 5.5 creation routines: empty, the *_like family, arange,
linspace, from_list and from_string.
"""

import std.testing as testing
import linamo as la
from linamo.routines.creation import (
    arange,
    empty,
    empty_like,
    from_list,
    from_string,
    full_like,
    linspace,
    ones_like,
    zeros_like,
)


# ===----------------------------------------------------------------------===#
# empty
# ===----------------------------------------------------------------------===#


def test_empty_shape() raises:
    """Test that empty produces the requested shape and layout."""
    var mat = empty[Float64](3, 4)
    testing.assert_equal(mat.nrows(), 3)
    testing.assert_equal(mat.ncols(), 4)
    testing.assert_true(mat.is_c_contiguous())


def test_empty_is_writable() raises:
    """Test that every element of an empty matrix can be written and read."""
    var mat = empty[Int32](2, 2)
    for i in range(2):
        for j in range(2):
            mat[i, j] = Int32(i * 2 + j)
    testing.assert_equal(mat[0, 0], Int32(0))
    testing.assert_equal(mat[1, 1], Int32(3))


# ===----------------------------------------------------------------------===#
# zeros_like / ones_like / full_like / empty_like
# ===----------------------------------------------------------------------===#


def test_zeros_like() raises:
    """Test that zeros_like copies the shape and fills with zeros."""
    var src = la.matrix[Float64]([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    var mat = zeros_like(src)
    testing.assert_equal(mat.nrows(), 2)
    testing.assert_equal(mat.ncols(), 3)
    for i in range(2):
        for j in range(3):
            testing.assert_equal(mat[i, j], 0.0)


def test_ones_like() raises:
    """Test that ones_like copies the shape and fills with ones."""
    var src = la.matrix[Float64]([[1.0, 2.0], [3.0, 4.0]])
    var mat = ones_like(src)
    testing.assert_equal(mat[0, 0], 1.0)
    testing.assert_equal(mat[1, 1], 1.0)


def test_full_like() raises:
    """Test that full_like copies the shape and fills with the given value."""
    var src = la.matrix[Float64]([[1.0, 2.0], [3.0, 4.0]])
    var mat = full_like(src, 7.5)
    testing.assert_equal(mat[0, 1], 7.5)
    testing.assert_equal(mat[1, 0], 7.5)


def test_empty_like_shape() raises:
    """Test that empty_like copies the shape."""
    var src = la.matrix[Int64]([[1, 2, 3], [4, 5, 6]])
    var mat = empty_like(src)
    testing.assert_equal(mat.nrows(), 2)
    testing.assert_equal(mat.ncols(), 3)


def test_like_accepts_a_view() raises:
    """Test that the *_like family takes a view and copies the view's shape."""
    var src = la.matrix[Float64](
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
    )
    var mat = zeros_like(src[0:2, 1:3])
    testing.assert_equal(mat.nrows(), 2)
    testing.assert_equal(mat.ncols(), 2)


def test_like_result_is_c_contiguous_from_f_input() raises:
    """Test that a *_like result is C-contiguous even for an F-order input."""
    var src = la.matrix[Float64]([[1.0, 2.0], [3.0, 4.0]], "F")
    testing.assert_true(src.is_f_contiguous())
    var mat = zeros_like(src)
    testing.assert_true(mat.is_c_contiguous())


# ===----------------------------------------------------------------------===#
# arange
# ===----------------------------------------------------------------------===#


def test_arange_stop_only() raises:
    """Test the single-argument form, which starts at zero with step one."""
    var mat = arange[Float64](5.0)
    testing.assert_equal(mat.nrows(), 1)
    testing.assert_equal(mat.ncols(), 5)
    for k in range(5):
        testing.assert_equal(mat[0, k], Float64(k))


def test_arange_start_stop_step() raises:
    """Test a fractional step over a half-open interval."""
    var mat = arange[Float64](1.0, 2.0, 0.25)
    testing.assert_equal(mat.ncols(), 4)
    testing.assert_equal(mat[0, 0], 1.0)
    testing.assert_equal(mat[0, 3], 1.75)


def test_arange_excludes_stop() raises:
    """Test that stop is excluded when it lands exactly on a step."""
    var mat = arange[Int64](0, 6, 2)
    testing.assert_equal(mat.ncols(), 3)
    testing.assert_equal(mat[0, 2], Int64(4))


def test_arange_negative_step() raises:
    """Test that a negative step counts down."""
    var mat = arange[Int64](10, 0, -3)
    testing.assert_equal(mat.ncols(), 4)
    testing.assert_equal(mat[0, 0], Int64(10))
    testing.assert_equal(mat[0, 3], Int64(1))


def test_arange_zero_step_raises() raises:
    """Test that a zero step raises."""
    var raised = False
    try:
        var _m = arange[Float64](0.0, 5.0, 0.0)
    except:
        raised = True
    testing.assert_true(raised, "A zero step should raise")


def test_arange_empty_range_raises() raises:
    """Test that a range containing no values raises."""
    var raised = False
    try:
        var _m = arange[Float64](5.0, 0.0, 1.0)
    except:
        raised = True
    testing.assert_true(raised, "An empty range should raise")


# ===----------------------------------------------------------------------===#
# linspace
# ===----------------------------------------------------------------------===#


def test_linspace_endpoint() raises:
    """Test that the endpoint is included and hit exactly by default."""
    var mat = linspace[Float64](0.0, 1.0, 5)
    testing.assert_equal(mat.nrows(), 1)
    testing.assert_equal(mat.ncols(), 5)
    testing.assert_equal(mat[0, 0], 0.0)
    testing.assert_equal(mat[0, 2], 0.5)
    testing.assert_equal(mat[0, 4], 1.0)


def test_linspace_no_endpoint() raises:
    """Test that endpoint=False makes the interval half-open."""
    var mat = linspace[Float64](0.0, 1.0, 5, endpoint=False)
    testing.assert_equal(mat.ncols(), 5)
    testing.assert_equal(mat[0, 0], 0.0)
    testing.assert_almost_equal(mat[0, 1], 0.2)
    testing.assert_true(mat[0, 4] < 1.0)


def test_linspace_single_value() raises:
    """Test that num=1 yields just the start value."""
    var mat = linspace[Float64](3.0, 9.0, 1)
    testing.assert_equal(mat.ncols(), 1)
    testing.assert_equal(mat[0, 0], 3.0)


def test_linspace_descending() raises:
    """Test that stop below start counts down."""
    var mat = linspace[Float64](1.0, 0.0, 3)
    testing.assert_equal(mat[0, 0], 1.0)
    testing.assert_equal(mat[0, 1], 0.5)
    testing.assert_equal(mat[0, 2], 0.0)


def test_linspace_zero_num_raises() raises:
    """Test that num below 1 raises."""
    var raised = False
    try:
        var _m = linspace[Float64](0.0, 1.0, 0)
    except:
        raised = True
    testing.assert_true(raised, "num=0 should raise")


# ===----------------------------------------------------------------------===#
# from_list
# ===----------------------------------------------------------------------===#


def test_from_list_c_order() raises:
    """Test that a flat list is read row-first under C order."""
    var mat = from_list[Float64]([1.0, 2.0, 3.0, 4.0], 2, 2)
    testing.assert_equal(mat[0, 0], 1.0)
    testing.assert_equal(mat[0, 1], 2.0)
    testing.assert_equal(mat[1, 0], 3.0)


def test_from_list_f_order() raises:
    """Test that a flat list is read column-first under F order."""
    var mat = from_list[Float64]([1.0, 2.0, 3.0, 4.0], 2, 2, "F")
    testing.assert_equal(mat[0, 0], 1.0)
    testing.assert_equal(mat[1, 0], 2.0)
    testing.assert_equal(mat[0, 1], 3.0)


def test_from_list_length_mismatch_raises() raises:
    """Test that a length that does not match the shape raises."""
    var raised = False
    try:
        var _m = from_list[Float64]([1.0, 2.0, 3.0], 2, 2)
    except:
        raised = True
    testing.assert_true(raised, "A length mismatch should raise")


# ===----------------------------------------------------------------------===#
# from_string
# ===----------------------------------------------------------------------===#


def test_from_string_nested_deduces_shape() raises:
    """Test that nested brackets give the shape."""
    var mat = from_string[Float64]("[[1, 2, 3], [4, 5.5, 6]]")
    testing.assert_equal(mat.nrows(), 2)
    testing.assert_equal(mat.ncols(), 3)
    testing.assert_equal(mat[1, 1], 5.5)


def test_from_string_whitespace_separated() raises:
    """Test that whitespace works as an element separator."""
    var mat = from_string[Float64]("[[1 2 3]\n [4 5 6]]")
    testing.assert_equal(mat.nrows(), 2)
    testing.assert_equal(mat[1, 2], 6.0)


def test_from_string_single_row() raises:
    """Test that an unnested literal is one row."""
    var mat = from_string[Float64]("1 2 3")
    testing.assert_equal(mat.nrows(), 1)
    testing.assert_equal(mat.ncols(), 3)
    testing.assert_equal(mat[0, 2], 3.0)


def test_from_string_single_bracket_is_one_row() raises:
    """Test that a singly-bracketed literal is also one row."""
    var mat = from_string[Float64]("[1, 2, 3]")
    testing.assert_equal(mat.nrows(), 1)
    testing.assert_equal(mat.ncols(), 3)


def test_from_string_negative_and_exponent() raises:
    """Test that signs and exponents parse."""
    var mat = from_string[Float64]("[[-1.5, 2e2]]")
    testing.assert_equal(mat[0, 0], -1.5)
    testing.assert_equal(mat[0, 1], 200.0)


def test_from_string_integer_dtype() raises:
    """Test parsing into an integer dtype."""
    var mat = from_string[Int32]("[[1, 2], [3, 4]]")
    testing.assert_equal(mat[1, 1], Int32(4))


def test_from_string_with_shape_ignores_nesting() raises:
    """Test that the shaped overload reads every element in order."""
    var mat = from_string[Int32]("[1, 2, 3, 4]", 2, 2)
    testing.assert_equal(mat.nrows(), 2)
    testing.assert_equal(mat[1, 0], Int32(3))


def test_from_string_with_shape_f_order() raises:
    """Test that the shaped overload honours F order."""
    var mat = from_string[Float64]("[[1,2],[3,4]]", 2, 2, "F")
    testing.assert_equal(mat[0, 0], 1.0)
    testing.assert_equal(mat[1, 0], 2.0)


def test_from_string_ragged_raises() raises:
    """Test that rows of unequal length raise."""
    var raised = False
    try:
        var _m = from_string[Float64]("[[1, 2], [3]]")
    except:
        raised = True
    testing.assert_true(raised, "A ragged literal should raise")


def test_from_string_bad_token_raises() raises:
    """Test that an unparseable token raises."""
    var raised = False
    try:
        var _m = from_string[Float64]("[[1, x], [3, 4]]")
    except:
        raised = True
    testing.assert_true(raised, "A non-numeric token should raise")


def test_from_string_unbalanced_raises() raises:
    """Test that unbalanced brackets raise."""
    var raised = False
    try:
        var _m = from_string[Float64]("[[1, 2], [3, 4]")
    except:
        raised = True
    testing.assert_true(raised, "Unbalanced brackets should raise")


def test_from_string_too_deep_raises() raises:
    """Test that nesting beyond two levels raises."""
    var raised = False
    try:
        var _m = from_string[Float64]("[[[1, 2]], [[3, 4]]]")
    except:
        raised = True
    testing.assert_true(raised, "Three levels of nesting should raise")


def test_from_string_empty_raises() raises:
    """Test that a literal with no elements raises."""
    var raised = False
    try:
        var _m = from_string[Float64]("[]")
    except:
        raised = True
    testing.assert_true(raised, "An empty literal should raise")


def main() raises:
    testing.TestSuite.discover_tests[__functions_in_module()]().run()
