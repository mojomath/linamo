"""
Tests for the layout invariant that `Matrix` constructors assert.

A matrix's shape, its two strides and the length of its buffer are one
invariant bundle: indexing computes `row * row_stride + col * col_stride`, so
together they decide which buffer slots `m[i, j]` reaches. Two properties have
to hold, and each catches a distinct failure the other misses:

1. `layout_is_dense` --- distinct indices reach distinct slots. A zero
   `row_stride` makes every row the same row, so `m[0, 0] = 5` also writes
   `m[1, 0]`. Positive strides are not enough on their own: `(1, 1)` on a 2x2
   sends both `[0, 1]` and `[1, 0]` to offset 1.
2. `layout_fits_buffer` --- no index runs off the end. A C-major `(2, 1)` on a
   3x2 is perfectly dense and still overruns a four-element buffer.

The constructors state these with `debug_assert`, which aborts rather than
raises and so cannot be caught by `assert_raises`. The predicates are what a
test can see; the assertions wire them in, and both are exercised whenever the
suite runs, since `tests/test_all.sh` passes `-D ASSERT=all`.
"""

import std.testing as testing
import linamo as la
from linamo.utils.indexing import layout_fits_buffer, layout_is_dense


# ===----------------------------------------------------------------------===#
# Distinct indices reach distinct slots
# ===----------------------------------------------------------------------===#


def test_dense_accepts_c_major() raises:
    """A 3x2 C-major matrix has strides (ncols, 1)."""
    testing.assert_true(layout_is_dense(3, 2, 2, 1))


def test_dense_accepts_f_major() raises:
    """A 3x2 F-major matrix has strides (1, nrows)."""
    testing.assert_true(layout_is_dense(3, 2, 1, 3))


def test_dense_rejects_zero_row_stride() raises:
    """`row_stride == 0` aliases every row onto row 0."""
    testing.assert_false(layout_is_dense(3, 2, 0, 1))


def test_dense_rejects_zero_col_stride() raises:
    """`col_stride == 0` aliases every column onto column 0."""
    testing.assert_false(layout_is_dense(3, 2, 1, 0))


def test_dense_rejects_negative_stride() raises:
    """An owning matrix never walks its own buffer backwards."""
    testing.assert_false(layout_is_dense(3, 2, -2, 1))


def test_dense_rejects_positive_but_aliasing_strides() raises:
    """Positivity alone is not enough: `(1, 1)` on a 2x2 collides.

    Both `[0, 1]` and `[1, 0]` land on offset 1. This is the case that makes
    the C-/F-major test necessary rather than merely convenient.
    """
    testing.assert_false(layout_is_dense(2, 2, 1, 1))


def test_dense_rejects_padded_strides() raises:
    """Padding between rows is legal for a view, not for an owning matrix."""
    testing.assert_false(layout_is_dense(3, 2, 4, 1))


def test_dense_accepts_empty() raises:
    """An empty matrix reaches no offset, so any strides are vacuously fine."""
    testing.assert_true(layout_is_dense(0, 2, 0, 0))
    testing.assert_true(layout_is_dense(3, 0, 0, 0))


def test_dense_accepts_single_row_and_column() raises:
    """A vector satisfies both the C- and the F-major test."""
    testing.assert_true(layout_is_dense(1, 4, 4, 1))
    testing.assert_true(layout_is_dense(4, 1, 1, 4))


# ===----------------------------------------------------------------------===#
# No index runs off the end of the buffer
# ===----------------------------------------------------------------------===#


def test_fits_accepts_an_exact_buffer() raises:
    """A 3x2 C-major matrix reaches offset 5, so six elements is exact."""
    testing.assert_true(layout_fits_buffer(3, 2, 2, 1, 6))


def test_fits_rejects_a_buffer_one_short() raises:
    """Five elements leaves `m[2, 1]` outside the buffer."""
    testing.assert_false(layout_fits_buffer(3, 2, 2, 1, 5))


def test_fits_rejects_an_overrunning_stride() raises:
    """A dense-looking stride can still walk far past the end."""
    testing.assert_false(layout_fits_buffer(3, 2, 100, 1, 6))


def test_fits_is_independent_of_denseness() raises:
    """The two checks are not redundant; each catches what the other misses.

    `(2, 1)` on a 3x2 is dense and overruns a four-element buffer, while
    `(0, 1)` fits a six-element buffer comfortably and aliases.
    """
    testing.assert_true(layout_is_dense(3, 2, 2, 1))
    testing.assert_false(layout_fits_buffer(3, 2, 2, 1, 4))

    testing.assert_false(layout_is_dense(3, 2, 0, 1))
    testing.assert_true(layout_fits_buffer(3, 2, 0, 1, 6))


def test_fits_accepts_empty() raises:
    """An empty matrix fits any buffer, including an empty one."""
    testing.assert_true(layout_fits_buffer(0, 2, 2, 1, 0))
    testing.assert_true(layout_fits_buffer(3, 0, 2, 1, 0))


# ===----------------------------------------------------------------------===#
# What the library itself builds
# ===----------------------------------------------------------------------===#


def test_library_matrices_satisfy_the_invariant() raises:
    """Every construction route produces a layout that passes both checks.

    This is the property the `debug_assert`s turn into a suite-wide check: if
    any routine ever produced a padded or aliasing owning matrix, the whole
    suite would abort under `-D ASSERT=all` rather than fail here.
    """
    var c = la.matrix[DType.float64]([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    testing.assert_true(
        layout_is_dense(c.nrows(), c.ncols(), c.row_stride(), c.col_stride())
    )
    testing.assert_true(
        layout_fits_buffer(
            c.nrows(), c.ncols(), c.row_stride(), c.col_stride(), len(c.data())
        )
    )

    var f = la.matrix[DType.float64](
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], order="F"
    )
    testing.assert_true(
        layout_is_dense(f.nrows(), f.ncols(), f.row_stride(), f.col_stride())
    )
    testing.assert_true(
        layout_fits_buffer(
            f.nrows(), f.ncols(), f.row_stride(), f.col_stride(), len(f.data())
        )
    )

    var t = la.transpose(c)
    testing.assert_true(
        layout_is_dense(t.nrows(), t.ncols(), t.row_stride(), t.col_stride())
    )
    testing.assert_true(
        layout_fits_buffer(
            t.nrows(), t.ncols(), t.row_stride(), t.col_stride(), len(t.data())
        )
    )


def main() raises:
    var suite = testing.TestSuite.discover_tests[__functions_in_module()]()
    suite^.run()
