"""
A tour of `MatrixView`, the non-owning window onto a matrix.

A view is a shape, a pair of strides and an offset over somebody else's
storage. Creating one copies nothing, so slicing a 10000x10000 matrix costs
the same as slicing a 2x2 one, and writing through a view writes into the
matrix it came from.

The rule worth learning first: **slicing always yields a read-only view.**
`m[0:2, :]`, `m.view()`, `m.rows()` and iteration all hand back something you
cannot write through, no matter how `m` was bound. That is deliberate - a
mutable view is an exclusive borrow, and Mojo will not pass two of them into
one call, so if slicing inherited mutability then `a[0:1, :] - a[1:2, :]`
would not compile. Writing is the rarer case, so it gets the explicit
spelling: `view_mut`, `fill`, `assign` and friends from
`linamo.routines.mutation`.

Run with:

```bash
pixi run mojo run -I src examples/matrix_view.mojo
```
"""

from linamo.prelude import *
from linamo.routines.mutation import (
    assign,
    cols_mut,
    fill,
    rows_mut,
    store,
    view_mut,
)


def main() raises:
    what_a_view_is()
    slicing_forms()
    view_on_view()
    getitem()
    layout_of_a_view()
    read_only_by_default()
    writing_through_a_view()
    setitem_regions()
    mutable_iteration()
    demoting_with_as_imm()
    iteration()
    arithmetic()
    materialisation()


def _grid[dtype: DType](nrows: Int, ncols: Int) raises -> la.Matrix[dtype]:
    """Builds an `nrows x ncols` matrix whose entries read `row * 10 + col`."""
    var m = la.zeros[dtype](nrows, ncols)
    for i in range(nrows):
        for j in range(ncols):
            m[i, j] = Scalar[dtype](i * 10 + j)
    return m^


def _banner(title: String):
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


# ===----------------------------------------------------------------------=== #
# What a view is
# ===----------------------------------------------------------------------=== #


def what_a_view_is() raises:
    _banner("WHAT A VIEW IS")

    var m = _grid[int64](5, 6)
    print("A 5x6 matrix. Entry (i, j) reads as i*10 + j:\n", m)

    # `m[rows, cols]` builds a view. Nothing is copied: `v` is four integers
    # and a pointer into `m`.
    var v = m[1:4, 2:5]
    print("m[1:4, 2:5] - a 3x3 window:\n", v)

    # `m.view()` is the whole matrix as a view, i.e. the same as `m[:, :]`.
    print("m.view() is m[:, :]:\n", m.view())

    # The view sees `m`'s storage, so a change to `m` shows up in the view.
    m[1, 2] = -999
    print("After m[1, 2] = -999, the same view now reads:\n", v)


# ===----------------------------------------------------------------------=== #
# Slicing forms
# ===----------------------------------------------------------------------=== #


def slicing_forms() raises:
    _banner("SLICING FORMS")

    var m = _grid[int64](6, 8)
    print("A 6x8 matrix:\n", m)

    print("m[:, :] - everything:\n", m[:, :])
    print("m[2:4, 1:4] - a block:\n", m[2:4, 1:4])

    # An omitted bound runs to the edge.
    print("m[3:, :] - from row 3 to the bottom:\n", m[3:, :])
    print("m[:2, :] - the first two rows:\n", m[:2, :])
    print("m[:, 5:] - from column 5 to the right edge:\n", m[:, 5:])
    print("m[:, :3] - the first three columns:\n", m[:, :3])

    # A step selects every k-th row or column. The result is a strided view,
    # still with no copying.
    print("m[::2, :] - every other row:\n", m[::2, :])
    print("m[:, ::3] - every third column:\n", m[:, ::3])
    print("m[1::2, 1::3] - both at once:\n", m[1::2, 1::3])
    print("m[0:6:3, 0:8:4] - explicit start, stop and step:\n", m[0:6:3, 0:8:4])

    # A single row or column is just a slice one wide. There is no
    # `m[1, :]` overload: mixing an `Int` and a `Slice` would have to decide
    # whether to drop the axis, and Linamo keeps every view two-dimensional.
    print("m[2:3, :] - row 2 as a 1x8 view:\n", m[2:3, :])
    print("m[:, 4:5] - column 4 as a 6x1 view:\n", m[:, 4:5])

    # Negative bounds count from the end, as in Python.
    print("m[-2:, :] - the last two rows:\n", m[-2:, :])
    print("m[:, -3:] - the last three columns:\n", m[:, -3:])
    print("m[1:-1, 1:-1] - drop the border:\n", m[1:-1, 1:-1])

    # An empty selection is legal and gives a view with a zero dimension.
    var empty = m[2:2, :]
    print("m[2:2, :] is empty:", empty.nrows, "x", empty.ncols)


# ===----------------------------------------------------------------------=== #
# View on view
# ===----------------------------------------------------------------------=== #


def view_on_view() raises:
    _banner("VIEW ON VIEW")

    var m = _grid[int64](8, 10)
    print("An 8x10 matrix:\n", m)

    # Slicing a view slices the view's own coordinates, not the parent's.
    var v1 = m[1::2, 0:9:3]
    print("v1 = m[1::2, 0:9:3] - rows 1,3,5,7 and columns 0,3,6:\n", v1)

    var v2 = v1[1::2, 1:]
    print("v2 = v1[1::2, 1:] - v1's rows 1,3 and columns 1,2:\n", v2)

    # The strides multiply through, so an arbitrarily deep chain is still a
    # single stride computation at access time.
    print(
        "v2 strides: row =",
        v2.get_row_stride(),
        " col =",
        v2.get_col_stride(),
        " offset =",
        v2.get_offset(),
    )
    print("v2[0, 0] is m[3, 3]:", v2[0, 0], "==", m[3, 3])

    # A third level behaves the same way.
    var v3 = v2[:, 1:]
    print("v3 = v2[:, 1:]:\n", v3)


# ===----------------------------------------------------------------------=== #
# Element access
# ===----------------------------------------------------------------------=== #


def getitem() raises:
    _banner("GETITEM")

    var m = _grid[int64](6, 8)
    var v = m[1:5:2, 2:8:2]
    print("v = m[1:5:2, 2:8:2]:\n", v)

    # Indices are the view's own, counted from its top-left corner.
    print("v[0, 0] =", v[0, 0], " (which is m[1, 2] =", m[1, 2], ")")
    print("v[1, 2] =", v[1, 2], " (which is m[3, 6] =", m[3, 6], ")")

    print("Shape:", v.get_nrows(), "x", v.get_ncols(), " size:", v.get_size())
    print("len(v) - the row count, matching iteration:", len(v))

    # `get_unsafe` skips the bounds check; only `-D ASSERT=all` will catch a
    # bad index.
    print("v.get_unsafe(1, 1) =", v.get_unsafe(1, 1))

    # SIMD loads work on views too. A strided view cannot be loaded in one
    # instruction, so `load` gathers instead - the answer is the same and only
    # the speed differs. The width is a SIMD lane count, so it has to be a
    # power of two.
    print("v.load[2](0, 0) =", v.load[2](0, 0))
    print(
        "m.view().load[4](0, 0) - contiguous, a single vector load:",
        m.view().load[4](0, 0),
    )


# ===----------------------------------------------------------------------=== #
# The layout of a view
# ===----------------------------------------------------------------------=== #


def layout_of_a_view() raises:
    _banner("LAYOUT OF A VIEW")

    var m = _grid[int64](6, 8)

    var whole = m.view()
    print(
        "m.view():          strides",
        whole.get_row_stride(),
        whole.get_col_stride(),
        " offset",
        whole.get_offset(),
        " c_contiguous",
        whole.is_c_contiguous(),
    )

    var block = m[2:5, 1:4]
    print(
        "m[2:5, 1:4]:       strides",
        block.get_row_stride(),
        block.get_col_stride(),
        " offset",
        block.get_offset(),
        " c_contiguous",
        block.is_c_contiguous(),
    )

    var one_row = m[2:3, :]
    print(
        "m[2:3, :]:         strides",
        one_row.get_row_stride(),
        one_row.get_col_stride(),
        " offset",
        one_row.get_offset(),
        " row_contiguous",
        one_row.is_row_contiguous(),
    )

    var strided = m[::2, ::2]
    print(
        "m[::2, ::2]:       strides",
        strided.get_row_stride(),
        strided.get_col_stride(),
        " offset",
        strided.get_offset(),
        " c_contiguous",
        strided.is_c_contiguous(),
    )

    # A block of a row-major matrix is not contiguous, but each of its rows
    # is - which is exactly the case the SIMD kernels dispatch on.
    print("m[2:5, 1:4].is_row_contiguous():", block.is_row_contiguous())


# ===----------------------------------------------------------------------=== #
# Read-only by default
# ===----------------------------------------------------------------------=== #


def read_only_by_default() raises:
    _banner("READ-ONLY BY DEFAULT")

    var m = _grid[int64](4, 4)

    # `.mut` is the view's mutability, carried as a compile-time parameter.
    # Every implicit path gives False, even though `m` is a `var`.
    print("m[0:2, 0:2].mut :", m[0:2, 0:2].mut)
    print("m.view().mut    :", m.view().mut)

    # The explicit path gives True.
    var w = view_mut(m, Slice(0, 2), Slice(0, 2))
    print("view_mut(m, ...).mut :", w.mut)

    # And a slice *of* a mutable view is read-only again, for the same reason:
    # you should not be able to reach a second mutable window by accident.
    print("view_mut(m, ...)[0:1, 0:1].mut :", w[0:1, 0:1].mut)
    print(
        "view_mut(w, ...).mut - the explicit path again:",
        view_mut(w, Slice(0, 1), Slice(0, 1)).mut,
    )

    # This is the expression the rule exists to protect. Two read-only views
    # of the same matrix in one expression: fine.
    var diff = m[0:1, :] - m[1:2, :]
    print("m[0:1, :] - m[1:2, :] - two views of one matrix:\n", diff)


# ===----------------------------------------------------------------------=== #
# Writing through a view
# ===----------------------------------------------------------------------=== #


def writing_through_a_view() raises:
    _banner("WRITING THROUGH A VIEW")

    var m = la.zeros[int64](5, 6)
    print("A 5x6 matrix of zeros:\n", m)

    # A mutable view writes straight into the parent's storage.
    var w = view_mut(m, Slice(1, 4), Slice(1, 5))
    w[0, 0] = 1
    w[1, 1] = 2
    w[2, 2] = 3
    print(
        "After writing to w[0,0], w[1,1], w[2,2] of view_mut(m, 1:4, 1:5):\n", m
    )

    # `store` is the SIMD write, the counterpart of `load`.
    store[width=4](w, 0, 0, SIMD[int64, 4](7, 7, 7, 7))
    print("After store(w, 0, 0, [7, 7, 7, 7]):\n", m)

    # A `view_mut` of a `view_mut` keeps write access, and its coordinates are
    # the parent view's.
    var inner = view_mut(w, Slice(1, 3), Slice(0, 2))
    inner[0, 0] = -1
    inner[1, 1] = -2
    print("After writing through view_mut(w, 1:3, 0:2):\n", m)

    # Strided windows are writable in exactly the same way.
    var checker = view_mut(m, Slice(0, 5, 2), Slice(0, 6, 2))
    for i in range(checker.nrows):
        for j in range(checker.ncols):
            checker[i, j] = 5
    print("After filling view_mut(m, 0:5:2, 0:6:2) with 5:\n", m)


# ===----------------------------------------------------------------------=== #
# Region writes
# ===----------------------------------------------------------------------=== #


def setitem_regions() raises:
    _banner("REGION WRITES")

    var m = la.zeros[int64](6, 6)

    # `m[a:b, c:d] = src` is not spellable in Mojo 1.0 without breaking
    # ordinary slicing, so region writes are named calls.
    var w = m.view_mut(Slice(0, 6), Slice(0, 6))

    # Dropping the slices writes the whole view. This form cannot fail, so
    # unlike the region form it is not declared `raises`.
    fill(w, -1)
    print("fill(w, -1) - the whole view:\n", m)

    fill(w, Slice(0, 3), Slice(0, 3), 1)
    print("fill(w, 0:3, 0:3, 1):\n", m)

    fill(w, Slice(3, 6), Slice(3, 6), 2)
    print("fill(w, 3:6, 3:6, 2):\n", m)

    # `assign` copies a whole block in. The shapes must match exactly - there
    # is no broadcasting.
    var block = la.matrix[int64]([[8, 9], [9, 8]])
    assign(w, Slice(0, 2), Slice(4, 6), block)
    print("assign(w, 0:2, 4:6, [[8, 9], [9, 8]]):\n", m)

    # The source may itself be a view. Copying a block of `m` into another
    # part of `m` has to go through an owning snapshot, though: `w` already
    # holds `m` mutably, and Mojo will not let one call borrow `m` mutably and
    # immutably at once, so `assign(w, ..., m[0:2, 4:6])` is a compile error.
    var snapshot = m[0:2, 4:6].to_matrix()
    assign(w, Slice(4, 6), Slice(0, 2), snapshot)
    print("assign(w, 4:6, 0:2, a snapshot of m[0:2, 4:6]):\n", m)

    # A mismatch raises rather than truncating.
    try:
        assign(w, Slice(0, 2), Slice(0, 2), la.ones[int64](3, 3))
    except e:
        print("Assigning a 3x3 into a 2x2 region raises:")
        print(e)

    # `Matrix` carries the same two operations as methods, for when you have
    # the owner in hand and do not need a view at all.
    m.set(Slice(2, 4), Slice(2, 4), 0)
    print("m.set(2:4, 2:4, 0) - the method form:\n", m)


# ===----------------------------------------------------------------------=== #
# Mutable iteration
# ===----------------------------------------------------------------------=== #


def mutable_iteration() raises:
    _banner("MUTABLE ITERATION")

    var m = _grid[float64](4, 4)
    print("A 4x4 matrix:\n", m)

    # `m.rows()` is read-only. `rows_mut` is the writable walk, and it is what
    # an in-place row operation wants - Gaussian elimination, normalisation,
    # scaling a row by a pivot.
    for row in rows_mut(m):
        var total = Scalar[float64](0)
        for j in range(row.ncols):
            total += row[0, j]
        if total != 0:
            for j in range(row.ncols):
                row[0, j] /= total
    print("After dividing each row by its own sum:\n", m)

    # Columns too.
    var n = _grid[int64](3, 4)
    for col in cols_mut(n):
        for i in range(col.nrows):
            col[i, 0] *= 2
    print("After doubling every column of a 3x4 grid:\n", n)


# ===----------------------------------------------------------------------=== #
# Demoting a mutable view
# ===----------------------------------------------------------------------=== #


def demoting_with_as_imm() raises:
    _banner("AS_IMM")

    var m = _grid[int64](4, 4)
    var w = view_mut(m, Slice(0, 4), Slice(0, 4))
    print("w.mut          :", w.mut)
    print("w.as_imm().mut :", w.as_imm().mut)

    # A mutable view is an exclusive borrow, so it cannot appear twice in one
    # expression. Demoting it lifts that restriction - the same move as going
    # from `&mut T` to `&T` in Rust. There is no way back.
    var top = w.as_imm()[0:1, :]
    var bottom = w.as_imm()[3:4, :]
    print("Combining two demoted windows of one mutable view:\n", top + bottom)


# ===----------------------------------------------------------------------=== #
# Iteration
# ===----------------------------------------------------------------------=== #


def iteration() raises:
    _banner("ITERATION")

    var m = _grid[int64](5, 6)
    var v = m[1:4, 1:5]
    print("v = m[1:4, 1:5]:\n", v)

    # Iterating a view walks its rows, each yielded as another view.
    print("for row in v:")
    for row in v:
        print("  ", row.__str__())

    print("v.cols():")
    for col in v.cols():
        print("  ", col.__str__())

    print("v.rows[False]() - bottom to top:")
    for row in v.rows[False]():
        print("  ", row.__str__())

    # Reduce over rows without materialising anything.
    print("Row sums:")
    for row in v:
        var total = Scalar[int64](0)
        for j in range(row.ncols):
            total += row[0, j]
        print("  ", total)


# ===----------------------------------------------------------------------=== #
# Arithmetic on views
# ===----------------------------------------------------------------------=== #


def arithmetic() raises:
    _banner("ARITHMETIC ON VIEWS")

    var m = _grid[float64](4, 4)
    print("A 4x4 matrix:\n", m)

    var top = m[0:2, :]
    var bottom = m[2:4, :]

    # View with view. The result is always an owning `Matrix` - an operation
    # has to put its output somewhere, and a view owns nothing.
    print("m[0:2, :] + m[2:4, :]:\n", top + bottom)
    print("m[2:4, :] - m[0:2, :]:\n", bottom - top)
    print("m[0:2, :] * m[2:4, :] - elementwise:\n", top * bottom)

    # View with matrix, and matrix with view: both directions work.
    var ones = la.ones[float64](2, 4)
    print("view + matrix:\n", top + ones)
    print("matrix + view:\n", ones + top)

    # Scalars on either side.
    print("view * 10:\n", top * 10.0)
    print("100 - view:\n", 100.0 - top)

    # Matrix multiplication of two windows.
    var a = m[0:2, 0:3]
    var b = m[1:4, 0:2]
    print("m[0:2, 0:3] @ m[1:4, 0:2]:\n", a @ b)

    # Comparisons give boolean matrices.
    print("m[0:2, :] > m[2:4, :]:\n", top > bottom)
    print("m[0:2, :] >= 2:\n", top >= 2.0)

    # The named routines take views directly, which is the point of the whole
    # design: a decomposition of a sub-block copies nothing on the way in.
    print("transpose(m[0:2, :]):\n", la.transpose(top))
    print("trace(m[1:4, 1:4]):", la.trace(m[1:4, 1:4]))

    var big = la.matrix[float64](
        [
            [9.0, 2.0, 1.0, 0.0],
            [2.0, 8.0, 3.0, 1.0],
            [1.0, 3.0, 7.0, 2.0],
            [0.0, 1.0, 2.0, 6.0],
        ]
    )
    var block = big[1:4, 1:4]
    print("A 3x3 sub-block of a larger matrix:\n", block)
    print("det of the block:", la.det(block))
    print("inv of the block:\n", la.inv(block))
    print("cholesky of the block:\n", la.cholesky(block))


# ===----------------------------------------------------------------------=== #
# Materialisation
# ===----------------------------------------------------------------------=== #


def materialisation() raises:
    _banner("MATERIALISATION")

    var m = _grid[int64](6, 6)
    var v = m[1:6:2, 1:6:2]
    print("A strided view v = m[1:6:2, 1:6:2]:\n", v)

    # `to_matrix()` is the one deliberate allocation in the view API: it walks
    # the strided window and lays it out densely in fresh storage.
    var owned = v.to_matrix()
    print("v.to_matrix() - dense, row-major, independent:\n", owned)
    print(
        "Strides of the copy:", owned.get_row_stride(), owned.get_col_stride()
    )

    # Independent, so a write to the copy leaves the original alone.
    owned[0, 0] = -1
    print("After owned[0, 0] = -1, the view still reads v[0, 0] =", v[0, 0])

    # `copy()` on a view is *not* this: it duplicates the handle, and both
    # copies still look at the same elements.
    var handle = v.copy()
    m[1, 1] = -777
    print("After m[1, 1] = -777, the copied handle reads", handle[0, 0])
