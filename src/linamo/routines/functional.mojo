"""
Generic traversal of a matrix, and of one axis of a matrix.

Every reduction in the library is the same walk with a different accumulator,
so the walk is written once, here, and each routine supplies only its kernel.
`sum`, `prod`, `min`, `max`, `all` and `any` are two lines each on top of this
module.

There are two levels, and they compose:

`fold` reduces a whole view to one scalar. It carries the layout dispatch -
row-contiguous, column-contiguous, strided - so a kernel written against it
never branches on strides. This is the same three-way split `matmul` uses, and
writing it once here is the reason the reductions do not repeat it.

`apply_along_axis` walks one axis and calls a per-lane kernel, collecting the
results into a matrix. The lane kernel is usually `fold` with the same
accumulator the full reduction uses, which is what makes `sum(m)` and
`sum(m, axis=0)` share an implementation rather than merely resemble one.

A note on what `axis` means, because the two indices run opposite ways. `axis`
follows NumPy: it names the dimension that is *reduced away*. Reducing axis 0
collapses the rows, so the result has one entry per column and the traversal
walks columns - `MatrixAxisIter` axis `1`. The inversion happens once, in
`apply_along_axis`, and nothing above this module sees it.
"""

from linamo.types.matrix import Matrix
from linamo.types.matrix_iter import MatrixAxisIter
from linamo.types.matrix_view import MatrixView


def fold[
    dtype: DType,
    origin: Origin[mut=False],
    func: def(Scalar[dtype], Scalar[dtype]) thin -> Scalar[dtype],
](v: MatrixView[dtype, origin], init: Scalar[dtype]) -> Scalar[dtype]:
    """Reduces every element of a view to a single scalar.

    The accumulator is threaded left to right in memory order, so `func` should
    be associative if the traversal order is not to matter. For the operations
    that use this - sum, product, min, max, and the boolean folds - it is.

    Parameters:
        dtype: The data type of the matrix elements.
        origin: The origin of the view, which is read-only. Everything the
            library hands out is, apart from `routines.mutation`; call
            `as_imm()` on a mutable view to reduce it.
        func: The accumulator, called as `func(accumulator, element)`.

    Args:
        v: The view to reduce.
        init: The starting value of the accumulator, and the result for an
            empty view.

    Returns:
        The accumulated scalar.

    Notes:

    The two fast paths walk consecutive addresses and index the span directly,
    skipping the multiply that `v[i, j]` performs per element. They are
    selected by `is_row_contiguous` / `is_col_contiguous`, the weaker of the
    two contiguity tests: a lane taken out of a larger matrix has a unit stride
    along its own extent but says nothing about the other, which is exactly
    what these need.

    Vectorising the fold is left to Phase 10. It needs a SIMD-level accumulator
    plus a horizontal reducer, which cannot be expressed by the scalar `func`
    parameter above without making the function type generic over lane count.
    """
    var acc = init

    if v.is_row_contiguous():
        for i in range(v.nrows()):
            var base = v.offset() + i * v.row_stride()
            for j in range(v.ncols()):
                acc = func(acc, v._data[base + j])
    elif v.is_col_contiguous():
        for j in range(v.ncols()):
            var base = v.offset() + j * v.col_stride()
            for i in range(v.nrows()):
                acc = func(acc, v._data[base + i])
    else:
        for i in range(v.nrows()):
            for j in range(v.ncols()):
                acc = func(acc, v[i, j])

    return acc


def apply_along_axis[
    dtype: DType,
    origin: Origin[mut=False],
    axis: Int,
    func: def(MatrixView[dtype, origin]) thin -> Scalar[dtype],
](m: MatrixView[dtype, origin]) raises -> Matrix[dtype] where (
    axis == 0 or axis == 1
):
    """Applies a lane kernel along one axis, collecting the results.

    Each lane is handed to `func` as a view onto the original buffer; nothing
    is copied, and a lane of a strided matrix is itself just a strided view.

    Parameters:
        dtype: The data type of the matrix elements.
        origin: The origin of the input view, which is read-only.
        axis: The dimension reduced away. 0 collapses the rows and yields a
            `1 x ncols` result; 1 collapses the columns and yields `nrows x 1`.
        func: The per-lane kernel, taking a lane view and returning one scalar.

    Args:
        m: The matrix or view to walk.

    Returns:
        A new `1 x ncols` matrix for `axis=0`, or `nrows x 1` for `axis=1`.

    Notes:

    `axis` is a compile-time parameter so the traversal can be specialised and
    so a wrong value is a build error rather than a raise. The public
    reductions take a runtime `axis` argument and branch onto these two
    instantiations.
    """
    # `axis` names the dimension to remove; the iterator names the dimension to
    # walk. They are complements.
    comptime lane_axis = 1 - axis

    comptime if axis == 0:
        var result = Matrix[dtype](1, m.ncols(), m.ncols(), 1)
        var k = 0
        for lane in MatrixAxisIter[axis=lane_axis](m):
            result._data[k] = func(lane)
            k += 1
        return result^
    else:
        var result = Matrix[dtype](m.nrows(), 1, 1, 1)
        var k = 0
        for lane in MatrixAxisIter[axis=lane_axis](m):
            result._data[k] = func(lane)
            k += 1
        return result^
