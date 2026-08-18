"""
Sorting a matrix one lane at a time.

Sorting is not a reduction: it produces one output per input, and it needs the
whole lane in hand at once rather than an accumulator threaded through it. So
these do not go through `apply_along_axis`. Each lane is gathered into a
contiguous buffer, sorted with the standard library's sort, and written back.
Gathering is what makes a strided lane - a column of a row-major matrix, or a
row of `m[0:8:2, 1:9:2]` - cost the same as a dense one.

Every routine here requires an explicit `axis`. NumPy's default is `axis=-1`
for `sort` but a full reduction for `sum`, and carrying that inconsistency
into a library with only two dimensions would make `sort(m)` read like
`sum(m)` while meaning something entirely different.

`sort` and `argsort` allocate a result and leave the operand untouched.
`sort_inplace` takes the matrix itself by mutable reference and rewrites it
through its own strides, so a transposed or column-major matrix keeps its
layout. It takes a `Matrix` rather than a `MatrixView` for the reason given in
`routines/mutation.mojo`: a view is generic over `origin` and Mojo checks a
body against the read-only instantiation too.
"""

from std.builtin.sort import sort as _sort_list

from linamo.types.errors import ValueError
from linamo.types.matrix import Matrix
from linamo.types.matrix_view import MatrixView


def _check_axis(axis: Int, function: String) raises:
    if axis != 0 and axis != 1:
        raise ValueError(function=function, message="Axis must be 0 or 1.")


def sort[
    dtype: DType, origin: Origin[mut=False]
](m: MatrixView[dtype, origin], axis: Int) raises -> Matrix[dtype]:
    """Returns a copy with each lane sorted ascending.

    Parameters:
        dtype: The data type of the matrix elements.
        origin: The origin of the operand.

    Args:
        m: The matrix or view to sort.
        axis: The dimension to sort along. 0 sorts each column, 1 sorts each
            row.

    Returns:
        A new C-contiguous matrix of the same shape, each lane ascending.

    Raises:
        ValueError: If `axis` is neither 0 nor 1.
    """
    _check_axis(axis, "sort(m, axis)")
    var result = Matrix[dtype](m.nrows, m.ncols, m.ncols, 1)

    if axis == 0:
        for j in range(m.ncols):
            var lane = List[Scalar[dtype]](capacity=m.nrows)
            for i in range(m.nrows):
                lane.append(m[i, j])
            _sort_list(lane)
            for i in range(m.nrows):
                result.data[i * m.ncols + j] = lane[i]
    else:
        for i in range(m.nrows):
            var lane = List[Scalar[dtype]](capacity=m.ncols)
            for j in range(m.ncols):
                lane.append(m[i, j])
            _sort_list(lane)
            for j in range(m.ncols):
                result.data[i * m.ncols + j] = lane[j]

    return result^


def sort_inplace[dtype: DType](mut m: Matrix[dtype], axis: Int) raises:
    """Sorts each lane of a matrix ascending, in place.

    The matrix keeps its own strides, so a column-major or transposed matrix is
    not silently re-laid-out the way `sort` (which always returns a fresh
    C-contiguous result) would.

    Parameters:
        dtype: The data type of the matrix elements.

    Args:
        m: The matrix to sort. Modified in place.
        axis: The dimension to sort along. 0 sorts each column, 1 sorts each
            row.

    Raises:
        ValueError: If `axis` is neither 0 nor 1.
    """
    _check_axis(axis, "sort_inplace(m, axis)")

    if axis == 0:
        for j in range(m.ncols):
            var lane = List[Scalar[dtype]](capacity=m.nrows)
            for i in range(m.nrows):
                lane.append(m[i, j])
            _sort_list(lane)
            for i in range(m.nrows):
                m[i, j] = lane[i]
    else:
        for i in range(m.nrows):
            var lane = List[Scalar[dtype]](capacity=m.ncols)
            for j in range(m.ncols):
                lane.append(m[i, j])
            _sort_list(lane)
            for j in range(m.ncols):
                m[i, j] = lane[j]


def argsort[
    dtype: DType, origin: Origin[mut=False]
](m: MatrixView[dtype, origin], axis: Int) raises -> Matrix[DType.int64]:
    """Returns the indices that would sort each lane ascending.

    Ties keep their original relative order, so the result is a stable
    permutation and `argsort` agrees with `sort` element for element.

    Parameters:
        dtype: The data type of the matrix elements.
        origin: The origin of the operand.

    Args:
        m: The matrix or view to sort.
        axis: The dimension to sort along. 0 sorts each column, 1 sorts each
            row.

    Returns:
        A matrix of `int64` indices, each counted within its own lane.

    Raises:
        ValueError: If `axis` is neither 0 nor 1.
    """
    _check_axis(axis, "argsort(m, axis)")
    var result = Matrix[DType.int64](m.nrows, m.ncols, m.ncols, 1)

    if axis == 0:
        for j in range(m.ncols):
            var order = _stable_order(m, j, 0)
            for i in range(m.nrows):
                result.data[i * m.ncols + j] = Int64(order[i])
    else:
        for i in range(m.nrows):
            var order = _stable_order(m, i, 1)
            for j in range(m.ncols):
                result.data[i * m.ncols + j] = Int64(order[j])

    return result^


def _stable_order[
    dtype: DType, origin: Origin[mut=False]
](m: MatrixView[dtype, origin], lane: Int, axis: Int) -> List[Int]:
    """Returns the stable ascending permutation of one lane.

    Insertion sort on the index list. The lane is already gathered by index
    rather than by value, so a comparison sort that moves indices is the
    natural shape; insertion sort is chosen because it is stable by
    construction and needs no side buffer. Lanes are a single matrix dimension,
    so this is quadratic in one dimension only - revisit in Phase 10 if a
    profile ever says it matters.
    """
    var n = m.nrows if axis == 0 else m.ncols
    var order = List[Int](capacity=n)

    for k in range(n):
        var value = m[k, lane] if axis == 0 else m[lane, k]
        var pos = len(order)
        while pos > 0:
            var prev = order[pos - 1]
            var prev_value = m[prev, lane] if axis == 0 else m[lane, prev]
            if prev_value > value:
                pos -= 1
            else:
                break
        order.insert(pos, k)

    return order^
