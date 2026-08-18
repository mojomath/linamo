"""
Reductions that aggregate a matrix: sums and running sums.

Each routine comes in two arities. `sum(m)` reduces the whole matrix to one
scalar; `sum(m, axis)` reduces one dimension away and returns a matrix. Both
run through `routines/functional.mojo`, so the axis form is the full form
applied lane by lane rather than a second implementation of the same idea.

`axis` follows NumPy: it names the dimension that disappears. `sum(m, axis=0)`
adds down the columns and returns a `1 x ncols` result; `sum(m, axis=1)` adds
across the rows and returns `nrows x 1`. The result keeps two dimensions in
both cases, as NumPy's `keepdims=True` would - this library has no 1-D type to
degrade to, and a `1 x n` result composes with everything else here.
"""

from linamo.routines.functional import apply_along_axis, fold
from linamo.types.errors import ValueError
from linamo.types.matrix import Matrix
from linamo.types.matrix_view import MatrixView


def _add_op[dtype: DType](a: Scalar[dtype], b: Scalar[dtype]) -> Scalar[dtype]:
    return a + b


def _sum_lane[
    dtype: DType, origin: Origin[mut=False]
](v: MatrixView[dtype, origin]) -> Scalar[dtype]:
    """Sums one lane. This is the kernel `apply_along_axis` calls per lane."""
    return fold[func=_add_op[dtype]](v, Scalar[dtype](0))


def sum[
    dtype: DType, origin: Origin[mut=False]
](m: MatrixView[dtype, origin]) -> Scalar[dtype]:
    """Sums every element of a matrix or view.

    Parameters:
        dtype: The data type of the matrix elements.
        origin: The origin of the operand.

    Args:
        m: The matrix or view to sum.

    Returns:
        The sum of all elements, or zero if the operand is empty.
    """
    return _sum_lane(m)


def sum[
    dtype: DType, origin: Origin[mut=False]
](m: MatrixView[dtype, origin], axis: Int) raises -> Matrix[dtype]:
    """Sums along one axis.

    Parameters:
        dtype: The data type of the matrix elements.
        origin: The origin of the operand.

    Args:
        m: The matrix or view to sum.
        axis: The dimension to remove. 0 sums down the columns and returns a
            `1 x ncols` matrix; 1 sums across the rows and returns `nrows x 1`.

    Returns:
        A new matrix holding one sum per lane.

    Raises:
        ValueError: If `axis` is neither 0 nor 1.
    """
    if axis == 0:
        return apply_along_axis[axis=0, func=_sum_lane[dtype, origin]](m)
    elif axis == 1:
        return apply_along_axis[axis=1, func=_sum_lane[dtype, origin]](m)
    raise ValueError(
        function="sum(m, axis)",
        message="Axis must be 0 or 1.",
    )


# ===----------------------------------------------------------------------===#
# Running sums
# ===----------------------------------------------------------------------===#
# A scan is not a fold: it produces one output per input rather than one per
# lane, so it does not go through `apply_along_axis`. It gets its own walk.


def cumsum[
    dtype: DType, origin: Origin[mut=False]
](m: MatrixView[dtype, origin]) raises -> Matrix[dtype]:
    """Returns the running sum over every element, in row-major order.

    Mirrors NumPy's `cumsum` with no axis: the matrix is read as if flattened
    C-contiguously, and the result has the same shape as the input rather than
    being flattened, so it can be read back with the original indices.

    Parameters:
        dtype: The data type of the matrix elements.
        origin: The origin of the operand.

    Args:
        m: The matrix or view to scan.

    Returns:
        A new C-contiguous matrix of running sums.
    """
    var result = Matrix[dtype](m.nrows, m.ncols, m.ncols, 1)
    var acc = Scalar[dtype](0)
    var k = 0
    for i in range(m.nrows):
        for j in range(m.ncols):
            acc += m[i, j]
            result.data[k] = acc
            k += 1
    return result^


def cumsum[
    dtype: DType, origin: Origin[mut=False]
](m: MatrixView[dtype, origin], axis: Int) raises -> Matrix[dtype]:
    """Returns the running sum along one axis.

    Parameters:
        dtype: The data type of the matrix elements.
        origin: The origin of the operand.

    Args:
        m: The matrix or view to scan.
        axis: The dimension to accumulate along. 0 runs down each column,
            1 runs across each row.

    Returns:
        A new C-contiguous matrix with the same shape as the input.

    Raises:
        ValueError: If `axis` is neither 0 nor 1.
    """
    if axis != 0 and axis != 1:
        raise ValueError(
            function="cumsum(m, axis)",
            message="Axis must be 0 or 1.",
        )

    var result = Matrix[dtype](m.nrows, m.ncols, m.ncols, 1)
    if axis == 0:
        for j in range(m.ncols):
            var acc = Scalar[dtype](0)
            for i in range(m.nrows):
                acc += m[i, j]
                result.data[i * m.ncols + j] = acc
    else:
        for i in range(m.nrows):
            var acc = Scalar[dtype](0)
            for j in range(m.ncols):
                acc += m[i, j]
                result.data[i * m.ncols + j] = acc
    return result^
