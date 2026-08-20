"""
Index-returning reductions: where the smallest and largest elements are.

These do not go through `fold`, because a fold carries one accumulator and
these carry two - the best value seen and where it was. The traversal is the
same otherwise, and the tie rule matches NumPy: the *first* extremum in the
scan order wins.

The full-matrix forms return a single flat index counted in row-major order,
as NumPy's `argmin` on a 2-D array does, regardless of how the operand is
actually laid out in memory. Use `divmod(idx, m.ncols())` to get back to a
row/column pair. The axis forms return a matrix of `int64` indices, one per
lane, each counted within its own lane.
"""

from linamo.types.errors import ValueError
from linamo.types.matrix import Matrix
from linamo.types.matrix_view import MatrixView


def _arg_extreme[
    T: Copyable & Deinitable & Comparable,
    origin: Origin[mut=False],
    //,
    want_max: Bool,
](v: MatrixView[T, origin]) -> Int:
    """Returns the row-major position of the first extremum in `v`."""
    var best = v[0, 0].copy()
    var best_at = 0
    var k = 0
    for i in range(v.nrows()):
        for j in range(v.ncols()):
            comptime if want_max:
                if v[i, j] > best:
                    best = v[i, j].copy()
                    best_at = k
            else:
                if v[i, j] < best:
                    best = v[i, j].copy()
                    best_at = k
            k += 1
    return best_at


def _arg_axis[
    T: Copyable & Deinitable & Comparable,
    origin: Origin[mut=False],
    //,
    want_max: Bool,
](m: MatrixView[T, origin], axis: Int) raises -> Matrix[Scalar[DType.int64]]:
    """Shared body of the axis forms of `argmin` and `argmax`."""
    if m.size() == 0:
        raise ValueError(
            function="argmin/argmax(m, axis)",
            message="Cannot reduce an empty matrix.",
        )
    if axis == 0:
        var result = Matrix[Scalar[DType.int64]](1, m.ncols(), m.ncols(), 1)
        for j in range(m.ncols()):
            result._data[j] = Int64(
                _arg_extreme[want_max=want_max](m[:, j : j + 1])
            )
        return result^
    elif axis == 1:
        var result = Matrix[Scalar[DType.int64]](m.nrows(), 1, 1, 1)
        for i in range(m.nrows()):
            result._data[i] = Int64(
                _arg_extreme[want_max=want_max](m[i : i + 1, :])
            )
        return result^
    raise ValueError(
        function="argmin/argmax(m, axis)", message="Axis must be 0 or 1."
    )


def argmin[
    T: Copyable & Deinitable & Comparable, origin: Origin[mut=False]
](m: MatrixView[T, origin]) raises -> Int:
    """Returns the row-major index of the smallest element.

    Parameters:
        T: The type of the matrix elements.
        origin: The origin of the operand.

    Args:
        m: The matrix or view to search.

    Returns:
        The flat row-major index of the first smallest element.

    Raises:
        ValueError: If the operand is empty.
    """
    if m.size() == 0:
        raise ValueError(
            function="argmin(m)", message="Cannot search an empty matrix."
        )
    return _arg_extreme[want_max=False](m)


def argmin[
    T: Copyable & Deinitable & Comparable, origin: Origin[mut=False]
](m: MatrixView[T, origin], axis: Int) raises -> Matrix[Scalar[DType.int64]]:
    """Returns the index of the smallest element along one axis.

    Parameters:
        T: The type of the matrix elements.
        origin: The origin of the operand.

    Args:
        m: The matrix or view to search.
        axis: The dimension to remove. 0 returns `1 x ncols`, 1 returns
            `nrows x 1`.

    Returns:
        A matrix of `int64` indices, each counted within its own lane.

    Raises:
        ValueError: If `axis` is neither 0 nor 1, or the operand is empty.
    """
    return _arg_axis[want_max=False](m, axis)


def argmax[
    T: Copyable & Deinitable & Comparable, origin: Origin[mut=False]
](m: MatrixView[T, origin]) raises -> Int:
    """Returns the row-major index of the largest element.

    Parameters:
        T: The type of the matrix elements.
        origin: The origin of the operand.

    Args:
        m: The matrix or view to search.

    Returns:
        The flat row-major index of the first largest element.

    Raises:
        ValueError: If the operand is empty.
    """
    if m.size() == 0:
        raise ValueError(
            function="argmax(m)", message="Cannot search an empty matrix."
        )
    return _arg_extreme[want_max=True](m)


def argmax[
    T: Copyable & Deinitable & Comparable, origin: Origin[mut=False]
](m: MatrixView[T, origin], axis: Int) raises -> Matrix[Scalar[DType.int64]]:
    """Returns the index of the largest element along one axis.

    Parameters:
        T: The type of the matrix elements.
        origin: The origin of the operand.

    Args:
        m: The matrix or view to search.
        axis: The dimension to remove. 0 returns `1 x ncols`, 1 returns
            `nrows x 1`.

    Returns:
        A matrix of `int64` indices, each counted within its own lane.

    Raises:
        ValueError: If `axis` is neither 0 nor 1, or the operand is empty.
    """
    return _arg_axis[want_max=True](m, axis)
