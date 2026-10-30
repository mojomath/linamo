"""
Reductions that aggregate a matrix: sums, running sums, and the moments
`mean`, `variance` and `stddev`, with `cov` and `corrcoef` across variables.

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

from std.math import sqrt

from decimo import BigInt, Numeric, Rootable

from linamo.errors import ValueError
from linamo.routines.functional import apply_along_axis, fold
from linamo.traits.conjugable import Conjugable
from linamo.types.complex import Complex
from linamo.types.matrix import Matrix
from linamo.types.matrix_view import MatrixView


def _add_op[dtype: DType](a: Scalar[dtype], b: Scalar[dtype]) -> Scalar[dtype]:
    """Adds up two scalars of the same data type."""
    return a + b


def _sum_lane[
    dtype: DType, origin: Origin[mut=False]
](v: MatrixView[Scalar[dtype], origin]) -> Scalar[dtype]:
    """Sums one lane. This is the kernel `apply_along_axis` calls per lane."""
    return fold[func=_add_op[dtype]](v, Scalar[dtype](0))


def sum[
    dtype: DType, origin: Origin[mut=False], //
](m: MatrixView[Scalar[dtype], origin]) -> Scalar[dtype]:
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
    T: Numeric, origin: Origin[mut=False], //
](m: MatrixView[T, origin]) raises -> T:
    """Sums every element of a matrix or view.

    The arbitrary-precision twin of the routine above. There is no `fold` here:
    `fold` is written against `Scalar[dtype]`, and an element that carries its
    own `__add__` needs the plain accumulation loop instead.

    Parameters:
        T: The type of the matrix elements.
        origin: The origin of the operand.

    Args:
        m: The matrix or view to reduce.

    Returns:
        The sum of every element, or zero for an empty input.
    """
    var accumulation = T.zero()
    for i in range(m.nrows()):
        for j in range(m.ncols()):
            accumulation = accumulation + m[i, j]
    return accumulation^


def sum[
    dtype: DType, origin: Origin[mut=False], //
](m: MatrixView[Scalar[dtype], origin], axis: Int) raises -> Matrix[
    Scalar[dtype]
]:
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
    dtype: DType, origin: Origin[mut=False], //
](m: MatrixView[Scalar[dtype], origin]) raises -> Matrix[Scalar[dtype]]:
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
    var result = Matrix[Scalar[dtype]](m.nrows(), m.ncols(), m.ncols(), 1)
    var accumulation = Scalar[dtype](0)
    var k = 0
    for i in range(m.nrows()):
        for j in range(m.ncols()):
            accumulation += m[i, j]
            result._data[k] = accumulation
            k += 1
    return result^


def cumsum[
    dtype: DType, origin: Origin[mut=False], //
](m: MatrixView[Scalar[dtype], origin], axis: Int) raises -> Matrix[
    Scalar[dtype]
]:
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

    var result = Matrix[Scalar[dtype]](m.nrows(), m.ncols(), m.ncols(), 1)

    if axis == 0:
        for j in range(m.ncols()):
            var accumulation = Scalar[dtype](0)
            for i in range(m.nrows()):
                accumulation += m[i, j]
                result._data[i * m.ncols() + j] = accumulation
    else:
        for i in range(m.nrows()):
            var accumulation = Scalar[dtype](0)
            for j in range(m.ncols()):
                accumulation += m[i, j]
                result._data[i * m.ncols() + j] = accumulation

    return result^


# ===----------------------------------------------------------------------===#
# Moments: mean, variance, stddev
# ===----------------------------------------------------------------------===#
# All three walk a matrix lane by lane. `axis` is 0 or 1 as in `sum`, and the
# private value `_ALL` stands for "every element, as a single lane", so the full
# form and the per-axis form run through the same kernels.
#
# Element types. A floating-point scalar takes the `Scalar[dtype]` overloads.
# Any other `Numeric` element takes the generic ones, except `BInt`: its
# division truncates, so the mean of `[1, 2]` would come back as 1 with no
# error. It is refused at compile time, as is an integer dtype; `astype` turns
# an integer matrix into a floating-point one.
#
# Complex elements change one thing, and it is the same thing everywhere: the
# squared deviation is `(x - mu) * conj(x - mu)`, a real number, not
# `(x - mu)^2`, which is complex and can cancel. Generic code cannot ask whether
# `T` conjugates --- Mojo wants a `where` clause proven at the call site --- so,
# as with the pivot ranking in `linalg.mojo`, the product is a parameter of one
# shared core: `_mul_plain` for a real element, `_mul_conj` for a complex one.

comptime _ALL = -1
"""The `axis` value the lane kernels read as "the whole matrix"."""


def _check_float[dtype: DType]():
    comptime assert dtype.is_floating_point(), (
        "Statistics need a floating-point element type: an integer division"
        " truncates. Convert first, e.g. `m.astype[DType.float64]()`."
    )


def _check_divides[T: AnyType]():
    comptime assert not (T == BigInt), (
        "Statistics are not defined over `BInt`: its division truncates, so a"
        " mean would be silently wrong. Build the matrix over `Decimal` or a"
        " floating-point type instead."
    )


def _check_axis(function: String, axis: Int) raises:
    if axis != 0 and axis != 1:
        raise ValueError(function=function, message="Axis must be 0 or 1.")


def _lane_count(nrows: Int, ncols: Int, axis: Int) -> Int:
    """Returns how many lanes `axis` divides a matrix into."""
    if axis == 0:
        return ncols
    elif axis == 1:
        return nrows
    return 1


def _lane_len(nrows: Int, ncols: Int, axis: Int) -> Int:
    """Returns how many elements each lane holds."""
    if axis == 0:
        return nrows
    elif axis == 1:
        return ncols
    return nrows * ncols


def _coords(ncols: Int, axis: Int, lane: Int, k: Int) -> Tuple[Int, Int]:
    """Returns the `(row, col)` of element `k` of a lane."""
    if axis == 0:
        return (k, lane)
    elif axis == 1:
        return (lane, k)
    return (k // ncols, k % ncols)


def _lanes_to_matrix[
    T: Copyable & Deinitable
](var data: List[T], axis: Int) -> Matrix[T]:
    """Shapes one value per lane as `1 x n` for `axis=0`, `n x 1` for `axis=1`.
    """
    var n = len(data)
    if axis == 0:
        return Matrix[T](
            buffer=data^, nrows=1, ncols=n, row_stride=n, col_stride=1
        )
    return Matrix[T](buffer=data^, nrows=n, ncols=1, row_stride=1, col_stride=1)


def _check_dof(function: String, n: Int, ddof: Int) raises:
    if n - ddof <= 0:
        raise ValueError(
            function=function,
            message=(
                "Not enough observations: "
                + String(n)
                + " with ddof="
                + String(ddof)
                + " leaves no degrees of freedom."
            ),
        )


def _from_count[T: Numeric](n: Int) raises -> T:
    """Returns `n` as an element, since `Numeric` has no conversion from `Int`.

    Binary doubling, so this costs `O(log n)` additions rather than `n`.
    """
    var result = T.zero()
    var power = T.one()
    var k = n
    while k > 0:
        if k & 1:
            result = result + power
        power = power + power
        k >>= 1
    return result^


def _mul_plain[T: Numeric](a: T, b: T) raises -> T:
    return a * b


def _mul_conj[d: DType](a: Complex[d], b: Complex[d]) raises -> Complex[d]:
    return a * b.conj()


# --- Lane kernels: floating-point scalars ------------------------------------


def _mean_lane[
    dtype: DType, origin: Origin[mut=False], //
](m: MatrixView[Scalar[dtype], origin], axis: Int, lane: Int) raises -> Scalar[
    dtype
]:
    var n = _lane_len(m.nrows(), m.ncols(), axis)
    if n == 0:
        raise ValueError(
            function="mean()",
            message="Cannot take the mean of an empty matrix.",
        )
    var acc = Scalar[dtype](0)
    for k in range(n):
        var ij = _coords(m.ncols(), axis, lane, k)
        acc += m[ij[0], ij[1]]
    return acc / Scalar[dtype](n)


def _variance_lane[
    dtype: DType, origin: Origin[mut=False], //
](
    m: MatrixView[Scalar[dtype], origin], axis: Int, lane: Int, ddof: Int
) raises -> Scalar[dtype]:
    var n = _lane_len(m.nrows(), m.ncols(), axis)
    _check_dof("variance()", n, ddof)
    var mu = _mean_lane(m, axis, lane)
    var acc = Scalar[dtype](0)
    for k in range(n):
        var ij = _coords(m.ncols(), axis, lane, k)
        var dev = m[ij[0], ij[1]] - mu
        acc += dev * dev
    return acc / Scalar[dtype](n - ddof)


# --- Lane kernels: Numeric elements -------------------------------------------


def _mean_lane[
    T: Numeric, origin: Origin[mut=False], //
](m: MatrixView[T, origin], axis: Int, lane: Int) raises -> T:
    var n = _lane_len(m.nrows(), m.ncols(), axis)
    if n == 0:
        raise ValueError(
            function="mean()",
            message="Cannot take the mean of an empty matrix.",
        )
    var acc = T.zero()
    for k in range(n):
        var ij = _coords(m.ncols(), axis, lane, k)
        acc = acc + m[ij[0], ij[1]]
    return acc / _from_count[T](n)


def _variance_lane[
    T: Numeric,
    origin: Origin[mut=False],
    //,
    mul_conj: def(T, T) raises thin -> T,
](m: MatrixView[T, origin], axis: Int, lane: Int, ddof: Int) raises -> T:
    var n = _lane_len(m.nrows(), m.ncols(), axis)
    _check_dof("variance()", n, ddof)
    var mu = _mean_lane(m, axis, lane)
    var acc = T.zero()
    for k in range(n):
        var ij = _coords(m.ncols(), axis, lane, k)
        var dev = m[ij[0], ij[1]] - mu
        acc = acc + mul_conj(dev, dev)
    return acc / _from_count[T](n - ddof)


def _variance_axis[
    T: Numeric,
    origin: Origin[mut=False],
    //,
    mul_conj: def(T, T) raises thin -> T,
](m: MatrixView[T, origin], axis: Int, ddof: Int) raises -> Matrix[T]:
    _check_axis("variance(m, axis)", axis)
    var lanes = _lane_count(m.nrows(), m.ncols(), axis)
    var data = List[T](capacity=lanes)
    for lane in range(lanes):
        data.append(_variance_lane[mul_conj=mul_conj](m, axis, lane, ddof))
    return _lanes_to_matrix(data^, axis)


def _std_axis[
    T: Numeric & Rootable,
    origin: Origin[mut=False],
    //,
    mul_conj: def(T, T) raises thin -> T,
](m: MatrixView[T, origin], axis: Int, ddof: Int) raises -> Matrix[T]:
    _check_axis("stddev(m, axis)", axis)
    var lanes = _lane_count(m.nrows(), m.ncols(), axis)
    var data = List[T](capacity=lanes)
    for lane in range(lanes):
        data.append(
            _variance_lane[mul_conj=mul_conj](m, axis, lane, ddof).sqrt()
        )
    return _lanes_to_matrix(data^, axis)


# --- mean ---------------------------------------------------------------------


def mean[
    dtype: DType, origin: Origin[mut=False], //
](m: MatrixView[Scalar[dtype], origin]) raises -> Scalar[dtype]:
    """Returns the arithmetic mean of every element.

    Parameters:
        dtype: The data type of the matrix elements. Must be floating-point.
        origin: The origin of the operand.

    Args:
        m: The matrix or view to average.

    Returns:
        The mean of all elements.

    Raises:
        ValueError: If the operand is empty.
    """
    _check_float[dtype]()
    return _mean_lane(m, _ALL, 0)


def mean[
    dtype: DType, origin: Origin[mut=False], //
](m: MatrixView[Scalar[dtype], origin], axis: Int) raises -> Matrix[
    Scalar[dtype]
]:
    """Returns the mean along one axis.

    Parameters:
        dtype: The data type of the matrix elements. Must be floating-point.
        origin: The origin of the operand.

    Args:
        m: The matrix or view to average.
        axis: The dimension to remove. 0 averages down the columns and returns
            a `1 x ncols` matrix; 1 averages across the rows and returns
            `nrows x 1`.

    Returns:
        A new matrix holding one mean per lane.

    Raises:
        ValueError: If `axis` is neither 0 nor 1, or a lane is empty.
    """
    _check_float[dtype]()
    _check_axis("mean(m, axis)", axis)
    var lanes = _lane_count(m.nrows(), m.ncols(), axis)
    var data = List[Scalar[dtype]](capacity=lanes)
    for lane in range(lanes):
        data.append(_mean_lane(m, axis, lane))
    return _lanes_to_matrix(data^, axis)


def mean[
    T: Numeric, origin: Origin[mut=False], //
](m: MatrixView[T, origin]) raises -> T:
    """Returns the arithmetic mean of every element.

    The `Numeric` twin of the routine above, for `Decimal`, `Dec128` and the
    complex types. `BInt` is refused at compile time.

    Parameters:
        T: The type of the matrix elements.
        origin: The origin of the operand.

    Args:
        m: The matrix or view to average.

    Returns:
        The mean of all elements.

    Raises:
        ValueError: If the operand is empty.
    """
    _check_divides[T]()
    return _mean_lane(m, _ALL, 0)


def mean[
    T: Numeric, origin: Origin[mut=False], //
](m: MatrixView[T, origin], axis: Int) raises -> Matrix[T]:
    """Returns the mean along one axis.

    Parameters:
        T: The type of the matrix elements.
        origin: The origin of the operand.

    Args:
        m: The matrix or view to average.
        axis: The dimension to remove, as in `sum(m, axis)`.

    Returns:
        A new matrix holding one mean per lane.

    Raises:
        ValueError: If `axis` is neither 0 nor 1, or a lane is empty.
    """
    _check_divides[T]()
    _check_axis("mean(m, axis)", axis)
    var lanes = _lane_count(m.nrows(), m.ncols(), axis)
    var data = List[T](capacity=lanes)
    for lane in range(lanes):
        data.append(_mean_lane(m, axis, lane))
    return _lanes_to_matrix(data^, axis)


# --- variance -----------------------------------------------------------------
# NumPy's `var`. `var` is a keyword in Mojo, so the routine takes the long name.


def variance[
    dtype: DType, origin: Origin[mut=False], //
](m: MatrixView[Scalar[dtype], origin], *, ddof: Int = 0) raises -> Scalar[
    dtype
]:
    """Returns the variance of every element: `sum((x - mean)^2) / (n - ddof)`.

    `ddof=0`, the default, is the population variance, as in NumPy; pass
    `ddof=1` for the unbiased sample variance.

    Parameters:
        dtype: The data type of the matrix elements. Must be floating-point.
        origin: The origin of the operand.

    Args:
        m: The matrix or view to reduce.
        ddof: Delta degrees of freedom, subtracted from the element count.

    Returns:
        The variance of all elements.

    Raises:
        ValueError: If `n - ddof` is not positive.
    """
    _check_float[dtype]()
    return _variance_lane(m, _ALL, 0, ddof)


def variance[
    dtype: DType, origin: Origin[mut=False], //
](
    m: MatrixView[Scalar[dtype], origin], axis: Int, *, ddof: Int = 0
) raises -> Matrix[Scalar[dtype]]:
    """Returns the variance along one axis.

    Parameters:
        dtype: The data type of the matrix elements. Must be floating-point.
        origin: The origin of the operand.

    Args:
        m: The matrix or view to reduce.
        axis: The dimension to remove, as in `sum(m, axis)`.
        ddof: Delta degrees of freedom, subtracted from the lane length.

    Returns:
        A new matrix holding one variance per lane.

    Raises:
        ValueError: If `axis` is neither 0 nor 1, or `n - ddof` is not
            positive.
    """
    _check_float[dtype]()
    _check_axis("variance(m, axis)", axis)
    var lanes = _lane_count(m.nrows(), m.ncols(), axis)
    var data = List[Scalar[dtype]](capacity=lanes)
    for lane in range(lanes):
        data.append(_variance_lane(m, axis, lane, ddof))
    return _lanes_to_matrix(data^, axis)


def variance[
    T: Numeric, origin: Origin[mut=False], //
](m: MatrixView[T, origin], *, ddof: Int = 0) raises -> T where not conforms_to(
    T, Conjugable
):
    """Returns the variance of every element.

    Parameters:
        T: The type of the matrix elements.
        origin: The origin of the operand.

    Args:
        m: The matrix or view to reduce.
        ddof: Delta degrees of freedom, subtracted from the element count.

    Returns:
        The variance of all elements.

    Raises:
        ValueError: If `n - ddof` is not positive.
    """
    _check_divides[T]()
    return _variance_lane[mul_conj=_mul_plain[T]](m, _ALL, 0, ddof)


def variance[
    T: Numeric, origin: Origin[mut=False], //
](m: MatrixView[T, origin], axis: Int, *, ddof: Int = 0) raises -> Matrix[
    T
] where not conforms_to(T, Conjugable):
    """Returns the variance along one axis.

    Parameters:
        T: The type of the matrix elements.
        origin: The origin of the operand.

    Args:
        m: The matrix or view to reduce.
        axis: The dimension to remove, as in `sum(m, axis)`.
        ddof: Delta degrees of freedom, subtracted from the lane length.

    Returns:
        A new matrix holding one variance per lane.

    Raises:
        ValueError: If `axis` is neither 0 nor 1, or `n - ddof` is not
            positive.
    """
    _check_divides[T]()
    return _variance_axis[mul_conj=_mul_plain[T]](m, axis, ddof)


def variance[
    d: DType, origin: Origin[mut=False], //
](m: MatrixView[Complex[d], origin], *, ddof: Int = 0) raises -> Complex[d]:
    """Returns the variance of every element: `sum(|x - mean|^2) / (n - ddof)`.

    The result is real, returned as a complex number with a zero imaginary
    part.

    Parameters:
        d: The component dtype of the matrix elements.
        origin: The origin of the operand.

    Args:
        m: The matrix or view to reduce.
        ddof: Delta degrees of freedom, subtracted from the element count.

    Returns:
        The variance of all elements.

    Raises:
        ValueError: If `n - ddof` is not positive.
    """
    return _variance_lane[mul_conj=_mul_conj[d]](m, _ALL, 0, ddof)


def variance[
    d: DType, origin: Origin[mut=False], //
](
    m: MatrixView[Complex[d], origin], axis: Int, *, ddof: Int = 0
) raises -> Matrix[Complex[d]]:
    """Returns the variance along one axis, each real-valued.

    Parameters:
        d: The component dtype of the matrix elements.
        origin: The origin of the operand.

    Args:
        m: The matrix or view to reduce.
        axis: The dimension to remove, as in `sum(m, axis)`.
        ddof: Delta degrees of freedom, subtracted from the lane length.

    Returns:
        A new matrix holding one variance per lane.

    Raises:
        ValueError: If `axis` is neither 0 nor 1, or `n - ddof` is not
            positive.
    """
    return _variance_axis[mul_conj=_mul_conj[d]](m, axis, ddof)


# --- stddev -------------------------------------------------------------------
# NumPy's `std`. `std` is the root of the Mojo standard library and cannot be
# redefined, so the routine takes the long name, as `variance` does.


def stddev[
    dtype: DType, origin: Origin[mut=False], //
](m: MatrixView[Scalar[dtype], origin], *, ddof: Int = 0) raises -> Scalar[
    dtype
]:
    """Returns the standard deviation of every element, `sqrt(variance)`.

    Parameters:
        dtype: The data type of the matrix elements. Must be floating-point.
        origin: The origin of the operand.

    Args:
        m: The matrix or view to reduce.
        ddof: Delta degrees of freedom, as in `variance`.

    Returns:
        The standard deviation of all elements.

    Raises:
        ValueError: If `n - ddof` is not positive.
    """
    _check_float[dtype]()
    return sqrt(_variance_lane(m, _ALL, 0, ddof))


def stddev[
    dtype: DType, origin: Origin[mut=False], //
](
    m: MatrixView[Scalar[dtype], origin], axis: Int, *, ddof: Int = 0
) raises -> Matrix[Scalar[dtype]]:
    """Returns the standard deviation along one axis.

    Parameters:
        dtype: The data type of the matrix elements. Must be floating-point.
        origin: The origin of the operand.

    Args:
        m: The matrix or view to reduce.
        axis: The dimension to remove, as in `sum(m, axis)`.
        ddof: Delta degrees of freedom, as in `variance`.

    Returns:
        A new matrix holding one standard deviation per lane.

    Raises:
        ValueError: If `axis` is neither 0 nor 1, or `n - ddof` is not
            positive.
    """
    _check_float[dtype]()
    _check_axis("stddev(m, axis)", axis)
    var lanes = _lane_count(m.nrows(), m.ncols(), axis)
    var data = List[Scalar[dtype]](capacity=lanes)
    for lane in range(lanes):
        data.append(sqrt(_variance_lane(m, axis, lane, ddof)))
    return _lanes_to_matrix(data^, axis)


def stddev[
    T: Numeric & Rootable, origin: Origin[mut=False], //
](m: MatrixView[T, origin], *, ddof: Int = 0) raises -> T where not conforms_to(
    T, Conjugable
):
    """Returns the standard deviation of every element.

    Parameters:
        T: The type of the matrix elements.
        origin: The origin of the operand.

    Args:
        m: The matrix or view to reduce.
        ddof: Delta degrees of freedom, as in `variance`.

    Returns:
        The standard deviation of all elements.

    Raises:
        ValueError: If `n - ddof` is not positive.
    """
    _check_divides[T]()
    return _variance_lane[mul_conj=_mul_plain[T]](m, _ALL, 0, ddof).sqrt()


def stddev[
    T: Numeric & Rootable, origin: Origin[mut=False], //
](m: MatrixView[T, origin], axis: Int, *, ddof: Int = 0) raises -> Matrix[
    T
] where not conforms_to(T, Conjugable):
    """Returns the standard deviation along one axis.

    Parameters:
        T: The type of the matrix elements.
        origin: The origin of the operand.

    Args:
        m: The matrix or view to reduce.
        axis: The dimension to remove, as in `sum(m, axis)`.
        ddof: Delta degrees of freedom, as in `variance`.

    Returns:
        A new matrix holding one standard deviation per lane.

    Raises:
        ValueError: If `axis` is neither 0 nor 1, or `n - ddof` is not
            positive.
    """
    _check_divides[T]()
    return _std_axis[mul_conj=_mul_plain[T]](m, axis, ddof)


def stddev[
    d: DType, origin: Origin[mut=False], //
](m: MatrixView[Complex[d], origin], *, ddof: Int = 0) raises -> Complex[d]:
    """Returns the standard deviation of every element, `sqrt(variance)`.

    Real-valued, returned as a complex number with a zero imaginary part.

    Parameters:
        d: The component dtype of the matrix elements.
        origin: The origin of the operand.

    Args:
        m: The matrix or view to reduce.
        ddof: Delta degrees of freedom, as in `variance`.

    Returns:
        The standard deviation of all elements.

    Raises:
        ValueError: If `n - ddof` is not positive.
    """
    return _variance_lane[mul_conj=_mul_conj[d]](m, _ALL, 0, ddof).sqrt()


def stddev[
    d: DType, origin: Origin[mut=False], //
](
    m: MatrixView[Complex[d], origin], axis: Int, *, ddof: Int = 0
) raises -> Matrix[Complex[d]]:
    """Returns the standard deviation along one axis, each real-valued.

    Parameters:
        d: The component dtype of the matrix elements.
        origin: The origin of the operand.

    Args:
        m: The matrix or view to reduce.
        axis: The dimension to remove, as in `sum(m, axis)`.
        ddof: Delta degrees of freedom, as in `variance`.

    Returns:
        A new matrix holding one standard deviation per lane.

    Raises:
        ValueError: If `axis` is neither 0 nor 1, or `n - ddof` is not
            positive.
    """
    return _std_axis[mul_conj=_mul_conj[d]](m, axis, ddof)


# ===----------------------------------------------------------------------===#
# Covariance and correlation
# ===----------------------------------------------------------------------===#
# The variables are the *columns* by default: an `n_obs x n_vars` data matrix
# gives an `n_vars x n_vars` result, as in R, pandas and statsmodels. NumPy's
# default is the other way round, so `rowvar=True` is there for that layout.
#
# `cov` defaults to `ddof=1`, the unbiased sample covariance, where `variance`
# defaults to `ddof=0`. Both defaults are NumPy's. `corrcoef` takes no `ddof`:
# it cancels.
#
# Entry `[i, j]` is `sum_k (x_ki - mu_i) * conj(x_kj - mu_j) / (n - ddof)`, so
# the result is symmetric for a real element and Hermitian for a complex one.


def _centred[
    T: Numeric, origin: Origin[mut=False], //
](m: MatrixView[T, origin], axis: Int) raises -> List[T]:
    """Returns the data with each variable's mean subtracted, variable-major."""
    var p = _lane_count(m.nrows(), m.ncols(), axis)
    var n = _lane_len(m.nrows(), m.ncols(), axis)
    var c = List[T](capacity=p * n)
    for i in range(p):
        var mu = _mean_lane(m, axis, i)
        for k in range(n):
            var ij = _coords(m.ncols(), axis, i, k)
            c.append(m[ij[0], ij[1]] - mu)
    return c^


def _cov_core[
    T: Numeric,
    origin: Origin[mut=False],
    //,
    mul_conj: def(T, T) raises thin -> T,
](m: MatrixView[T, origin], rowvar: Bool, ddof: Int) raises -> Matrix[T]:
    var axis = 1 if rowvar else 0
    var p = _lane_count(m.nrows(), m.ncols(), axis)
    var n = _lane_len(m.nrows(), m.ncols(), axis)
    _check_dof("cov()", n, ddof)
    var c = _centred(m, axis)
    var denom = _from_count[T](n - ddof)
    var data = List[T](capacity=p * p)
    for i in range(p):
        for j in range(p):
            var acc = T.zero()
            for k in range(n):
                acc = acc + mul_conj(c[i * n + k], c[j * n + k])
            data.append(acc / denom)
    return Matrix[T](buffer=data^, nrows=p, ncols=p, row_stride=p, col_stride=1)


def _corrcoef_core[
    T: Numeric & Rootable,
    origin: Origin[mut=False],
    //,
    mul_conj: def(T, T) raises thin -> T,
](m: MatrixView[T, origin], rowvar: Bool) raises -> Matrix[T]:
    var c = _cov_core[mul_conj=mul_conj](m, rowvar, 1)
    var p = c.nrows()
    var sd = List[T](capacity=p)
    for i in range(p):
        sd.append(c[i, i].sqrt())
    var data = List[T](capacity=p * p)
    for i in range(p):
        for j in range(p):
            data.append(c[i, j] / sd[i] / sd[j])
    return Matrix[T](buffer=data^, nrows=p, ncols=p, row_stride=p, col_stride=1)


def cov[
    dtype: DType, origin: Origin[mut=False], //
](
    m: MatrixView[Scalar[dtype], origin], rowvar: Bool = False, *, ddof: Int = 1
) raises -> Matrix[Scalar[dtype]]:
    """Returns the covariance matrix of the variables in `m`.

    Parameters:
        dtype: The data type of the matrix elements. Must be floating-point.
        origin: The origin of the operand.

    Args:
        m: The data. Each column is a variable and each row an observation,
            unless `rowvar` is set.
        rowvar: If True, each row is a variable instead, as in NumPy's default.
        ddof: Delta degrees of freedom. The default 1 gives the unbiased
            sample covariance; 0 gives the population one.

    Returns:
        A new `p x p` symmetric matrix, `p` the number of variables.

    Raises:
        ValueError: If the number of observations minus `ddof` is not
            positive.
    """
    _check_float[dtype]()
    var axis = 1 if rowvar else 0
    var p = _lane_count(m.nrows(), m.ncols(), axis)
    var n = _lane_len(m.nrows(), m.ncols(), axis)
    _check_dof("cov()", n, ddof)

    var c = List[Scalar[dtype]](capacity=p * n)
    for i in range(p):
        var mu = _mean_lane(m, axis, i)
        for k in range(n):
            var ij = _coords(m.ncols(), axis, i, k)
            c.append(m[ij[0], ij[1]] - mu)

    # Symmetric, so each off-diagonal pair is computed once.
    var denom = Scalar[dtype](n - ddof)
    var data = List[Scalar[dtype]](length=p * p, fill=0)
    for i in range(p):
        for j in range(i, p):
            var acc = Scalar[dtype](0)
            for k in range(n):
                acc += c[i * n + k] * c[j * n + k]
            data[i * p + j] = acc / denom
            data[j * p + i] = data[i * p + j]
    return Matrix[Scalar[dtype]](
        buffer=data^, nrows=p, ncols=p, row_stride=p, col_stride=1
    )


def cov[
    T: Numeric, origin: Origin[mut=False], //
](
    m: MatrixView[T, origin], rowvar: Bool = False, *, ddof: Int = 1
) raises -> Matrix[T] where not conforms_to(T, Conjugable):
    """Returns the covariance matrix of the variables in `m`.

    Parameters:
        T: The type of the matrix elements.
        origin: The origin of the operand.

    Args:
        m: The data, one variable per column unless `rowvar` is set.
        rowvar: If True, each row is a variable instead.
        ddof: Delta degrees of freedom; 1 by default.

    Returns:
        A new `p x p` symmetric matrix, `p` the number of variables.

    Raises:
        ValueError: If the number of observations minus `ddof` is not
            positive.
    """
    _check_divides[T]()
    return _cov_core[mul_conj=_mul_plain[T]](m, rowvar, ddof)


def cov[
    d: DType, origin: Origin[mut=False], //
](
    m: MatrixView[Complex[d], origin], rowvar: Bool = False, *, ddof: Int = 1
) raises -> Matrix[Complex[d]]:
    """Returns the covariance matrix of complex variables, which is Hermitian.

    Parameters:
        d: The component dtype of the matrix elements.
        origin: The origin of the operand.

    Args:
        m: The data, one variable per column unless `rowvar` is set.
        rowvar: If True, each row is a variable instead.
        ddof: Delta degrees of freedom; 1 by default.

    Returns:
        A new `p x p` Hermitian matrix, `p` the number of variables.

    Raises:
        ValueError: If the number of observations minus `ddof` is not
            positive.
    """
    return _cov_core[mul_conj=_mul_conj[d]](m, rowvar, ddof)


# TODO: consider adding Spearman correlation matrix
def corrcoef[
    dtype: DType, origin: Origin[mut=False], //
](m: MatrixView[Scalar[dtype], origin], rowvar: Bool = False) raises -> Matrix[
    Scalar[dtype]
]:
    """Returns the Pearson correlation matrix of the variables in `m`.

    Entry `[i, j]` is `cov[i, j] / sqrt(cov[i, i] * cov[j, j])`, clipped to
    `[-1, 1]` against rounding as NumPy does. A variable with zero variance
    gives NaN in its row and column.

    Parameters:
        dtype: The data type of the matrix elements. Must be floating-point.
        origin: The origin of the operand.

    Args:
        m: The data, one variable per column unless `rowvar` is set.
        rowvar: If True, each row is a variable instead.

    Returns:
        A new `p x p` symmetric matrix with ones on the diagonal.

    Raises:
        ValueError: If there are fewer than two observations.
    """
    var c = cov(m, rowvar, ddof=1)
    var p = c.nrows()
    var sd = List[Scalar[dtype]](capacity=p)
    for i in range(p):
        sd.append(sqrt(c[i, i]))
    var data = List[Scalar[dtype]](capacity=p * p)
    for i in range(p):
        for j in range(p):
            var r = c[i, j] / sd[i] / sd[j]
            if r > 1:
                r = 1
            elif r < -1:
                r = -1
            data.append(r)
    return Matrix[Scalar[dtype]](
        buffer=data^, nrows=p, ncols=p, row_stride=p, col_stride=1
    )


def corrcoef[
    T: Numeric & Rootable, origin: Origin[mut=False], //
](m: MatrixView[T, origin], rowvar: Bool = False) raises -> Matrix[
    T
] where not conforms_to(T, Conjugable):
    """Returns the Pearson correlation matrix of the variables in `m`.

    Parameters:
        T: The type of the matrix elements.
        origin: The origin of the operand.

    Args:
        m: The data, one variable per column unless `rowvar` is set.
        rowvar: If True, each row is a variable instead.

    Returns:
        A new `p x p` symmetric matrix with ones on the diagonal.

    Raises:
        ValueError: If there are fewer than two observations.
        ZeroDivisionError: If a variable has zero variance.
    """
    _check_divides[T]()
    return _corrcoef_core[mul_conj=_mul_plain[T]](m, rowvar)


def corrcoef[
    d: DType, origin: Origin[mut=False], //
](m: MatrixView[Complex[d], origin], rowvar: Bool = False) raises -> Matrix[
    Complex[d]
]:
    """Returns the correlation matrix of complex variables, which is Hermitian.

    Parameters:
        d: The component dtype of the matrix elements.
        origin: The origin of the operand.

    Args:
        m: The data, one variable per column unless `rowvar` is set.
        rowvar: If True, each row is a variable instead.

    Returns:
        A new `p x p` Hermitian matrix with ones on the diagonal.

    Raises:
        ValueError: If there are fewer than two observations.
    """
    return _corrcoef_core[mul_conj=_mul_conj[d]](m, rowvar)
