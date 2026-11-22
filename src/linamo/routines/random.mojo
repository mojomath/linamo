"""
Random matrix generation.

`rand` draws from a uniform distribution and `randn` from a normal one.
`seed` is here because a generator you cannot pin is a generator you cannot
test --- it forwards to the Mojo standard library's global RNG, which is what
both draw from.
"""

from std.math import floor
from std.random import (
    randn_float64,
    random_float64,
    random_si64,
    seed as _seed_rng,
)

from linamo.errors import ValueError
from linamo.types.matrix import Matrix


def seed():
    """Seeds the global random number generator from the current time."""
    _seed_rng()


def seed(value: Int):
    """Seeds the global random number generator with a fixed value.

    Two runs that seed with the same value draw the same numbers, which is what
    makes a test over `rand` reproducible.

    Args:
        value: The seed.
    """
    _seed_rng(value)


# `low` and `high` have dtypes of their own, separate from the element's.
# Were they typed `Scalar[dtype]`, a literal such as `1.0` would fix `dtype` to
# float64, and `rand[Float32](2, 2, 1.0, 2.0)` would fail the `where` clause.
# Kept apart, they cannot clash with `T`. The body never uses `T` either, only
# `dtype`, so it needs no `rebind`: `T` is there so that callers write
# `rand[Float32]`, as they write `Matrix[Float32]`, and the `where` clause
# ties it to `dtype`.
#
# The price is that the bounds are no longer checked against the element type
# at compile time. So for an integer element type the body checks at run time
# that each bound is a whole number the type can hold.
def rand[
    low_dtype: DType = DType.float64,
    high_dtype: DType = DType.float64,
    dtype: DType = DType.float64,
    //,
    T: Copyable & Deinitable = Scalar[dtype],
](
    nrows: Int,
    ncols: Int,
    low: Scalar[low_dtype] = 0,
    high: Scalar[high_dtype] = 1,
) raises -> Matrix[Scalar[dtype]] where (T == Scalar[dtype]):
    """Creates a matrix of uniformly distributed random values in `[low, high]`.

    Draws from the Mojo standard library's global generator, so `seed(n)` makes
    the result reproducible.

    Parameters:
        low_dtype: The dtype of `low`, deduced rather than written.
        high_dtype: The dtype of `high`, deduced rather than written.
        dtype: The dtype behind `T`, deduced rather than written.
        T: The type of the matrix elements. Defaults to `Float64`.

    Args:
        nrows: The number of rows in the matrix.
        ncols: The number of columns in the matrix.
        low: The lower bound of the range, of any scalar type. Defaults to 0.
        high: The upper bound of the range, of any scalar type. Defaults
            to 1.

    Returns:
        A new C-contiguous `nrows x ncols` matrix of random values.

    Raises:
        ValueError: If `low` is greater than `high`, or if the element type
            is an integer and a bound is not a whole number it can hold.
    """
    comptime fn_name = "rand(nrows, ncols, low, high)"
    var lo = Float64(low)
    var hi = Float64(high)
    if lo > hi:
        raise ValueError(
            function=fn_name,
            message=String(
                "`low` must not exceed `high`, got low=", low, ", high=", high
            ),
        )

    var size = nrows * ncols
    var data = List[Scalar[dtype]](unsafe_uninit_length=size)
    comptime if dtype.is_floating_point():
        for k in range(size):
            data[k] = Scalar[dtype](random_float64(lo, hi))
    else:
        var smallest = Float64(Scalar[dtype].MIN)
        var largest = Float64(Scalar[dtype].MAX)
        for bound in [lo, hi]:
            if bound != floor(bound) or bound < smallest or bound > largest:
                raise ValueError(
                    function=fn_name,
                    message=String(
                        "For ",
                        dtype,
                        (
                            " elements, `low` and `high` must be whole numbers"
                            " from "
                        ),
                        Scalar[dtype].MIN,
                        " to ",
                        Scalar[dtype].MAX,
                        ", got low=",
                        low,
                        ", high=",
                        high,
                    ),
                )
        # `random_si64` treats both bounds as inclusive, which matches the
        # closed interval this routine documents.
        for k in range(size):
            data[k] = Scalar[dtype](random_si64(Int64(lo), Int64(hi)))
    return Matrix[Scalar[dtype]](
        buffer=data^,
        nrows=nrows,
        ncols=ncols,
        row_stride=ncols,
        col_stride=1,
    )


# As in `rand`, `mean` and `stddev` have dtypes of their own and the body
# needs no `rebind`.
def randn[
    mean_dtype: DType = DType.float64,
    stddev_dtype: DType = DType.float64,
    dtype: DType = DType.float64,
    //,
    T: Copyable & Deinitable = Scalar[dtype],
](
    nrows: Int,
    ncols: Int,
    mean: Scalar[mean_dtype] = 0,
    stddev: Scalar[stddev_dtype] = 1,
) raises -> Matrix[Scalar[dtype]] where (T == Scalar[dtype]):
    """Creates a matrix of normally distributed random values.

    With the defaults the values follow the standard normal distribution, as
    NumPy's `numpy.random.randn` does. `mean` and `stddev` shift and scale it,
    as the `loc` and `scale` of `numpy.random.normal` do.

    Draws from the Mojo standard library's global generator, so `seed(n)` makes
    the result reproducible.

    Parameters:
        mean_dtype: The dtype of `mean`, deduced rather than written.
        stddev_dtype: The dtype of `stddev`, deduced rather than written.
        dtype: The dtype behind `T`, deduced rather than written.
        T: The type of the matrix elements. Must be floating-point. Defaults
            to `Float64`.

    Args:
        nrows: The number of rows in the matrix.
        ncols: The number of columns in the matrix.
        mean: The mean of the distribution, of any scalar type. Defaults
            to 0.
        stddev: The standard deviation of the distribution, of any scalar
            type. Defaults to 1.

    Returns:
        A new C-contiguous `nrows x ncols` matrix of random values.

    Raises:
        ValueError: If `stddev` is negative.
    """
    comptime assert dtype.is_floating_point(), (
        "`randn` needs a floating-point element type: a normal draw is not an"
        " integer. Use `rand` for random integers."
    )
    comptime fn_name = "randn(nrows, ncols, mean, stddev)"
    # The generator draws in float64, so both are widened to it.
    var mean_ = Float64(mean)
    var stddev_ = Float64(stddev)
    if stddev_ < 0:
        raise ValueError(
            function=fn_name,
            message=String("`stddev` must not be negative, got ", stddev_),
        )

    var size = nrows * ncols
    var data = List[Scalar[dtype]](unsafe_uninit_length=size)
    for k in range(size):
        data[k] = Scalar[dtype](randn_float64(mean_, stddev_))
    return Matrix[Scalar[dtype]](
        buffer=data^,
        nrows=nrows,
        ncols=ncols,
        row_stride=ncols,
        col_stride=1,
    )
