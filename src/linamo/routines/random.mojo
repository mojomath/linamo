"""
Random matrix generation.

`rand` is the one entry point Phase 5.5 needs; `randn` and the rest of the
distribution family arrive in Phase 9. `seed` is here because a generator you
cannot pin is a generator you cannot test --- it forwards to the Mojo standard
library's global RNG, which is what `rand` draws from.
"""

from std.random import random_float64, random_si64, seed as _seed_rng

from linamo.types.errors import ValueError
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


def rand[
    dtype: DType = DType.float64, //, T: Copyable & Deinitable = Scalar[dtype]
](
    nrows: Int,
    ncols: Int,
    low: T = rebind[T](Scalar[dtype](0)),
    high: T = rebind[T](Scalar[dtype](1)),
) raises -> Matrix[Scalar[dtype]] where (T == Scalar[dtype]):
    """Creates a matrix of uniformly distributed random values in `[low, high]`.

    Draws from the Mojo standard library's global generator, so `seed(n)` makes
    the result reproducible.

    Parameters:
        dtype: The dtype behind `T`, deduced rather than written.
        T: The type of the matrix elements. Defaults to `Float64`.

    Args:
        nrows: The number of rows in the matrix.
        ncols: The number of columns in the matrix.
        low: The lower bound of the range. Defaults to 0.
        high: The upper bound of the range. Defaults to 1.

    Returns:
        A new C-contiguous `nrows x ncols` matrix of random values.

    Raises:
        ValueError: If `low` is greater than `high`.
    """
    comptime fn_name = "rand(nrows, ncols, low, high)"
    # `T` is `Scalar[dtype]`, but the compiler does not refine it inside the
    # body, so the two bounds are restated once.
    var low_ = rebind[Scalar[dtype]](low)
    var high_ = rebind[Scalar[dtype]](high)
    if low_ > high_:
        raise ValueError(
            function=fn_name,
            message=String(
                "`low` must not exceed `high`, got low=",
                low_,
                ", high=",
                high_,
            ),
        )

    var size = nrows * ncols
    var data = List[Scalar[dtype]](unsafe_uninit_length=size)
    comptime if dtype.is_floating_point():
        var lo = Float64(low_)
        var hi = Float64(high_)
        for k in range(size):
            data[k] = Scalar[dtype](random_float64(lo, hi))
    else:
        # `random_si64` treats both bounds as inclusive, which matches the
        # closed interval this routine documents.
        var lo = Int64(low_)
        var hi = Int64(high_)
        for k in range(size):
            data[k] = Scalar[dtype](random_si64(lo, hi))
    return Matrix[Scalar[dtype]](
        buffer=data^,
        nrows=nrows,
        ncols=ncols,
        row_stride=ncols,
        col_stride=1,
    )
