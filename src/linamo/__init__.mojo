# ===---------------------------------------------------------------------- ===#
# Element types
# ===---------------------------------------------------------------------- ===#
# A matrix is parameterised on an element *type*, not on a `DType`, so the
# element of a `Matrix[Float64]` is spelled with the stdlib's own name. That
# leaves nothing for this library to alias: `la.f64` and `Float64` named the
# same type, and one spelling is better than two.
#
# Two things do need a name here. `bool_` is the element of a comparison mask,
# which the stdlib has no name for --- `Bool` is a different type from
# `Scalar[DType.bool]`, and a matrix stores the latter. And the arbitrary-
# precision element types come from decimo, re-exported below so that using
# them takes no second import.

comptime bool_ = Scalar[DType.bool]
"""The element type of a comparison mask. Trailing underscore as in NumPy,
because `bool` is taken --- and because Mojo's `Bool` is a distinct type that
a matrix does not store."""

# Decimo's arbitrary-precision numbers, and the trait they conform to. These
# are exactly the element types that carry their own arithmetic: `Matrix[BInt]`
# adds, multiplies and multiplies-out with the same operators a
# `Matrix[Float64]` uses. Decimo's other types (`BigUInt`, `Rational`,
# `BigFloat`) are deliberately absent --- they do not conform to `Numeric`, so
# a matrix of them would have no arithmetic, and re-exporting them here would
# promise one.
from decimo import (
    Numeric,
    Parsable,
    BigInt,
    BInt,
    Integer,
    BigDecimal,
    BDec,
    Decimal,
    Decimal128,
    Dec128,
    RoundingMode,
    ROUND_DOWN,
    ROUND_HALF_UP,
    ROUND_HALF_DOWN,
    ROUND_HALF_EVEN,
    ROUND_UP,
    ROUND_CEILING,
    ROUND_FLOOR,
)

# ===---------------------------------------------------------------------- ===#
# Types and routines
# ===---------------------------------------------------------------------- ===#
# Everything a user reaches for is re-exported here, so `import linamo as la`
# is the only import a program needs and `la.<name>` is the only spelling.

from linamo.types.matrix import Matrix
from linamo.types.matrix_view import MatrixView
from linamo.types.static_matrix import StaticMatrix

from linamo.routines.creation import (
    arange,
    diag,
    empty,
    empty_like,
    eye,
    from_list,
    from_string,
    full,
    full_like,
    identity,
    linspace,
    matrix,
    ones,
    ones_like,
    smatrix,
    zeros,
    zeros_like,
)
from linamo.routines.linalg import (
    transpose,
    trace,
    lu,
    cholesky,
    qr,
    det,
    solve,
    inv,
    matrix_power,
    lstsq,
)
from linamo.routines.functional import apply_along_axis, fold
from linamo.routines.logic import all, any
from linamo.routines.manipulation import (
    astype,
    broadcast_to,
    contiguous,
    flatten,
    reorder_layout,
    reshape,
    reshape_view,
    resize,
)
from linamo.routines.math import div, matmul, max, min, mul, pow, prod, cumprod
from linamo.routines.mutation import (
    assign,
    cols_mut,
    fill,
    rows_mut,
    store,
    view_mut,
)
from linamo.routines.random import rand, seed
from linamo.routines.searching import argmax, argmin
from linamo.routines.sorting import argsort, sort, sort_inplace
from linamo.routines.statistics import cumsum, sum
from linamo.routines.numpy_interop import from_numpy, to_numpy
from linamo.utils.test_utils import (
    assert_matrices_equal,
    assert_matrices_close,
)
