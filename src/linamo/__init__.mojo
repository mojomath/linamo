# Element-type aliases.
# `int` and `uint` in particular are names no library should be
# claiming unqualified. Each type has a NumPy-style long name and a
# Rust-style short one; they are the same alias, so pick one per codebase.
#
# These name element *types*, not `DType` values: `Matrix[la.f64]` is
# `Matrix[Float64]` is `Matrix[Scalar[DType.float64]]`, all one type. The
# stdlib already spells most of them (`Float64`, `Int32`, ...), so these exist
# for the lowercase NumPy register and for `bool_`, which the stdlib has no
# name for --- `Bool` is a different type from `Scalar[DType.bool]`.

comptime float64 = Float64
"""64-bit floating point. Short form: `f64`."""
comptime f64 = Float64
"""64-bit floating point. Long form: `float64`."""
comptime float32 = Float32
"""32-bit floating point. Short form: `f32`."""
comptime f32 = Float32
"""32-bit floating point. Long form: `float32`."""

comptime int64 = Int64
"""64-bit signed integer. Short form: `i64`."""
comptime i64 = Int64
"""64-bit signed integer. Long form: `int64`."""
comptime int32 = Int32
"""32-bit signed integer. Short form: `i32`."""
comptime i32 = Int32
"""32-bit signed integer. Long form: `int32`."""
comptime int16 = Int16
"""16-bit signed integer. Short form: `i16`."""
comptime i16 = Int16
"""16-bit signed integer. Long form: `int16`."""
comptime int8 = Int8
"""8-bit signed integer. Short form: `i8`."""
comptime i8 = Int8
"""8-bit signed integer. Long form: `int8`."""

comptime uint64 = UInt64
"""64-bit unsigned integer. Short form: `u64`."""
comptime u64 = UInt64
"""64-bit unsigned integer. Long form: `uint64`."""
comptime uint32 = UInt32
"""32-bit unsigned integer. Short form: `u32`."""
comptime u32 = UInt32
"""32-bit unsigned integer. Long form: `uint32`."""
comptime uint16 = UInt16
"""16-bit unsigned integer. Short form: `u16`."""
comptime u16 = UInt16
"""16-bit unsigned integer. Long form: `uint16`."""
comptime uint8 = UInt8
"""8-bit unsigned integer. Short form: `u8`."""
comptime u8 = UInt8
"""8-bit unsigned integer. Long form: `uint8`."""

comptime int = Int
"""The platform's default signed integer width."""
comptime uint = UInt
"""The platform's default unsigned integer width."""

comptime bool_ = Scalar[DType.bool]
"""The element type of a comparison mask. Trailing underscore as in NumPy,
because `bool` is taken --- and because Mojo's `Bool` is a distinct type that
a matrix does not store."""

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
from linamo.routines.math import max, min, prod, cumprod
from linamo.routines.random import rand, seed
from linamo.routines.searching import argmax, argmin
from linamo.routines.sorting import argsort, sort, sort_inplace
from linamo.routines.statistics import cumsum, sum
from linamo.routines.numpy_interop import from_numpy, to_numpy
from linamo.utils.test_utils import (
    assert_matrices_equal,
    assert_matrices_close,
)
