# Element-type aliases.
# `int` and `uint` in particular are names no library should be
# claiming unqualified. Each type has a NumPy-style long name and a
# Rust-style short one; they are the same alias, so pick one per codebase.

comptime float64 = DType.float64
"""64-bit floating point. Short form: `f64`."""
comptime f64 = DType.float64
"""64-bit floating point. Long form: `float64`."""
comptime float32 = DType.float32
"""32-bit floating point. Short form: `f32`."""
comptime f32 = DType.float32
"""32-bit floating point. Long form: `float32`."""

comptime int64 = DType.int64
"""64-bit signed integer. Short form: `i64`."""
comptime i64 = DType.int64
"""64-bit signed integer. Long form: `int64`."""
comptime int32 = DType.int32
"""32-bit signed integer. Short form: `i32`."""
comptime i32 = DType.int32
"""32-bit signed integer. Long form: `int32`."""
comptime int16 = DType.int16
"""16-bit signed integer. Short form: `i16`."""
comptime i16 = DType.int16
"""16-bit signed integer. Long form: `int16`."""
comptime int8 = DType.int8
"""8-bit signed integer. Short form: `i8`."""
comptime i8 = DType.int8
"""8-bit signed integer. Long form: `int8`."""

comptime uint64 = DType.uint64
"""64-bit unsigned integer. Short form: `u64`."""
comptime u64 = DType.uint64
"""64-bit unsigned integer. Long form: `uint64`."""
comptime uint32 = DType.uint32
"""32-bit unsigned integer. Short form: `u32`."""
comptime u32 = DType.uint32
"""32-bit unsigned integer. Long form: `uint32`."""
comptime uint16 = DType.uint16
"""16-bit unsigned integer. Short form: `u16`."""
comptime u16 = DType.uint16
"""16-bit unsigned integer. Long form: `uint16`."""
comptime uint8 = DType.uint8
"""8-bit unsigned integer. Short form: `u8`."""
comptime u8 = DType.uint8
"""8-bit unsigned integer. Long form: `uint8`."""

comptime int = DType.int
"""The platform's default signed integer width."""
comptime uint = DType.uint
"""The platform's default unsigned integer width."""


from linamo.types.matrix import Matrix
from linamo.types.matrix_view import MatrixView
from linamo.types.static_matrix import StaticMatrix

from linamo.routines.creation import (
    arange,
    diag,
    empty,
    empty_like,
    eye,
    fromlist,
    fromstring,
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
from linamo.routines.numpy_interop import matrix_from_numpy, to_numpy
from linamo.utils.test_utils import (
    assert_matrices_equal,
    assert_matrices_close,
)
