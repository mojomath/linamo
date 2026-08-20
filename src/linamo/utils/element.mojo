"""
Compile-time facts about a matrix element type.
"""


def dtype_of[T: AnyType]() -> DType:
    """Returns the `DType` behind a scalar element type.

    `Scalar[d]` is the element type and `d` is the dtype behind it, and most of
    the library moves in that direction: a routine written against
    `Matrix[Scalar[d]]` has `d` inferred for it. This is the other direction,
    for the one place that needs it --- `StaticMatrix`, whose buffer is a
    single SIMD register and so has to name the dtype where the field is
    declared, while its parameter stays an element type like every other
    matrix type's.

    A type that is not a scalar is a compile error rather than a wrong answer.

    Parameters:
        T: The element type. Must be some `Scalar[d]`.

    Returns:
        The `d` for which `T` is `Scalar[d]`.
    """
    comptime if T == Scalar[DType.bool]:
        return DType.bool
    elif T == Int8:
        return DType.int8
    elif T == UInt8:
        return DType.uint8
    elif T == Int16:
        return DType.int16
    elif T == UInt16:
        return DType.uint16
    elif T == Int32:
        return DType.int32
    elif T == UInt32:
        return DType.uint32
    elif T == Int64:
        return DType.int64
    elif T == UInt64:
        return DType.uint64
    elif T == Int128:
        return DType.int128
    elif T == UInt128:
        return DType.uint128
    elif T == Int:
        return DType.int
    elif T == UInt:
        return DType.uint
    elif T == BFloat16:
        return DType.bfloat16
    elif T == Float16:
        return DType.float16
    elif T == Float32:
        return DType.float32
    elif T == Float64:
        return DType.float64
    else:
        comptime assert False, (
            "`StaticMatrix` holds one SIMD register, so its element type has"
            " to be a scalar. Use `Matrix` for an element type that is not."
        )
