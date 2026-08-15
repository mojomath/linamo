"""
Defines logical and comparison routines for matrices.

Element-wise comparisons take two operands of the same `dtype` and return a
`Matrix[DType.bool]` of the same shape — the NumPy convention, where `a > b`
is a mask rather than a single verdict. `Matrix` therefore does not conform to
`EqualityComparable`: `a == b` is an element-wise mask, and asking whether two
matrices are wholly identical is a separate question (see
`utils/test_utils.mojo`).
"""

from std.algorithm import vectorize
from std.sys import simd_width_of

from linamo.types.errors import ValueError
from linamo.types.matrix import Matrix
from linamo.types.matrix_view import MatrixView
from linamo.utils.indexing import get_offset


# --------------------------------------------------------------------------- #
# Core view-based comparison implementation
# --------------------------------------------------------------------------- #


def _compare_view[
    dtype: DType,
    func: def(Scalar[dtype], Scalar[dtype]) thin -> Scalar[DType.bool],
    origin_a: Origin,
    origin_b: Origin,
](
    a: MatrixView[dtype, origin_a],
    b: MatrixView[dtype, origin_b],
) raises -> Matrix[DType.bool]:
    """Core element-wise comparison of two MatrixView operands.

    When both operands are C-contiguous, a SIMD-vectorised fast path is taken.
    Otherwise, a stride-aware double loop is used.

    The comparison itself is applied lane-by-lane rather than with SIMD's
    `gt`/`lt` methods: the kernel arrives as a `Scalar`-to-`Scalar` parameter,
    and Mojo reserves the `<`/`>` operators for `Scalar` widths anyway. The
    resulting boolean vector is still stored in one shot.

    The result is always a freshly allocated, C-contiguous `Matrix[bool]`.
    """
    if a.nrows != b.nrows or a.ncols != b.ncols:
        raise ValueError(
            file="src/linamo/routines/logic.mojo",
            function="_compare_view()",
            message="Input matrices must have the same shape.",
            previous_error=None,
        )
    var M = a.nrows
    var N = a.ncols
    var total = M * N
    var result = Matrix[DType.bool](M, N, N, 1)

    if a.is_c_contiguous() and b.is_c_contiguous():
        comptime simd_w = simd_width_of[dtype]()
        var a_ptr = a.data.unsafe_ptr()
        var b_ptr = b.data.unsafe_ptr()
        var a_off = a.offset
        var b_off = b.offset

        def vec_cmp[
            w: Int
        ](idx: Int) {mut result, imm a_ptr, imm b_ptr, imm a_off, imm b_off,}:
            var a_chunk = a_ptr.unsafe_load[width=w](a_off + idx)
            var b_chunk = b_ptr.unsafe_load[width=w](b_off + idx)
            var res = SIMD[DType.bool, w](fill=False)

            comptime for lane in range(w):
                res[lane] = func(a_chunk[lane], b_chunk[lane])
            result.data._data.unsafe_store[width=w](idx, res)

        vectorize[simd_w](total, vec_cmp)
    else:
        for i in range(M):
            for j in range(N):
                result.data[i * N + j] = func(a[i, j], b[i, j])

    return result^


# --------------------------------------------------------------------------- #
# Core view-based scalar comparison implementation
# --------------------------------------------------------------------------- #


def _scalar_compare_view[
    dtype: DType,
    func: def(Scalar[dtype], Scalar[dtype]) thin -> Scalar[DType.bool],
    origin: Origin,
](mat: MatrixView[dtype, origin], scalar: Scalar[dtype]) -> Matrix[DType.bool]:
    """Core element-wise comparison of a MatrixView against a scalar.

    The result is always a freshly allocated, C-contiguous `Matrix[bool]`.
    """
    var M = mat.nrows
    var N = mat.ncols
    var total = M * N
    var result = Matrix[DType.bool](M, N, N, 1)

    if mat.is_c_contiguous():
        comptime simd_w = simd_width_of[dtype]()
        var m_ptr = mat.data.unsafe_ptr()
        var m_off = mat.offset

        def vec_cmp[
            w: Int
        ](idx: Int) {mut result, imm m_ptr, imm m_off, imm scalar,}:
            var m_chunk = m_ptr.unsafe_load[width=w](m_off + idx)
            var res = SIMD[DType.bool, w](fill=False)

            comptime for lane in range(w):
                res[lane] = func(m_chunk[lane], scalar)
            result.data._data.unsafe_store[width=w](idx, res)

        vectorize[simd_w](total, vec_cmp)
    else:
        for i in range(M):
            for j in range(N):
                result.data[i * N + j] = func(mat[i, j], scalar)

    return result^


# ===---------------------------------------------------------------------- ===#
# Element-wise comparisons
# ===---------------------------------------------------------------------- ===#
# Each comparison has 4 matrix-matrix overloads (view x view, mat x mat,
# mat x view, view x mat) plus 2 scalar overloads, mirroring the layout of
# `routines/math.mojo`. All return `Matrix[DType.bool]`.


# --------------------------------------------------------------------------- #
# greater
# --------------------------------------------------------------------------- #


def greater[
    dtype: DType, origin_a: Origin, origin_b: Origin
](
    a: MatrixView[dtype, origin_a], b: MatrixView[dtype, origin_b]
) raises -> Matrix[DType.bool]:
    """Element-wise greater-than comparison of two matrix views (`a > b`)."""
    return _compare_view[func=Scalar[dtype].__gt__](a, b)


def greater[
    dtype: DType
](a: Matrix[dtype], b: Matrix[dtype]) raises -> Matrix[DType.bool]:
    """Element-wise greater-than comparison of two matrices (`a > b`)."""
    return _compare_view[func=Scalar[dtype].__gt__](a.view(), b.view())


def greater[
    dtype: DType, origin_b: Origin
](a: Matrix[dtype], b: MatrixView[dtype, origin_b]) raises -> Matrix[
    DType.bool
]:
    """Element-wise greater-than comparison of a matrix and a view (`a > b`)."""
    return _compare_view[func=Scalar[dtype].__gt__](a.view(), b)


def greater[
    dtype: DType, origin_a: Origin
](a: MatrixView[dtype, origin_a], b: Matrix[dtype]) raises -> Matrix[
    DType.bool
]:
    """Element-wise greater-than comparison of a view and a matrix (`a > b`)."""
    return _compare_view[func=Scalar[dtype].__gt__](a, b.view())


def scalar_greater[
    dtype: DType
](mat: Matrix[dtype], scalar: Scalar[dtype]) -> Matrix[DType.bool]:
    """Element-wise greater-than comparison of a matrix against a scalar."""
    return _scalar_compare_view[func=Scalar[dtype].__gt__](mat.view(), scalar)


def scalar_greater[
    dtype: DType, origin: Origin
](mat: MatrixView[dtype, origin], scalar: Scalar[dtype]) -> Matrix[DType.bool]:
    """Element-wise greater-than comparison of a matrix view against a scalar.
    """
    return _scalar_compare_view[func=Scalar[dtype].__gt__](mat, scalar)


# --------------------------------------------------------------------------- #
# greater_equal
# --------------------------------------------------------------------------- #


def greater_equal[
    dtype: DType, origin_a: Origin, origin_b: Origin
](
    a: MatrixView[dtype, origin_a], b: MatrixView[dtype, origin_b]
) raises -> Matrix[DType.bool]:
    """Element-wise greater-than-or-equal comparison of two matrix views (`a >= b`).
    """
    return _compare_view[func=Scalar[dtype].__ge__](a, b)


def greater_equal[
    dtype: DType
](a: Matrix[dtype], b: Matrix[dtype]) raises -> Matrix[DType.bool]:
    """Element-wise greater-than-or-equal comparison of two matrices (`a >= b`).
    """
    return _compare_view[func=Scalar[dtype].__ge__](a.view(), b.view())


def greater_equal[
    dtype: DType, origin_b: Origin
](a: Matrix[dtype], b: MatrixView[dtype, origin_b]) raises -> Matrix[
    DType.bool
]:
    """Element-wise greater-than-or-equal comparison of a matrix and a view (`a >= b`).
    """
    return _compare_view[func=Scalar[dtype].__ge__](a.view(), b)


def greater_equal[
    dtype: DType, origin_a: Origin
](a: MatrixView[dtype, origin_a], b: Matrix[dtype]) raises -> Matrix[
    DType.bool
]:
    """Element-wise greater-than-or-equal comparison of a view and a matrix (`a >= b`).
    """
    return _compare_view[func=Scalar[dtype].__ge__](a, b.view())


def scalar_greater_equal[
    dtype: DType
](mat: Matrix[dtype], scalar: Scalar[dtype]) -> Matrix[DType.bool]:
    """Element-wise greater-than-or-equal comparison of a matrix against a scalar.
    """
    return _scalar_compare_view[func=Scalar[dtype].__ge__](mat.view(), scalar)


def scalar_greater_equal[
    dtype: DType, origin: Origin
](mat: MatrixView[dtype, origin], scalar: Scalar[dtype]) -> Matrix[DType.bool]:
    """Element-wise greater-than-or-equal comparison of a matrix view against a scalar.
    """
    return _scalar_compare_view[func=Scalar[dtype].__ge__](mat, scalar)


# --------------------------------------------------------------------------- #
# less
# --------------------------------------------------------------------------- #


def less[
    dtype: DType, origin_a: Origin, origin_b: Origin
](
    a: MatrixView[dtype, origin_a], b: MatrixView[dtype, origin_b]
) raises -> Matrix[DType.bool]:
    """Element-wise less-than comparison of two matrix views (`a < b`)."""
    return _compare_view[func=Scalar[dtype].__lt__](a, b)


def less[
    dtype: DType
](a: Matrix[dtype], b: Matrix[dtype]) raises -> Matrix[DType.bool]:
    """Element-wise less-than comparison of two matrices (`a < b`)."""
    return _compare_view[func=Scalar[dtype].__lt__](a.view(), b.view())


def less[
    dtype: DType, origin_b: Origin
](a: Matrix[dtype], b: MatrixView[dtype, origin_b]) raises -> Matrix[
    DType.bool
]:
    """Element-wise less-than comparison of a matrix and a view (`a < b`)."""
    return _compare_view[func=Scalar[dtype].__lt__](a.view(), b)


def less[
    dtype: DType, origin_a: Origin
](a: MatrixView[dtype, origin_a], b: Matrix[dtype]) raises -> Matrix[
    DType.bool
]:
    """Element-wise less-than comparison of a view and a matrix (`a < b`)."""
    return _compare_view[func=Scalar[dtype].__lt__](a, b.view())


def scalar_less[
    dtype: DType
](mat: Matrix[dtype], scalar: Scalar[dtype]) -> Matrix[DType.bool]:
    """Element-wise less-than comparison of a matrix against a scalar."""
    return _scalar_compare_view[func=Scalar[dtype].__lt__](mat.view(), scalar)


def scalar_less[
    dtype: DType, origin: Origin
](mat: MatrixView[dtype, origin], scalar: Scalar[dtype]) -> Matrix[DType.bool]:
    """Element-wise less-than comparison of a matrix view against a scalar."""
    return _scalar_compare_view[func=Scalar[dtype].__lt__](mat, scalar)


# --------------------------------------------------------------------------- #
# less_equal
# --------------------------------------------------------------------------- #


def less_equal[
    dtype: DType, origin_a: Origin, origin_b: Origin
](
    a: MatrixView[dtype, origin_a], b: MatrixView[dtype, origin_b]
) raises -> Matrix[DType.bool]:
    """Element-wise less-than-or-equal comparison of two matrix views (`a <= b`).
    """
    return _compare_view[func=Scalar[dtype].__le__](a, b)


def less_equal[
    dtype: DType
](a: Matrix[dtype], b: Matrix[dtype]) raises -> Matrix[DType.bool]:
    """Element-wise less-than-or-equal comparison of two matrices (`a <= b`)."""
    return _compare_view[func=Scalar[dtype].__le__](a.view(), b.view())


def less_equal[
    dtype: DType, origin_b: Origin
](a: Matrix[dtype], b: MatrixView[dtype, origin_b]) raises -> Matrix[
    DType.bool
]:
    """Element-wise less-than-or-equal comparison of a matrix and a view (`a <= b`).
    """
    return _compare_view[func=Scalar[dtype].__le__](a.view(), b)


def less_equal[
    dtype: DType, origin_a: Origin
](a: MatrixView[dtype, origin_a], b: Matrix[dtype]) raises -> Matrix[
    DType.bool
]:
    """Element-wise less-than-or-equal comparison of a view and a matrix (`a <= b`).
    """
    return _compare_view[func=Scalar[dtype].__le__](a, b.view())


def scalar_less_equal[
    dtype: DType
](mat: Matrix[dtype], scalar: Scalar[dtype]) -> Matrix[DType.bool]:
    """Element-wise less-than-or-equal comparison of a matrix against a scalar.
    """
    return _scalar_compare_view[func=Scalar[dtype].__le__](mat.view(), scalar)


def scalar_less_equal[
    dtype: DType, origin: Origin
](mat: MatrixView[dtype, origin], scalar: Scalar[dtype]) -> Matrix[DType.bool]:
    """Element-wise less-than-or-equal comparison of a matrix view against a scalar.
    """
    return _scalar_compare_view[func=Scalar[dtype].__le__](mat, scalar)


# --------------------------------------------------------------------------- #
# equal
# --------------------------------------------------------------------------- #


def equal[
    dtype: DType, origin_a: Origin, origin_b: Origin
](
    a: MatrixView[dtype, origin_a], b: MatrixView[dtype, origin_b]
) raises -> Matrix[DType.bool]:
    """Element-wise equality comparison of two matrix views (`a == b`)."""
    return _compare_view[func=Scalar[dtype].__eq__](a, b)


def equal[
    dtype: DType
](a: Matrix[dtype], b: Matrix[dtype]) raises -> Matrix[DType.bool]:
    """Element-wise equality comparison of two matrices (`a == b`)."""
    return _compare_view[func=Scalar[dtype].__eq__](a.view(), b.view())


def equal[
    dtype: DType, origin_b: Origin
](a: Matrix[dtype], b: MatrixView[dtype, origin_b]) raises -> Matrix[
    DType.bool
]:
    """Element-wise equality comparison of a matrix and a view (`a == b`)."""
    return _compare_view[func=Scalar[dtype].__eq__](a.view(), b)


def equal[
    dtype: DType, origin_a: Origin
](a: MatrixView[dtype, origin_a], b: Matrix[dtype]) raises -> Matrix[
    DType.bool
]:
    """Element-wise equality comparison of a view and a matrix (`a == b`)."""
    return _compare_view[func=Scalar[dtype].__eq__](a, b.view())


def scalar_equal[
    dtype: DType
](mat: Matrix[dtype], scalar: Scalar[dtype]) -> Matrix[DType.bool]:
    """Element-wise equality comparison of a matrix against a scalar."""
    return _scalar_compare_view[func=Scalar[dtype].__eq__](mat.view(), scalar)


def scalar_equal[
    dtype: DType, origin: Origin
](mat: MatrixView[dtype, origin], scalar: Scalar[dtype]) -> Matrix[DType.bool]:
    """Element-wise equality comparison of a matrix view against a scalar."""
    return _scalar_compare_view[func=Scalar[dtype].__eq__](mat, scalar)


# --------------------------------------------------------------------------- #
# not_equal
# --------------------------------------------------------------------------- #


def not_equal[
    dtype: DType, origin_a: Origin, origin_b: Origin
](
    a: MatrixView[dtype, origin_a], b: MatrixView[dtype, origin_b]
) raises -> Matrix[DType.bool]:
    """Element-wise inequality comparison of two matrix views (`a != b`)."""
    return _compare_view[func=Scalar[dtype].__ne__](a, b)


def not_equal[
    dtype: DType
](a: Matrix[dtype], b: Matrix[dtype]) raises -> Matrix[DType.bool]:
    """Element-wise inequality comparison of two matrices (`a != b`)."""
    return _compare_view[func=Scalar[dtype].__ne__](a.view(), b.view())


def not_equal[
    dtype: DType, origin_b: Origin
](a: Matrix[dtype], b: MatrixView[dtype, origin_b]) raises -> Matrix[
    DType.bool
]:
    """Element-wise inequality comparison of a matrix and a view (`a != b`)."""
    return _compare_view[func=Scalar[dtype].__ne__](a.view(), b)


def not_equal[
    dtype: DType, origin_a: Origin
](a: MatrixView[dtype, origin_a], b: Matrix[dtype]) raises -> Matrix[
    DType.bool
]:
    """Element-wise inequality comparison of a view and a matrix (`a != b`)."""
    return _compare_view[func=Scalar[dtype].__ne__](a, b.view())


def scalar_not_equal[
    dtype: DType
](mat: Matrix[dtype], scalar: Scalar[dtype]) -> Matrix[DType.bool]:
    """Element-wise inequality comparison of a matrix against a scalar."""
    return _scalar_compare_view[func=Scalar[dtype].__ne__](mat.view(), scalar)


def scalar_not_equal[
    dtype: DType, origin: Origin
](mat: MatrixView[dtype, origin], scalar: Scalar[dtype]) -> Matrix[DType.bool]:
    """Element-wise inequality comparison of a matrix view against a scalar."""
    return _scalar_compare_view[func=Scalar[dtype].__ne__](mat, scalar)
