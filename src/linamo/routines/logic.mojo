"""
Defines logical and comparison routines for matrices.

Element-wise comparisons take two operands of the same `dtype` and return a
`Matrix[Scalar[DType.bool]]` of the same shape — the NumPy convention, where `a > b`
is a mask rather than a single verdict. `Matrix` therefore does not conform to
`EqualityComparable`: `a == b` is an element-wise mask, and asking whether two
matrices are wholly identical is a separate question (see
`utils/test_utils.mojo`).
"""

from std.algorithm import vectorize
from std.math import isfinite, isnan
from std.sys import simd_width_of

from linamo.errors import ValueError
from linamo.types.matrix import Matrix
from linamo.types.matrix_view import MatrixView
from linamo.utils.indexing import get_offset


# --------------------------------------------------------------------------- #
# Core view-based comparison implementation
# --------------------------------------------------------------------------- #


def _compare_view[
    dtype: DType,
    origin_a: Origin,
    origin_b: Origin,
    //,
    func: def(Scalar[dtype], Scalar[dtype]) thin -> Scalar[DType.bool],
](
    a: MatrixView[Scalar[dtype], origin_a],
    b: MatrixView[Scalar[dtype], origin_b],
) raises -> Matrix[Scalar[DType.bool]]:
    """Core element-wise comparison of two MatrixView operands.

    When both operands are C-contiguous, a SIMD-vectorised fast path is taken.
    Otherwise, a stride-aware double loop is used.

    The comparison itself is applied lane-by-lane rather than with SIMD's
    `gt`/`lt` methods: the kernel arrives as a `Scalar`-to-`Scalar` parameter,
    and Mojo reserves the `<`/`>` operators for `Scalar` widths anyway. The
    resulting boolean vector is still stored in one shot.

    The result is always a freshly allocated, C-contiguous `Matrix[bool]`.
    """
    if a.nrows() != b.nrows() or a.ncols() != b.ncols():
        raise ValueError(
            function="_compare_view()",
            message="Input matrices must have the same shape.",
        )
    var M = a.nrows()
    var N = a.ncols()
    var total = M * N
    var result = Matrix[Scalar[DType.bool]](M, N, N, 1)

    if a.is_c_contiguous() and b.is_c_contiguous():
        comptime simd_w = simd_width_of[dtype]()
        var a_ptr = a._data.unsafe_ptr()
        var b_ptr = b._data.unsafe_ptr()
        var a_off = a.offset()
        var b_off = b.offset()

        def vec_cmp[
            w: Int
        ](idx: Int) {mut result, imm a_ptr, imm b_ptr, imm a_off, imm b_off,}:
            var a_chunk = a_ptr.unsafe_load[width=w](a_off + idx)
            var b_chunk = b_ptr.unsafe_load[width=w](b_off + idx)
            var res = SIMD[DType.bool, w](fill=False)

            comptime for lane in range(w):
                res[lane] = func(a_chunk[lane], b_chunk[lane])
            result._data._data.unsafe_store[width=w](idx, res)

        vectorize[simd_w](total, vec_cmp)
    else:
        for i in range(M):
            for j in range(N):
                result._data[i * N + j] = func(a[i, j], b[i, j])

    return result^


# --------------------------------------------------------------------------- #
# Core view-based scalar comparison implementation
# --------------------------------------------------------------------------- #


def _scalar_compare_view[
    dtype: DType,
    origin: Origin,
    //,
    func: def(Scalar[dtype], Scalar[dtype]) thin -> Scalar[DType.bool],
](mat: MatrixView[Scalar[dtype], origin], scalar: Scalar[dtype]) -> Matrix[
    Scalar[DType.bool]
]:
    """Core element-wise comparison of a MatrixView against a scalar.

    The result is always a freshly allocated, C-contiguous `Matrix[bool]`.
    """
    var M = mat.nrows()
    var N = mat.ncols()
    var total = M * N
    var result = Matrix[Scalar[DType.bool]](M, N, N, 1)

    if mat.is_c_contiguous():
        comptime simd_w = simd_width_of[dtype]()
        var m_ptr = mat._data.unsafe_ptr()
        var m_off = mat.offset()

        def vec_cmp[
            w: Int
        ](idx: Int) {mut result, imm m_ptr, imm m_off, imm scalar,}:
            var m_chunk = m_ptr.unsafe_load[width=w](m_off + idx)
            var res = SIMD[DType.bool, w](fill=False)

            comptime for lane in range(w):
                res[lane] = func(m_chunk[lane], scalar)
            result._data._data.unsafe_store[width=w](idx, res)

        vectorize[simd_w](total, vec_cmp)
    else:
        for i in range(M):
            for j in range(N):
                result._data[i * N + j] = func(mat[i, j], scalar)

    return result^


# ===---------------------------------------------------------------------- ===#
# Element-wise comparisons
# ===---------------------------------------------------------------------- ===#
# One signature per comparison, plus one scalar form. A `Matrix` operand
# converts implicitly to a read-only `MatrixView`, mirroring the layout of
# `routines/math.mojo`. All return `Matrix[Scalar[DType.bool]]`.


# --------------------------------------------------------------------------- #
# greater
# --------------------------------------------------------------------------- #


def greater[
    dtype: DType, origin_a: Origin, origin_b: Origin, //
](
    a: MatrixView[Scalar[dtype], origin_a],
    b: MatrixView[Scalar[dtype], origin_b],
) raises -> Matrix[Scalar[DType.bool]]:
    """Element-wise greater-than comparison, `a > b`."""
    return _compare_view[func=Scalar[dtype].__gt__](a, b)


def scalar_greater[
    dtype: DType, origin: Origin, //
](mat: MatrixView[Scalar[dtype], origin], scalar: Scalar[dtype]) -> Matrix[
    Scalar[DType.bool]
]:
    """Element-wise greater-than comparison of a matrix view against a scalar.
    """
    return _scalar_compare_view[func=Scalar[dtype].__gt__](mat, scalar)


# --------------------------------------------------------------------------- #
# greater_equal
# --------------------------------------------------------------------------- #


def greater_equal[
    dtype: DType, origin_a: Origin, origin_b: Origin, //
](
    a: MatrixView[Scalar[dtype], origin_a],
    b: MatrixView[Scalar[dtype], origin_b],
) raises -> Matrix[Scalar[DType.bool]]:
    """Element-wise greater-than-or-equal comparison, `a >= b`."""
    return _compare_view[func=Scalar[dtype].__ge__](a, b)


def scalar_greater_equal[
    dtype: DType, origin: Origin, //
](mat: MatrixView[Scalar[dtype], origin], scalar: Scalar[dtype]) -> Matrix[
    Scalar[DType.bool]
]:
    """Element-wise greater-than-or-equal comparison of a matrix view against a scalar.
    """
    return _scalar_compare_view[func=Scalar[dtype].__ge__](mat, scalar)


# --------------------------------------------------------------------------- #
# less
# --------------------------------------------------------------------------- #


def less[
    dtype: DType, origin_a: Origin, origin_b: Origin, //
](
    a: MatrixView[Scalar[dtype], origin_a],
    b: MatrixView[Scalar[dtype], origin_b],
) raises -> Matrix[Scalar[DType.bool]]:
    """Element-wise less-than comparison, `a < b`."""
    return _compare_view[func=Scalar[dtype].__lt__](a, b)


def scalar_less[
    dtype: DType, origin: Origin, //
](mat: MatrixView[Scalar[dtype], origin], scalar: Scalar[dtype]) -> Matrix[
    Scalar[DType.bool]
]:
    """Element-wise less-than comparison of a matrix view against a scalar."""
    return _scalar_compare_view[func=Scalar[dtype].__lt__](mat, scalar)


# --------------------------------------------------------------------------- #
# less_equal
# --------------------------------------------------------------------------- #


def less_equal[
    dtype: DType, origin_a: Origin, origin_b: Origin, //
](
    a: MatrixView[Scalar[dtype], origin_a],
    b: MatrixView[Scalar[dtype], origin_b],
) raises -> Matrix[Scalar[DType.bool]]:
    """Element-wise less-than-or-equal comparison, `a <= b`."""
    return _compare_view[func=Scalar[dtype].__le__](a, b)


def scalar_less_equal[
    dtype: DType, origin: Origin, //
](mat: MatrixView[Scalar[dtype], origin], scalar: Scalar[dtype]) -> Matrix[
    Scalar[DType.bool]
]:
    """Element-wise less-than-or-equal comparison of a matrix view against a scalar.
    """
    return _scalar_compare_view[func=Scalar[dtype].__le__](mat, scalar)


# --------------------------------------------------------------------------- #
# equal
# --------------------------------------------------------------------------- #


def equal[
    dtype: DType, origin_a: Origin, origin_b: Origin, //
](
    a: MatrixView[Scalar[dtype], origin_a],
    b: MatrixView[Scalar[dtype], origin_b],
) raises -> Matrix[Scalar[DType.bool]]:
    """Element-wise equality comparison, `a == b`."""
    return _compare_view[func=Scalar[dtype].__eq__](a, b)


def scalar_equal[
    dtype: DType, origin: Origin, //
](mat: MatrixView[Scalar[dtype], origin], scalar: Scalar[dtype]) -> Matrix[
    Scalar[DType.bool]
]:
    """Element-wise equality comparison of a matrix view against a scalar."""
    return _scalar_compare_view[func=Scalar[dtype].__eq__](mat, scalar)


# --------------------------------------------------------------------------- #
# not_equal
# --------------------------------------------------------------------------- #


def not_equal[
    dtype: DType, origin_a: Origin, origin_b: Origin, //
](
    a: MatrixView[Scalar[dtype], origin_a],
    b: MatrixView[Scalar[dtype], origin_b],
) raises -> Matrix[Scalar[DType.bool]]:
    """Element-wise inequality comparison, `a != b`."""
    return _compare_view[func=Scalar[dtype].__ne__](a, b)


def scalar_not_equal[
    dtype: DType, origin: Origin, //
](mat: MatrixView[Scalar[dtype], origin], scalar: Scalar[dtype]) -> Matrix[
    Scalar[DType.bool]
]:
    """Element-wise inequality comparison of a matrix view against a scalar."""
    return _scalar_compare_view[func=Scalar[dtype].__ne__](mat, scalar)


# ===---------------------------------------------------------------------- ===#
# Approximate comparison
# ===---------------------------------------------------------------------- ===#
# `isclose` answers the question `equal` cannot: whether two floating-point
# results agree to within a tolerance. `allclose` is that mask reduced with
# `all`, but it is written as its own walk so that it can stop at the first
# element that fails rather than allocate one bool per element first.


def _isclose_element[
    dtype: DType
](
    a: Scalar[dtype],
    b: Scalar[dtype],
    rtol: Scalar[dtype],
    atol: Scalar[dtype],
    equal_nan: Bool,
) -> Scalar[DType.bool]:
    """Tests one pair of elements for closeness, following NumPy.

    The tolerance test is `|a - b| <= atol + rtol * |b|`, asymmetric in the
    operands exactly as NumPy's is: `b` is read as the reference value.

    Non-finite operands never reach that test. Two equal infinities are close
    and opposite ones are not, which is what `a == b` already says, whereas
    `inf - inf` is a NaN that compares false against every tolerance. A NaN is
    close to nothing at all, itself included, unless `equal_nan` is set.
    """
    var a_nan = Bool(isnan(a))
    var b_nan = Bool(isnan(b))
    if a_nan or b_nan:
        return Scalar[DType.bool](equal_nan and a_nan and b_nan)
    if not Bool(isfinite(a)) or not Bool(isfinite(b)):
        return a == b
    return abs(a - b) <= atol + rtol * abs(b)


def _isclose_view[
    dtype: DType, origin_a: Origin, origin_b: Origin, //
](
    a: MatrixView[Scalar[dtype], origin_a],
    b: MatrixView[Scalar[dtype], origin_b],
    rtol: Scalar[dtype],
    atol: Scalar[dtype],
    equal_nan: Bool,
) raises -> Matrix[Scalar[DType.bool]]:
    """Core element-wise closeness test of two MatrixView operands.

    Laid out like `_compare_view`: a SIMD-vectorised fast path when both
    operands are C-contiguous, a stride-aware double loop otherwise. The
    tolerances are runtime arguments rather than a compile-time kernel, which
    is the one reason this cannot go through `_compare_view` itself.

    The result is always a freshly allocated, C-contiguous `Matrix[bool]`.
    """
    if a.nrows() != b.nrows() or a.ncols() != b.ncols():
        raise ValueError(
            function="_isclose_view()",
            message="Input matrices must have the same shape.",
        )
    var M = a.nrows()
    var N = a.ncols()
    var total = M * N
    var result = Matrix[Scalar[DType.bool]](M, N, N, 1)

    if a.is_c_contiguous() and b.is_c_contiguous():
        comptime simd_w = simd_width_of[dtype]()
        var a_ptr = a._data.unsafe_ptr()
        var b_ptr = b._data.unsafe_ptr()
        var a_off = a.offset()
        var b_off = b.offset()

        def vec_close[
            w: Int
        ](idx: Int) {
            mut result,
            imm a_ptr,
            imm b_ptr,
            imm a_off,
            imm b_off,
            imm rtol,
            imm atol,
            imm equal_nan,
        }:
            var a_chunk = a_ptr.unsafe_load[width=w](a_off + idx)
            var b_chunk = b_ptr.unsafe_load[width=w](b_off + idx)
            var res = SIMD[DType.bool, w](fill=False)

            comptime for lane in range(w):
                res[lane] = _isclose_element(
                    a_chunk[lane], b_chunk[lane], rtol, atol, equal_nan
                )
            result._data._data.unsafe_store[width=w](idx, res)

        vectorize[simd_w](total, vec_close)
    else:
        for i in range(M):
            for j in range(N):
                result._data[i * N + j] = _isclose_element(
                    a[i, j], b[i, j], rtol, atol, equal_nan
                )

    return result^


def _scalar_isclose_view[
    dtype: DType, origin: Origin, //
](
    mat: MatrixView[Scalar[dtype], origin],
    scalar: Scalar[dtype],
    rtol: Scalar[dtype],
    atol: Scalar[dtype],
    equal_nan: Bool,
) -> Matrix[Scalar[DType.bool]]:
    """Core element-wise closeness test of a MatrixView against a scalar.

    The scalar is the reference operand, so the test is
    `|m[i, j] - scalar| <= atol + rtol * |scalar|`.

    The result is always a freshly allocated, C-contiguous `Matrix[bool]`.
    """
    var M = mat.nrows()
    var N = mat.ncols()
    var total = M * N
    var result = Matrix[Scalar[DType.bool]](M, N, N, 1)

    if mat.is_c_contiguous():
        comptime simd_w = simd_width_of[dtype]()
        var m_ptr = mat._data.unsafe_ptr()
        var m_off = mat.offset()

        def vec_close[
            w: Int
        ](idx: Int) {
            mut result,
            imm m_ptr,
            imm m_off,
            imm scalar,
            imm rtol,
            imm atol,
            imm equal_nan,
        }:
            var m_chunk = m_ptr.unsafe_load[width=w](m_off + idx)
            var res = SIMD[DType.bool, w](fill=False)

            comptime for lane in range(w):
                res[lane] = _isclose_element(
                    m_chunk[lane], scalar, rtol, atol, equal_nan
                )
            result._data._data.unsafe_store[width=w](idx, res)

        vectorize[simd_w](total, vec_close)
    else:
        for i in range(M):
            for j in range(N):
                result._data[i * N + j] = _isclose_element(
                    mat[i, j], scalar, rtol, atol, equal_nan
                )

    return result^


def isclose[
    dtype: DType, origin_a: Origin, origin_b: Origin, //
](
    a: MatrixView[Scalar[dtype], origin_a],
    b: MatrixView[Scalar[dtype], origin_b],
    rtol: Scalar[dtype] = 1e-5,
    atol: Scalar[dtype] = 1e-8,
    equal_nan: Bool = False,
) raises -> Matrix[Scalar[DType.bool]]:
    """Returns an element-wise mask of `|a - b| <= atol + rtol * |b|`.

    Parameters:
        dtype: The data type of the matrix elements. Must be floating-point.
        origin_a: The origin of the first operand.
        origin_b: The origin of the second operand.

    Args:
        a: The first matrix or view.
        b: The second matrix or view, read as the reference operand.
        rtol: The relative tolerance, as a fraction of `|b|`.
        atol: The absolute tolerance, which is what decides the comparison
            when `b` is at or near zero.
        equal_nan: Whether a NaN counts as close to a NaN.

    Returns:
        A boolean matrix of the same shape.

    Raises:
        ValueError: If the two operands have different shapes.
    """
    comptime assert dtype.is_floating_point(), (
        "isclose and allclose require a floating-point dtype: integers are"
        " exact, so `equal` is the comparison for them"
    )
    return _isclose_view(a, b, rtol, atol, equal_nan)


def scalar_isclose[
    dtype: DType, origin: Origin, //
](
    mat: MatrixView[Scalar[dtype], origin],
    scalar: Scalar[dtype],
    rtol: Scalar[dtype] = 1e-5,
    atol: Scalar[dtype] = 1e-8,
    equal_nan: Bool = False,
) -> Matrix[Scalar[DType.bool]]:
    """Returns an element-wise mask of closeness to a single value.

    Parameters:
        dtype: The data type of the matrix elements. Must be floating-point.
        origin: The origin of the operand.

    Args:
        mat: The matrix or view to test.
        scalar: The reference value.
        rtol: The relative tolerance, as a fraction of `|scalar|`.
        atol: The absolute tolerance, which is what decides the comparison
            when `scalar` is zero.
        equal_nan: Whether a NaN element counts as close to a NaN reference.

    Returns:
        A boolean matrix of the same shape as `mat`.
    """
    comptime assert dtype.is_floating_point(), (
        "isclose and allclose require a floating-point dtype: integers are"
        " exact, so `equal` is the comparison for them"
    )
    return _scalar_isclose_view(mat, scalar, rtol, atol, equal_nan)


def allclose[
    dtype: DType, origin_a: Origin, origin_b: Origin, //
](
    a: MatrixView[Scalar[dtype], origin_a],
    b: MatrixView[Scalar[dtype], origin_b],
    rtol: Scalar[dtype] = 1e-5,
    atol: Scalar[dtype] = 1e-8,
    equal_nan: Bool = False,
) raises -> Bool:
    """Returns True if every pair of elements is close.

    True for an empty operand, as in NumPy: there is no element that fails.

    Parameters:
        dtype: The data type of the matrix elements. Must be floating-point.
        origin_a: The origin of the first operand.
        origin_b: The origin of the second operand.

    Args:
        a: The first matrix or view.
        b: The second matrix or view, read as the reference operand.
        rtol: The relative tolerance, as a fraction of `|b|`.
        atol: The absolute tolerance.
        equal_nan: Whether a NaN counts as close to a NaN.

    Returns:
        True if no pair of elements falls outside the tolerance.

    Raises:
        ValueError: If the two operands have different shapes.
    """
    comptime assert dtype.is_floating_point(), (
        "isclose and allclose require a floating-point dtype: integers are"
        " exact, so `equal` is the comparison for them"
    )
    if a.nrows() != b.nrows() or a.ncols() != b.ncols():
        raise ValueError(
            function="allclose()",
            message="Input matrices must have the same shape.",
        )
    for i in range(a.nrows()):
        for j in range(a.ncols()):
            if not _isclose_element(a[i, j], b[i, j], rtol, atol, equal_nan):
                return False
    return True


def scalar_allclose[
    dtype: DType, origin: Origin, //
](
    mat: MatrixView[Scalar[dtype], origin],
    scalar: Scalar[dtype],
    rtol: Scalar[dtype] = 1e-5,
    atol: Scalar[dtype] = 1e-8,
    equal_nan: Bool = False,
) -> Bool:
    """Returns True if every element is close to a single value.

    True for an empty operand, as in NumPy.

    Parameters:
        dtype: The data type of the matrix elements. Must be floating-point.
        origin: The origin of the operand.

    Args:
        mat: The matrix or view to test.
        scalar: The reference value.
        rtol: The relative tolerance, as a fraction of `|scalar|`.
        atol: The absolute tolerance.
        equal_nan: Whether a NaN element counts as close to a NaN reference.

    Returns:
        True if no element falls outside the tolerance.
    """
    comptime assert dtype.is_floating_point(), (
        "isclose and allclose require a floating-point dtype: integers are"
        " exact, so `equal` is the comparison for them"
    )
    for i in range(mat.nrows()):
        for j in range(mat.ncols()):
            if not _isclose_element(mat[i, j], scalar, rtol, atol, equal_nan):
                return False
    return True


# ===---------------------------------------------------------------------- ===#
# Logical connectives
# ===---------------------------------------------------------------------- ===#
# These are the element-wise connectives, not the bitwise ones: an operand of
# any dtype is read for truthiness first, so `logical_and` on two float
# matrices asks whether both entries are non-zero, and on two masks it is the
# ordinary conjunction. That is NumPy's `logical_*` rather than its `&`.
#
# Each is a kernel handed to the comparison cores above, so the strided and
# vectorised paths are the ones the comparisons already use.


def _logical_and_element[
    dtype: DType
](a: Scalar[dtype], b: Scalar[dtype]) -> Scalar[DType.bool]:
    """Conjunction of the truthiness of two elements."""
    var zero = Scalar[dtype](0)
    return (a != zero) & (b != zero)


def _logical_or_element[
    dtype: DType
](a: Scalar[dtype], b: Scalar[dtype]) -> Scalar[DType.bool]:
    """Disjunction of the truthiness of two elements."""
    var zero = Scalar[dtype](0)
    return (a != zero) | (b != zero)


def _logical_xor_element[
    dtype: DType
](a: Scalar[dtype], b: Scalar[dtype]) -> Scalar[DType.bool]:
    """Exclusive disjunction of the truthiness of two elements."""
    var zero = Scalar[dtype](0)
    return (a != zero) ^ (b != zero)


def logical_and[
    dtype: DType, origin_a: Origin, origin_b: Origin, //
](
    a: MatrixView[Scalar[dtype], origin_a],
    b: MatrixView[Scalar[dtype], origin_b],
) raises -> Matrix[Scalar[DType.bool]]:
    """Element-wise conjunction: True where both operands are non-zero."""
    return _compare_view[func=_logical_and_element[dtype]](a, b)


def scalar_logical_and[
    dtype: DType, origin: Origin, //
](mat: MatrixView[Scalar[dtype], origin], scalar: Scalar[dtype]) -> Matrix[
    Scalar[DType.bool]
]:
    """Element-wise conjunction of a matrix view with a single value."""
    return _scalar_compare_view[func=_logical_and_element[dtype]](mat, scalar)


def logical_or[
    dtype: DType, origin_a: Origin, origin_b: Origin, //
](
    a: MatrixView[Scalar[dtype], origin_a],
    b: MatrixView[Scalar[dtype], origin_b],
) raises -> Matrix[Scalar[DType.bool]]:
    """Element-wise disjunction: True where either operand is non-zero."""
    return _compare_view[func=_logical_or_element[dtype]](a, b)


def scalar_logical_or[
    dtype: DType, origin: Origin, //
](mat: MatrixView[Scalar[dtype], origin], scalar: Scalar[dtype]) -> Matrix[
    Scalar[DType.bool]
]:
    """Element-wise disjunction of a matrix view with a single value."""
    return _scalar_compare_view[func=_logical_or_element[dtype]](mat, scalar)


def logical_xor[
    dtype: DType, origin_a: Origin, origin_b: Origin, //
](
    a: MatrixView[Scalar[dtype], origin_a],
    b: MatrixView[Scalar[dtype], origin_b],
) raises -> Matrix[Scalar[DType.bool]]:
    """Element-wise exclusive disjunction: True where exactly one is non-zero.
    """
    return _compare_view[func=_logical_xor_element[dtype]](a, b)


def scalar_logical_xor[
    dtype: DType, origin: Origin, //
](mat: MatrixView[Scalar[dtype], origin], scalar: Scalar[dtype]) -> Matrix[
    Scalar[DType.bool]
]:
    """Element-wise exclusive disjunction with a single value."""
    return _scalar_compare_view[func=_logical_xor_element[dtype]](mat, scalar)


def logical_not[
    dtype: DType, origin: Origin, //
](mat: MatrixView[Scalar[dtype], origin]) -> Matrix[Scalar[DType.bool]]:
    """Element-wise negation: True where the operand is zero.

    Negating truthiness is testing against zero, so this is the scalar
    equality kernel with zero on the right; it needs no kernel of its own.

    Parameters:
        dtype: The data type of the matrix elements.
        origin: The origin of the operand.

    Args:
        mat: The matrix or view to negate.

    Returns:
        A boolean matrix of the same shape as `mat`.
    """
    return _scalar_compare_view[func=Scalar[dtype].__eq__](
        mat, Scalar[dtype](0)
    )


# ===---------------------------------------------------------------------- ===#
# Boolean reductions
# ===---------------------------------------------------------------------- ===#
# `all` and `any` accept any dtype, not just `DType.bool`, and test each
# element against zero the way NumPy and Python do. That matters because the
# comparisons above produce `Matrix[Scalar[DType.bool]]` but the operand is just as
# often a numeric matrix straight from arithmetic.
#
# They are written as direct walks rather than through `fold`, because a fold
# threads a `Scalar[dtype]` accumulator and the accumulator here is a `Bool` -
# a different type from the elements. Both short-circuit, which a fold could
# not do either.


def all[
    dtype: DType, origin: Origin[mut=False], //
](m: MatrixView[Scalar[dtype], origin]) -> Bool:
    """Returns True if every element is non-zero.

    True for an empty operand, as in NumPy and Python: there is no element
    that fails the test.

    Parameters:
        dtype: The data type of the matrix elements.
        origin: The origin of the operand.

    Args:
        m: The matrix or view to test.

    Returns:
        True if no element equals zero.
    """
    for i in range(m.nrows()):
        for j in range(m.ncols()):
            if not m[i, j]:
                return False
    return True


def all[
    dtype: DType, origin: Origin[mut=False], //
](m: MatrixView[Scalar[dtype], origin], axis: Int) raises -> Matrix[
    Scalar[DType.bool]
]:
    """Returns, per lane, whether every element is non-zero.

    Parameters:
        dtype: The data type of the matrix elements.
        origin: The origin of the operand.

    Args:
        m: The matrix or view to test.
        axis: The dimension to remove. 0 returns `1 x ncols`, 1 returns
            `nrows x 1`.

    Returns:
        A boolean matrix with one entry per lane.

    Raises:
        ValueError: If `axis` is neither 0 nor 1.
    """
    if axis == 0:
        var result = Matrix[Scalar[DType.bool]](1, m.ncols(), m.ncols(), 1)
        for j in range(m.ncols()):
            result._data[j] = all(m[:, j : j + 1])
        return result^
    elif axis == 1:
        var result = Matrix[Scalar[DType.bool]](m.nrows(), 1, 1, 1)
        for i in range(m.nrows()):
            result._data[i] = all(m[i : i + 1, :])
        return result^
    raise ValueError(function="all(m, axis)", message="Axis must be 0 or 1.")


def any[
    dtype: DType, origin: Origin[mut=False], //
](m: MatrixView[Scalar[dtype], origin]) -> Bool:
    """Returns True if at least one element is non-zero.

    False for an empty operand, as in NumPy and Python.

    Parameters:
        dtype: The data type of the matrix elements.
        origin: The origin of the operand.

    Args:
        m: The matrix or view to test.

    Returns:
        True if any element differs from zero.
    """
    for i in range(m.nrows()):
        for j in range(m.ncols()):
            if m[i, j]:
                return True
    return False


def any[
    dtype: DType, origin: Origin[mut=False], //
](m: MatrixView[Scalar[dtype], origin], axis: Int) raises -> Matrix[
    Scalar[DType.bool]
]:
    """Returns, per lane, whether at least one element is non-zero.

    Parameters:
        dtype: The data type of the matrix elements.
        origin: The origin of the operand.

    Args:
        m: The matrix or view to test.
        axis: The dimension to remove. 0 returns `1 x ncols`, 1 returns
            `nrows x 1`.

    Returns:
        A boolean matrix with one entry per lane.

    Raises:
        ValueError: If `axis` is neither 0 nor 1.
    """
    if axis == 0:
        var result = Matrix[Scalar[DType.bool]](1, m.ncols(), m.ncols(), 1)
        for j in range(m.ncols()):
            result._data[j] = any(m[:, j : j + 1])
        return result^
    elif axis == 1:
        var result = Matrix[Scalar[DType.bool]](m.nrows(), 1, 1, 1)
        for i in range(m.nrows()):
            result._data[i] = any(m[i : i + 1, :])
        return result^
    raise ValueError(function="any(m, axis)", message="Axis must be 0 or 1.")
