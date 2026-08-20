"""
Shape and layout manipulation routines for matrices.

Everything here answers the same question --- *the same elements, arranged
differently* --- and the answers split cleanly in two.

`reshape`, `resize`, `flatten`, `contiguous`, `reorder_layout` and `astype`
return a **new owning `Matrix`**. They allocate, and they never touch the
buffer of the input.

`reshape_view` and `broadcast_to` return a **`MatrixView` over the input's own
buffer**. They allocate nothing, and the origin they carry is the input's, so
the borrow checker keeps the source alive for exactly as long as the result is.

> **Invariant: a matrix's element buffer is fixed at construction.** Nothing in
> this module grows, shrinks or reallocates the `data` of an existing matrix.
> This is a safety rule, not a style preference. A `MatrixView` holds a `Span`
> over `origin_of(m._data)`, which captures the `List`'s heap pointer; growing
> that `List` reallocates and leaves every live view dangling, and Mojo 1.0
> does not catch it --- the borrow checker enforces origins at *call sites*,
> and a later `m._data.append(...)` in the same scope is not a call site it
> inspects. `resize` therefore returns a new matrix where NuMojo's mutated one
> in place.

Two conventions carried over from the rest of the routine layer. Operands are
`MatrixView`, and a `Matrix` converts implicitly (see the `@implicit`
constructor in `types/matrix_view.mojo`), so one signature per routine covers
both types. And an owning result is always C-contiguous; `contiguous(m, "F")`
is how a caller asks for the other layout.
"""

from linamo.types.errors import ValueError
from linamo.types.matrix import Matrix
from linamo.types.matrix_view import MatrixView


# ===---------------------------------------------------------------------- ===#
# Index-order helpers
# ===---------------------------------------------------------------------- ===#
# `order` is an *index* order, not a memory layout, exactly as in NumPy:
# "C" walks the last axis fastest (row by row), "F" walks the first axis
# fastest (column by column). Where the elements end up in memory is a
# separate question, answered by `contiguous`.


@always_inline
def _check_order(order: String, function: String) raises:
    """Rejects an `order` that is neither "C" nor "F"."""
    if order != "C" and order != "F":
        raise ValueError(
            function=function,
            message=String(
                "Unknown index order '",
                order,
                "'. Use 'C' (row-major) or 'F' (column-major).",
            ),
        )


@always_inline
def _unravel(k: Int, nrows: Int, ncols: Int, c_order: Bool) -> Tuple[Int, Int]:
    """Turns the flat position `k` into `(row, col)` under the given order."""
    if c_order:
        return (k // ncols, k % ncols)
    return (k % nrows, k // nrows)


@always_inline
def _ravel(row: Int, col: Int, nrows: Int, ncols: Int, c_order: Bool) -> Int:
    """Turns `(row, col)` back into a flat position under the given order."""
    if c_order:
        return row * ncols + col
    return col * nrows + row


# [Mojo Miji]
# The routines below fill their result buffer front to back with `append`
# rather than allocating `unsafe_uninit_length` and writing into it at a
# computed offset. For a scalar element the two are the same thing; for an
# element that owns a heap allocation they are not, because assigning into an
# uninitialised slot runs a destructor over whatever was there. Filling in
# buffer order costs an index computation per element and is correct for every
# element type.


# ===---------------------------------------------------------------------- ===#
# Reshaping
# ===---------------------------------------------------------------------- ===#


def reshape[
    T: Copyable & Deinitable, origin: Origin
](
    a: MatrixView[T, origin],
    nrows: Int,
    ncols: Int,
    order: String = "C",
) raises -> Matrix[T]:
    """Returns a new matrix with the given shape holding the same elements.

    The elements of `a` are read in `order` and written into the result in the
    same `order`, which is what NumPy's `reshape` does. The result itself is
    always C-contiguous in memory; pass it through `contiguous(..., "F")` to
    change that.

    Works for any input layout, including a strided slice, because it goes
    through `a[i, j]` rather than the buffer. For the zero-copy version, which
    is restricted to dense inputs, see `reshape_view`.

    Parameters:
        T: The type of the matrix elements.
        origin: The origin of the input view.

    Args:
        a: The matrix or view to reshape.
        nrows: The number of rows of the result.
        ncols: The number of columns of the result.
        order: The index order in which elements are read and written.

    Returns:
        A new `nrows x ncols` matrix holding the elements of `a`.

    Raises:
        ValueError: If `nrows * ncols` differs from the size of `a`, or if
            `order` is neither "C" nor "F".
    """
    comptime fn_name = "reshape(a, nrows, ncols, order)"
    _check_order(order, fn_name)
    var size = a.nrows() * a.ncols()
    if nrows < 0 or ncols < 0 or nrows * ncols != size:
        raise ValueError(
            function=fn_name,
            message=String(
                "Cannot reshape a matrix of size ",
                size,
                " into shape (",
                nrows,
                ", ",
                ncols,
                "). The element count must be preserved.",
            ),
        )

    var c_order = order == "C"
    var data = List[T](capacity=size)
    for p in range(size):
        # `p` walks the result buffer, which is C-contiguous, so it names the
        # destination cell directly; that cell's position in `order` is what
        # picks the source element.
        var k = _ravel(p // ncols, p % ncols, nrows, ncols, c_order)
        var src_row, src_col = _unravel(k, a.nrows(), a.ncols(), c_order)
        data.append(a[src_row, src_col].copy())
    return Matrix[T](
        buffer=data^, nrows=nrows, ncols=ncols, row_stride=ncols, col_stride=1
    )


def reshape_view[
    mut: Bool, //, T: Copyable & Deinitable, origin: Origin[mut=mut]
](a: MatrixView[T, origin], nrows: Int, ncols: Int) raises -> MatrixView[
    T, ImmOrigin(origin)
]:
    """Reinterprets a dense matrix under a new shape, without copying.

    This is the metadata-only reshape: the result views the *same buffer* as
    `a` and carries the same origin, so `a` (or the matrix it came from) is
    kept alive for as long as the result is. Nothing is allocated.

    It requires a dense input, because only then does "the elements in memory
    order" mean the same thing before and after. A C-contiguous input yields a
    C-contiguous view, an F-contiguous input an F-contiguous one; the memory
    order of the elements is untouched either way. Anything strided --- a
    slice with a step, a sub-block --- has to copy, so use `reshape`.

    The result is read-only regardless of how `a` was bound. Two views of one
    buffer with different shapes are aliases, and handing out a writable alias
    is exactly what the `Matrix` / `MatrixView` split exists to prevent.

    Parameters:
        mut: Whether the input view is mutable (inferred).
        T: The type of the matrix elements.
        origin: The origin of the input view.

    Args:
        a: The matrix or view to reinterpret. Must be C- or F-contiguous.
        nrows: The number of rows of the result.
        ncols: The number of columns of the result.

    Returns:
        A read-only view of the same buffer with the requested shape.

    Raises:
        ValueError: If `nrows * ncols` differs from the size of `a`, or if `a`
            is neither C- nor F-contiguous.
    """
    comptime fn_name = "reshape_view(a, nrows, ncols)"
    var size = a.nrows() * a.ncols()
    if nrows < 0 or ncols < 0 or nrows * ncols != size:
        raise ValueError(
            function=fn_name,
            message=String(
                "Cannot reshape a matrix of size ",
                size,
                " into shape (",
                nrows,
                ", ",
                ncols,
                "). The element count must be preserved.",
            ),
        )

    var row_stride: Int
    var col_stride: Int
    if a.is_c_contiguous():
        row_stride = ncols
        col_stride = 1
    elif a.is_f_contiguous():
        row_stride = 1
        col_stride = nrows
    else:
        raise ValueError(
            function=fn_name,
            message=(
                "A strided view cannot be reshaped without copying. Use"
                " `reshape` instead."
            ),
        )

    return MatrixView[T, ImmOrigin(origin)](
        buffer=a._data.as_imm(),
        nrows=nrows,
        ncols=ncols,
        row_stride=row_stride,
        col_stride=col_stride,
        offset=a.offset(),
    )


def flatten[
    T: Copyable & Deinitable, origin: Origin
](a: MatrixView[T, origin], order: String = "C") raises -> Matrix[T]:
    """Returns the elements of `a` as a new `1 x size` matrix.

    The row is filled in `order`: "C" reads `a` row by row, "F" column by
    column.

    Parameters:
        T: The type of the matrix elements.
        origin: The origin of the input view.

    Args:
        a: The matrix or view to flatten.
        order: The index order in which the elements are read.

    Returns:
        A new `1 x (nrows * ncols)` matrix.

    Raises:
        ValueError: If `order` is neither "C" nor "F".
    """
    _check_order(order, "flatten(a, order)")
    return reshape(a, 1, a.nrows() * a.ncols(), order)


def resize[
    dtype: DType, origin: Origin
](
    a: MatrixView[Scalar[dtype], origin], nrows: Int, ncols: Int
) raises -> Matrix[Scalar[dtype]]:
    """Returns a new matrix of the given shape, truncating or zero-padding.

    Elements are taken from `a` in C order and laid down in C order. If the new
    shape holds fewer elements the tail is dropped; if it holds more, the extra
    elements are zero. This is what `numpy.ndarray.resize` does, minus the
    mutation.

    Unlike NuMojo's `Matrix.resize`, this cannot act in place. Growing an
    owning matrix would mean reallocating its `List`, which invalidates the
    `Span` inside every live view of it --- and Mojo 1.0 will not catch that,
    because the reallocation is not a call site the borrow checker inspects.
    The buffer of an existing matrix is fixed at construction; use the returned
    matrix instead, `a = resize(a, m, n)` if the old one is not needed.

    Parameters:
        dtype: The data type of the matrix elements.
        origin: The origin of the input view.

    Args:
        a: The matrix or view to resize.
        nrows: The number of rows of the result.
        ncols: The number of columns of the result.

    Returns:
        A new `nrows x ncols` C-contiguous matrix.

    Raises:
        ValueError: If `nrows` or `ncols` is negative.
    """
    if nrows < 0 or ncols < 0:
        raise ValueError(
            function="resize(a, nrows, ncols)",
            message=String(
                "A shape cannot be negative, got (", nrows, ", ", ncols, ")."
            ),
        )

    var size = nrows * ncols
    var kept = min(size, a.nrows() * a.ncols())
    var data = List[Scalar[dtype]](unsafe_uninit_length=size)
    for k in range(kept):
        data[k] = a[k // a.ncols(), k % a.ncols()]
    for k in range(kept, size):
        data[k] = 0
    return Matrix[Scalar[dtype]](
        buffer=data^, nrows=nrows, ncols=ncols, row_stride=ncols, col_stride=1
    )


# ===---------------------------------------------------------------------- ===#
# Layout
# ===---------------------------------------------------------------------- ===#


def contiguous[
    T: Copyable & Deinitable, origin: Origin
](a: MatrixView[T, origin], order: String = "C") raises -> Matrix[T]:
    """Returns a dense copy of `a` in the requested memory layout.

    The shape and the element at every `(i, j)` are unchanged; only where the
    elements sit in memory differs. A copy is made even when `a` already has
    the requested layout, because the caller is asking for an owning matrix and
    a view cannot become one for free.

    This is how a strided view --- a sliced sub-block, a stepped slice ---
    becomes something the SIMD paths in `routines.math` can run over at full
    speed.

    Parameters:
        T: The type of the matrix elements.
        origin: The origin of the input view.

    Args:
        a: The matrix or view to copy.
        order: "C" for row-major, "F" for column-major.

    Returns:
        A new matrix with the same elements, dense in the requested layout.

    Raises:
        ValueError: If `order` is neither "C" nor "F".
    """
    _check_order(order, "contiguous(a, order)")
    var nrows = a.nrows()
    var ncols = a.ncols()
    var c_order = order == "C"
    var row_stride = ncols if c_order else 1
    var col_stride = 1 if c_order else nrows
    var data = List[T](capacity=nrows * ncols)
    for p in range(nrows * ncols):
        var i, j = _unravel(p, nrows, ncols, c_order)
        data.append(a[i, j].copy())
    return Matrix[T](
        buffer=data^,
        nrows=nrows,
        ncols=ncols,
        row_stride=row_stride,
        col_stride=col_stride,
    )


def reorder_layout[
    T: Copyable & Deinitable, origin: Origin
](a: MatrixView[T, origin]) raises -> Matrix[T]:
    """Returns a copy of `a` in the opposite memory layout.

    C-contiguous in, F-contiguous out, and the other way round. The shape and
    every element stay where they are logically; this only moves them in
    memory.

    Parameters:
        T: The type of the matrix elements.
        origin: The origin of the input view.

    Args:
        a: The matrix or view to copy. Must be C- or F-contiguous.

    Returns:
        A new matrix with the same elements in the opposite layout.

    Raises:
        ValueError: If `a` is neither C- nor F-contiguous, in which case there
            is no layout to flip. Use `contiguous(a, order)` and name the one
            you want.
    """
    if a.is_c_contiguous():
        return contiguous(a, "F")
    if a.is_f_contiguous():
        return contiguous(a, "C")
    raise ValueError(
        function="reorder_layout(a)",
        message=(
            "A view that is neither C- nor F-contiguous has no layout to"
            " flip. Use `contiguous(a, order)` instead."
        ),
    )


def broadcast_to[
    mut: Bool, //, T: Copyable & Deinitable, origin: Origin[mut=mut]
](a: MatrixView[T, origin], nrows: Int, ncols: Int) raises -> MatrixView[
    T, ImmOrigin(origin)
]:
    """Stretches size-1 dimensions of `a` to the given shape, without copying.

    A dimension of `a` either already matches the target or has extent 1, in
    which case it is stretched by giving it a stride of zero --- every index
    along it lands on the same element. So a `1 x n` row broadcast to
    `m x n` costs nothing and reads as `m` copies of that row, and the result
    shares its buffer and origin with `a`.

    The result is read-only, as it is in NumPy, and for a stronger reason here:
    several logical positions map onto one element, so a write would be visible
    at every one of them.

    A zero stride is not contiguous by any definition, so the result takes the
    strided path through the routine layer. Call `.to_matrix()` on it if a
    dense operand is wanted.

    Parameters:
        mut: Whether the input view is mutable (inferred).
        T: The type of the matrix elements.
        origin: The origin of the input view.

    Args:
        a: The matrix or view to broadcast.
        nrows: The number of rows of the result.
        ncols: The number of columns of the result.

    Returns:
        A read-only view of shape `nrows x ncols` over the buffer of `a`.

    Raises:
        ValueError: If a dimension of `a` neither matches the target nor is 1.
    """
    comptime fn_name = "broadcast_to(a, nrows, ncols)"

    var row_stride: Int
    if a.nrows() == nrows:
        row_stride = a.row_stride()
    elif a.nrows() == 1:
        row_stride = 0
    else:
        raise ValueError(
            function=fn_name,
            message=String(
                "Cannot broadcast ",
                a.nrows(),
                " rows to ",
                nrows,
                ". A dimension must match the target or be 1.",
            ),
        )

    var col_stride: Int
    if a.ncols() == ncols:
        col_stride = a.col_stride()
    elif a.ncols() == 1:
        col_stride = 0
    else:
        raise ValueError(
            function=fn_name,
            message=String(
                "Cannot broadcast ",
                a.ncols(),
                " columns to ",
                ncols,
                ". A dimension must match the target or be 1.",
            ),
        )

    return MatrixView[T, ImmOrigin(origin)](
        buffer=a._data.as_imm(),
        nrows=nrows,
        ncols=ncols,
        row_stride=row_stride,
        col_stride=col_stride,
        offset=a.offset(),
    )


# ===---------------------------------------------------------------------- ===#
# Type conversion
# ===---------------------------------------------------------------------- ===#


def astype[
    dtype: DType,
    origin: Origin,
    target: DType,
    //,
    Target: Copyable & Deinitable,
](a: MatrixView[Scalar[dtype], origin]) raises -> Matrix[Scalar[target]] where (
    Target == Scalar[target]
):
    """Returns a C-contiguous copy of `a` with its elements cast to `target`.

    The cast is Mojo's `SIMD.cast`, so the usual rules apply: a float to an
    integer type truncates towards zero, and a narrowing conversion wraps.

    Parameters:
        dtype: The data type of the input elements.
        origin: The origin of the input view.
        target: The dtype behind `Target`, deduced rather than written.
        Target: The type of the result elements.

    Args:
        a: The matrix or view to cast.

    Returns:
        A new `Matrix[Scalar[target]]` with the same shape.
    """
    var nrows = a.nrows()
    var ncols = a.ncols()
    var data = List[Scalar[target]](unsafe_uninit_length=nrows * ncols)
    for i in range(nrows):
        for j in range(ncols):
            data[i * ncols + j] = a[i, j].cast[target]()
    return Matrix[Scalar[target]](
        buffer=data^, nrows=nrows, ncols=ncols, row_stride=ncols, col_stride=1
    )
