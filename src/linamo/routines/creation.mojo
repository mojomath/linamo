"""
This module defines routines for creating matrices and matrix views in Linamo.
"""

from std.math import ceil

from decimo import Numeric, Parsable

from linamo.types.errors import ValueError
from linamo.types.matrix import Matrix
from linamo.types.matrix_view import MatrixView
from linamo.types.static_matrix import StaticMatrix
from linamo.utils.indexing import get_offset

# ===---------------------------------------------------------------------- ===#
# Matrix creation routines
# - Create matrices from nested lists.
# - Create matrices from flat lists with specified shapes.
# - Create static matrices from nested lists with compile-time dimensions.
# - Create static matrices from flat lists with compile-time dimensions.
# ===---------------------------------------------------------------------- ===#


def matrix[
    T: Copyable & Deinitable = Float64
](list: List[List[T]], order: String = "C") raises -> Matrix[T]:
    """Initializes the matrix with a list of lists.

    Parameters:
        T: The type of the matrix elements. Defaults to `Float64`.

    Args:
        list: A list of lists where each inner list represents a row of the matrix.
        order: A string indicating the memory layout order. "C" for row-major
            and "F" for column-major. Defaults to "C".

    Raises:
        ValueError: If the input list is empty, if the rows have different
        lengths, or if the order is invalid.
    """

    if len(list) == 0:
        raise ValueError(
            function="matrix()",
            message="Data cannot be an empty list.",
        )

    var nrows = len(list)
    var ncols = len(list[0])

    var row_stride: Int
    var col_stride: Int
    if order == "C":
        row_stride = ncols
        col_stride = 1  # Row-major order
    elif order == "F":
        col_stride = nrows
        row_stride = 1  # Column-major order
    else:
        raise ValueError(
            function="matrix()",
            message="Invalid order. Must be 'C' or 'F'.",
        )

    for row in list:
        if len(row) != ncols:
            raise ValueError(
                function="matrix()",
                message="All rows must have the same length.",
            )

    # The buffer is filled front to back rather than allocated
    # `unsafe_uninit_length` and written into at a computed offset. For a
    # scalar element the two are the same; for an element that owns a heap
    # allocation they are not, because assigning into an uninitialised slot
    # runs a destructor over whatever happened to be there.
    var c_order = row_stride == ncols
    var data = List[T](capacity=nrows * ncols)
    for p in range(nrows * ncols):
        var row = p // ncols if c_order else p % nrows
        var col = p % ncols if c_order else p // nrows
        data.append(list[row][col].copy())
    return Matrix[T](
        buffer=data^,
        nrows=nrows,
        ncols=ncols,
        row_stride=row_stride,
        col_stride=col_stride,
    )


def matrix[
    T: Copyable & Deinitable = Float64
](
    *,
    var flat_list: List[T],
    nrows: Int,
    ncols: Int,
    order: String = "C",
) raises -> Matrix[T]:
    """Initializes the matrix with a list and shape.

    Parameters:
        T: The type of the matrix elements. Defaults to `Float64`.

    Args:
        flat_list: A list of elements to initialize the matrix.
        nrows: The number of rows in the matrix.
        ncols: The number of columns in the matrix.
        order: The memory layout order, either "C" for row-major or
            "F" for column-major. Defaults to "C".

    Raises:
        ValueError: If the length of the flat_list does not match the
        product of the shape dimensions.
    """
    if len(flat_list) == 0:
        raise ValueError(
            function="matrix()",
            message="Data cannot be an empty list.",
        )
    if len(flat_list) != nrows * ncols:
        raise ValueError(
            function="matrix()",
            message="Data length does not match the specified shape.",
        )
    var row_stride: Int
    var col_stride: Int
    if order == "C":
        row_stride = ncols  # Row-major order
        col_stride = 1
    elif order == "F":
        row_stride = 1  # Column-major order
        col_stride = nrows
    else:
        raise ValueError(
            function="matrix()",
            message="Invalid order. Must be 'C' or 'F'.",
        )
    return Matrix[T](
        buffer=flat_list^,
        nrows=nrows,
        ncols=ncols,
        row_stride=row_stride,
        col_stride=col_stride,
    )


def smatrix[
    dtype: DType = DType.float64,
    //,
    nrows: Int,
    ncols: Int,
    T: Copyable & Deinitable = Scalar[dtype],
](var list: List[List[T]]) raises -> StaticMatrix[
    Scalar[dtype], nrows, ncols
] where (T == Scalar[dtype]):
    """Initializes the static matrix with a list of lists.

    Parameters:
        dtype: The dtype behind `T`, deduced rather than written.
        nrows: The number of rows in the matrix.
        ncols: The number of columns in the matrix.
        T: The type of the matrix elements. Defaults to `Float64`.

    Args:
        list: A list of lists representing the rows of the matrix.
    """

    if len(list) != nrows:
        raise ValueError(
            function="matrix()",
            message="Number of rows in list does not match nrows.",
        )
    var result = StaticMatrix[
        Scalar[dtype], nrows, ncols
    ]()  # Initialize with zeros
    for row_index in range(len(list)):
        if len(list[row_index]) != ncols:
            raise ValueError(
                function="matrix()",
                message="All rows must have the same length as ncols.",
            )
        for col_index in range(ncols):
            result._set_flat(
                row_index * result.row_stride()
                + col_index * result.col_stride(),
                rebind[Scalar[dtype]](list[row_index][col_index]),
            )
    return result^


def smatrix[
    dtype: DType = DType.float64,
    //,
    nrows: Int,
    ncols: Int,
    T: Copyable & Deinitable = Scalar[dtype],
](*, var flat_list: List[T]) raises -> StaticMatrix[
    Scalar[dtype], nrows, ncols
] where (T == Scalar[dtype]):
    """Initializes the static matrix with a list of values.

    Parameters:
        dtype: The dtype behind `T`, deduced rather than written.
        nrows: The number of rows in the matrix.
        ncols: The number of columns in the matrix.
        T: The type of the matrix elements. Defaults to `Float64`.

    Args:
        flat_list: A list of values.
    """
    if len(flat_list) != nrows * ncols:
        raise ValueError(
            function="matrix()",
            message="Number of rows in list does not match nrows.",
        )
    var result = StaticMatrix[
        Scalar[dtype], nrows, ncols
    ]()  # Initialize with zeros
    var offset = 0
    for i in range(nrows):
        for j in range(ncols):
            result._set_flat(
                i * result.row_stride() + j * result.col_stride(),
                rebind[Scalar[dtype]](flat_list[offset]),
            )
            offset += 1
    return result^


# ===---------------------------------------------------------------------- ===#
# Convenience constructors
# - zeros(), ones(), full()
# - eye(), identity()
# - diag()
# ===---------------------------------------------------------------------- ===#
# Every routine here that fills a matrix with 0 or 1 comes in two overloads.
# The scalar one writes the literals straight into the buffer, which only a
# `Scalar[dtype]` accepts. The other asks `Numeric` for `zero()` and `one()`,
# and so serves `BigInt`, `BigDecimal` and `Decimal128`. The two are disjoint
# --- no type satisfies both `where` clauses --- so `zeros[Float64](3, 3)` and
# `zeros[Dec128](3, 3)` are the same call spelled with different elements.


def zeros[
    dtype: DType = DType.float64, //, T: Copyable & Deinitable = Scalar[dtype]
](nrows: Int, ncols: Int) -> Matrix[Scalar[dtype]] where T == Scalar[dtype]:
    """Creates a matrix filled with zeros.

    Parameters:
        dtype: The dtype behind `T`, deduced rather than written.
        T: The type of the matrix elements. Defaults to `Float64`.

    Args:
        nrows: The number of rows in the matrix.
        ncols: The number of columns in the matrix.

    Returns:
        A new matrix of shape (nrows, ncols) filled with zeros.
    """
    return Matrix[Scalar[dtype]](
        buffer=List[Scalar[dtype]](length=nrows * ncols, fill=0),
        nrows=nrows,
        ncols=ncols,
        row_stride=ncols,
        col_stride=1,
    )


def zeros[T: Numeric](nrows: Int, ncols: Int) -> Matrix[T]:
    """Creates a matrix filled with zeros.

    Parameters:
        T: The type of the matrix elements.

    Args:
        nrows: The number of rows in the matrix.
        ncols: The number of columns in the matrix.

    Returns:
        A new matrix of shape (nrows, ncols) filled with zeros.
    """
    return Matrix[T](
        buffer=List[T](length=nrows * ncols, fill=T.zero()),
        nrows=nrows,
        ncols=ncols,
        row_stride=ncols,
        col_stride=1,
    )


def ones[
    dtype: DType = DType.float64, //, T: Copyable & Deinitable = Scalar[dtype]
](nrows: Int, ncols: Int) -> Matrix[Scalar[dtype]] where T == Scalar[dtype]:
    """Creates a matrix filled with ones.

    Parameters:
        dtype: The dtype behind `T`, deduced rather than written.
        T: The type of the matrix elements. Defaults to `Float64`.

    Args:
        nrows: The number of rows in the matrix.
        ncols: The number of columns in the matrix.

    Returns:
        A new matrix of shape (nrows, ncols) filled with ones.
    """
    return Matrix[Scalar[dtype]](
        buffer=List[Scalar[dtype]](length=nrows * ncols, fill=1),
        nrows=nrows,
        ncols=ncols,
        row_stride=ncols,
        col_stride=1,
    )


def ones[T: Numeric](nrows: Int, ncols: Int) -> Matrix[T]:
    """Creates a matrix filled with ones.

    Parameters:
        T: The type of the matrix elements.

    Args:
        nrows: The number of rows in the matrix.
        ncols: The number of columns in the matrix.

    Returns:
        A new matrix of shape (nrows, ncols) filled with ones.
    """
    return Matrix[T](
        buffer=List[T](length=nrows * ncols, fill=T.one()),
        nrows=nrows,
        ncols=ncols,
        row_stride=ncols,
        col_stride=1,
    )


def full[
    T: Copyable & Deinitable = Float64
](nrows: Int, ncols: Int, fill_value: T) -> Matrix[T]:
    """Creates a matrix filled with a specified value.

    Parameters:
        T: The type of the matrix elements. Defaults to `Float64`.

    Args:
        nrows: The number of rows in the matrix.
        ncols: The number of columns in the matrix.
        fill_value: The value to fill the matrix with.

    Returns:
        A new matrix of shape (nrows, ncols) filled with fill_value.
    """
    return Matrix[T](
        buffer=List[T](length=nrows * ncols, fill=fill_value),
        nrows=nrows,
        ncols=ncols,
        row_stride=ncols,
        col_stride=1,
    )


def eye[
    dtype: DType = DType.float64, //, T: Copyable & Deinitable = Scalar[dtype]
](n: Int) -> Matrix[Scalar[dtype]] where T == Scalar[dtype]:
    """Creates an n×n identity matrix.

    Parameters:
        dtype: The dtype behind `T`, deduced rather than written.
        T: The type of the matrix elements. Defaults to `Float64`.

    Args:
        n: The number of rows and columns in the identity matrix.

    Returns:
        A new n×n identity matrix with ones on the diagonal and zeros elsewhere.
    """
    var data = List[Scalar[dtype]](length=n * n, fill=0)
    for i in range(n):
        data[i * n + i] = 1
    return Matrix[Scalar[dtype]](
        buffer=data^,
        nrows=n,
        ncols=n,
        row_stride=n,
        col_stride=1,
    )


def eye[T: Numeric](n: Int) -> Matrix[T]:
    """Creates an n×n identity matrix.

    Parameters:
        T: The type of the matrix elements.

    Args:
        n: The number of rows and columns in the identity matrix.

    Returns:
        A new n×n identity matrix with ones on the diagonal and zeros elsewhere.
    """
    var data = List[T](length=n * n, fill=T.zero())
    for i in range(n):
        data[i * n + i] = T.one()
    return Matrix[T](
        buffer=data^,
        nrows=n,
        ncols=n,
        row_stride=n,
        col_stride=1,
    )


def identity[
    dtype: DType = DType.float64, //, T: Copyable & Deinitable = Scalar[dtype]
](n: Int) -> Matrix[Scalar[dtype]] where T == Scalar[dtype]:
    """Creates an n×n identity matrix. Alias for `eye()`.

    Parameters:
        dtype: The dtype behind `T`, deduced rather than written.
        T: The type of the matrix elements. Defaults to `Float64`.

    Args:
        n: The number of rows and columns in the identity matrix.

    Returns:
        A new n×n identity matrix with ones on the diagonal and zeros elsewhere.
    """
    return eye[Scalar[dtype]](n)


def identity[T: Numeric](n: Int) -> Matrix[T]:
    """Creates an n×n identity matrix. Alias for `eye()`.

    Parameters:
        T: The type of the matrix elements.

    Args:
        n: The number of rows and columns in the identity matrix.

    Returns:
        A new n×n identity matrix with ones on the diagonal and zeros elsewhere.
    """
    return eye[T](n)


def diag[
    dtype: DType = DType.float64, //, T: Copyable & Deinitable = Scalar[dtype]
](var values: List[T]) -> Matrix[T] where T == Scalar[dtype]:
    """Creates a square diagonal matrix from a list of values.

    The off-diagonal elements are zeros, so this asks its element type for a
    zero and is available for the scalar types only.

    Parameters:
        dtype: The dtype behind `T`, deduced rather than written.
        T: The type of the matrix elements. Defaults to `Float64`.

    Args:
        values: A list of diagonal values.

    Returns:
        A new n×n matrix with the given values on the diagonal and zeros
        elsewhere, where n is the length of `values`.
    """
    var n = len(values)
    var data = List[T](length=n * n, fill=rebind[T](Scalar[dtype](0)))
    for i in range(n):
        data[i * n + i] = values[i].copy()
    return Matrix[T](
        buffer=data^,
        nrows=n,
        ncols=n,
        row_stride=n,
        col_stride=1,
    )


def diag[T: Numeric](var values: List[T]) -> Matrix[T]:
    """Creates a square diagonal matrix from a list of values.

    Parameters:
        T: The type of the matrix elements.

    Args:
        values: A list of diagonal values.

    Returns:
        A new n×n matrix with the given values on the diagonal and zeros
        elsewhere, where n is the length of `values`.
    """
    var n = len(values)
    var data = List[T](length=n * n, fill=T.zero())
    for i in range(n):
        data[i * n + i] = values[i].copy()
    return Matrix[T](
        buffer=data^,
        nrows=n,
        ncols=n,
        row_stride=n,
        col_stride=1,
    )


def diag[
    dtype: DType
](mat: Matrix[Scalar[dtype]]) raises -> List[Scalar[dtype]]:
    """Extracts the diagonal elements from a matrix.

    Parameters:
        dtype: The data type of the matrix elements.

    Args:
        mat: The input matrix.

    Returns:
        A list containing the diagonal elements of the matrix.

    Raises:
        ValueError: If the matrix is not square.
    """
    if mat.nrows() != mat.ncols():
        raise ValueError(
            function="diag()",
            message="Matrix must be square to extract diagonal.",
        )
    var n = mat.nrows()
    var result = List[Scalar[dtype]](length=n, fill=0)
    for i in range(n):
        result[i] = mat._data[
            get_offset(i, i, mat.row_stride(), mat.col_stride())
        ]
    return result^


# ===---------------------------------------------------------------------- ===#
# Uninitialised storage
# ===---------------------------------------------------------------------- ===#


def empty[
    dtype: DType = DType.float64, //, T: Copyable & Deinitable = Scalar[dtype]
](nrows: Int, ncols: Int) -> Matrix[Scalar[dtype]] where T == Scalar[dtype]:
    """Creates a matrix of the given shape whose elements are **unspecified**.

    This is `zeros()` without the zero-fill. Use it when every element is
    about to be written anyway --- reading an element before writing it gives
    whatever the allocator handed back, which is a garbage number rather than
    a crash, since matrix elements are plain scalars.

    Parameters:
        dtype: The dtype behind `T`, deduced rather than written.
        T: The type of the matrix elements. Defaults to `Float64`.

    Args:
        nrows: The number of rows in the matrix.
        ncols: The number of columns in the matrix.

    Returns:
        A new C-contiguous matrix of shape (nrows, ncols) with unspecified
        contents.
    """
    return Matrix[Scalar[dtype]](
        buffer=List[Scalar[dtype]](unsafe_uninit_length=nrows * ncols),
        nrows=nrows,
        ncols=ncols,
        row_stride=ncols,
        col_stride=1,
    )


# ===---------------------------------------------------------------------- ===#
# Shape-matching constructors
# ===---------------------------------------------------------------------- ===#
# Each takes an existing matrix or view and copies only its *shape*. The dtype
# is inferred from the argument; `astype` is how you change it. The result is
# always C-contiguous, following the rule in `routines/manipulation.mojo` that
# an owning result is dense in C order --- the input's layout is deliberately
# not reproduced, and for a strided view there is no layout to reproduce.


def zeros_like[
    dtype: DType, origin: Origin, //
](a: MatrixView[Scalar[dtype], origin]) -> Matrix[Scalar[dtype]]:
    """Creates a matrix of zeros with the same shape and dtype as `a`.

    Parameters:
        dtype: The data type of the matrix elements.
        origin: The origin of the input view.

    Args:
        a: The matrix or view whose shape is copied.

    Returns:
        A new C-contiguous matrix shaped like `a`, filled with zeros.
    """
    return zeros[Scalar[dtype]](a.nrows(), a.ncols())


def ones_like[
    dtype: DType, origin: Origin, //
](a: MatrixView[Scalar[dtype], origin]) -> Matrix[Scalar[dtype]]:
    """Creates a matrix of ones with the same shape and dtype as `a`.

    Parameters:
        dtype: The data type of the matrix elements.
        origin: The origin of the input view.

    Args:
        a: The matrix or view whose shape is copied.

    Returns:
        A new C-contiguous matrix shaped like `a`, filled with ones.
    """
    return ones[Scalar[dtype]](a.nrows(), a.ncols())


def full_like[
    dtype: DType, origin: Origin, //
](a: MatrixView[Scalar[dtype], origin], fill_value: Scalar[dtype]) -> Matrix[
    Scalar[dtype]
]:
    """Creates a matrix filled with `fill_value`, shaped and typed like `a`.

    Parameters:
        dtype: The data type of the matrix elements.
        origin: The origin of the input view.

    Args:
        a: The matrix or view whose shape is copied.
        fill_value: The value to fill the result with.

    Returns:
        A new C-contiguous matrix shaped like `a`, filled with `fill_value`.
    """
    return full[Scalar[dtype]](a.nrows(), a.ncols(), fill_value)


def empty_like[
    dtype: DType, origin: Origin, //
](a: MatrixView[Scalar[dtype], origin]) -> Matrix[Scalar[dtype]]:
    """Creates a matrix shaped and typed like `a`, with unspecified contents.

    Parameters:
        dtype: The data type of the matrix elements.
        origin: The origin of the input view.

    Args:
        a: The matrix or view whose shape is copied.

    Returns:
        A new C-contiguous matrix shaped like `a`, with unspecified contents.
    """
    return empty[Scalar[dtype]](a.nrows(), a.ncols())


# ===---------------------------------------------------------------------- ===#
# Range constructors
# ===---------------------------------------------------------------------- ===#
# Both return a `1 x n` **row** matrix, because Linamo has no 1-D type and a
# row is what NumPy's 1-D result prints as. Reach for `reshape(x, n, 1)` when
# a column is wanted.
#
# Both raise rather than return an empty matrix. `arange(5, 0)` is a mistake
# far more often than it is a deliberate request for zero elements, and a
# `1 x 0` matrix cannot be printed, indexed or multiplied by anything.


def arange[
    dtype: DType = DType.float64, //, T: Copyable & Deinitable = Scalar[dtype]
](start: T, stop: T, step: T = rebind[T](Scalar[dtype](1))) raises -> Matrix[
    Scalar[dtype]
] where (T == Scalar[dtype]):
    """Creates a row matrix of evenly spaced values over `[start, stop)`.

    `stop` is excluded, as in NumPy and in Python's `range`. The element count
    is `ceil((stop - start) / step)`, and `step` may be negative.

    Parameters:
        dtype: The dtype behind `T`, deduced rather than written.
        T: The type of the matrix elements. Defaults to `Float64`.

    Args:
        start: The first value.
        stop: The exclusive upper (or, for a negative step, lower) bound.
        step: The spacing between values. Defaults to 1.

    Returns:
        A new `1 x n` C-contiguous matrix holding the values.

    Raises:
        ValueError: If `step` is zero, or if the range is empty.
    """
    comptime fn_name = "arange(start, stop, step)"
    # `T` is `Scalar[dtype]`, but the compiler does not refine it inside the
    # body, so the three bounds are restated once and used as scalars below.
    var lo = rebind[Scalar[dtype]](start)
    var hi = rebind[Scalar[dtype]](stop)
    var by = rebind[Scalar[dtype]](step)
    if by == 0:
        raise ValueError(
            function=fn_name,
            message="`step` cannot be zero.",
        )

    var count = Int(ceil((Float64(hi) - Float64(lo)) / Float64(by)))
    if count <= 0:
        raise ValueError(
            function=fn_name,
            message=String(
                "The range from ",
                lo,
                " to ",
                hi,
                " with step ",
                by,
                " contains no values. Linamo has no zero-size matrices.",
            ),
        )

    var data = List[Scalar[dtype]](unsafe_uninit_length=count)
    for k in range(count):
        data[k] = lo + Scalar[dtype](k) * by
    return Matrix[Scalar[dtype]](
        buffer=data^,
        nrows=1,
        ncols=count,
        row_stride=count,
        col_stride=1,
    )


def arange[
    dtype: DType = DType.float64, //, T: Copyable & Deinitable = Scalar[dtype]
](stop: T) raises -> Matrix[Scalar[dtype]] where T == Scalar[dtype]:
    """Creates a row matrix of evenly spaced values over `[0, stop)`, step 1.

    Parameters:
        dtype: The dtype behind `T`, deduced rather than written.
        T: The type of the matrix elements. Defaults to `Float64`.

    Args:
        stop: The exclusive upper bound.

    Returns:
        A new `1 x n` C-contiguous matrix holding the values.

    Raises:
        ValueError: If `stop` is not positive.
    """
    return arange[Scalar[dtype]](0, rebind[Scalar[dtype]](stop), 1)


def linspace[
    dtype: DType = DType.float64, //, T: Copyable & Deinitable = Scalar[dtype]
](
    start: T,
    stop: T,
    num: Int = 50,
    endpoint: Bool = True,
) raises -> Matrix[
    Scalar[dtype]
] where (T == Scalar[dtype]):
    """Creates a row matrix of `num` evenly spaced values from `start` to `stop`.

    Unlike `arange`, the count is given and the spacing is derived, so the
    result never depends on floating-point accumulation. With `endpoint=True`
    the last element is exactly `stop`; with `endpoint=False` the interval is
    half-open and the spacing is `(stop - start) / num`.

    Parameters:
        dtype: The dtype behind `T`, deduced rather than written.
        T: The type of the matrix elements. Defaults to `Float64`.

    Args:
        start: The first value.
        stop: The last value if `endpoint` is True, else the exclusive bound.
        num: The number of values to generate. Defaults to 50.
        endpoint: Whether `stop` is included. Defaults to True.

    Returns:
        A new `1 x num` C-contiguous matrix holding the values.

    Raises:
        ValueError: If `num` is less than 1.
    """
    comptime fn_name = "linspace(start, stop, num, endpoint)"
    if num < 1:
        raise ValueError(
            function=fn_name,
            message=String(
                "`num` must be at least 1, got ",
                num,
                ". Linamo has no zero-size matrices.",
            ),
        )

    var first = Float64(rebind[Scalar[dtype]](start))
    var last = Float64(rebind[Scalar[dtype]](stop))
    var step = 0.0
    if num > 1:
        var divisor = Float64(num - 1) if endpoint else Float64(num)
        step = (last - first) / divisor

    var data = List[Scalar[dtype]](unsafe_uninit_length=num)
    for k in range(num):
        data[k] = Scalar[dtype](first + Float64(k) * step)
    if endpoint and num > 1:
        # Pin the last element instead of trusting `first + (num - 1) * step`,
        # which lands a rounding error short of `stop`. NumPy does the same.
        data[num - 1] = rebind[Scalar[dtype]](stop)
    return Matrix[Scalar[dtype]](
        buffer=data^,
        nrows=1,
        ncols=num,
        row_stride=num,
        col_stride=1,
    )


# ===---------------------------------------------------------------------- ===#
# Constructors from external representations
# ===---------------------------------------------------------------------- ===#


def from_list[
    T: Copyable & Deinitable = Float64
](
    var flat_list: List[T],
    nrows: Int,
    ncols: Int,
    order: String = "C",
) raises -> Matrix[T]:
    """Creates a matrix of the given shape from a flat list of elements.

    A positional spelling of `matrix(flat_list=..., nrows=..., ncols=...)`,
    named for the NuMojo routine it replaces.

    Parameters:
        T: The type of the matrix elements. Defaults to `Float64`.

    Args:
        flat_list: The elements, in `order`.
        nrows: The number of rows in the matrix.
        ncols: The number of columns in the matrix.
        order: The index order in which `flat_list` is read, "C" or "F".
            Defaults to "C".

    Returns:
        A new `nrows x ncols` matrix holding the elements.

    Raises:
        ValueError: If the list is empty, if its length is not `nrows * ncols`,
            or if `order` is neither "C" nor "F".
    """
    return matrix[T](
        flat_list=flat_list^, nrows=nrows, ncols=ncols, order=order
    )


def _parse_element[
    dtype: DType
](token: String, function: String) raises -> Scalar[dtype]:
    """Parses one whitespace-free token into an element of the matrix.

    Parameters:
        dtype: The data type of the matrix elements.

    Args:
        token: The text to parse.
        function: The caller's name, for the error message.

    Returns:
        The parsed value.

    Raises:
        ValueError: If `token` is not a number the dtype can hold.
    """
    try:
        comptime if dtype.is_floating_point():
            return Scalar[dtype](atof(token))
        else:
            return Scalar[dtype](atol(token))
    except:
        raise ValueError(
            function=function,
            message=String("Cannot parse '", token, "' as a number."),
        )


def _parse_parsable[
    T: Copyable & Deinitable & Parsable
](token: String, function: String) raises -> T:
    """Parses one whitespace-free token into an arbitrary-precision element.

    The twin of `_parse_element` above. `Scalar[dtype]` goes through `atof` and
    `atol`, which cap the value at the width of the dtype; a `Parsable` element
    reads the token itself and keeps every digit of it.

    Parameters:
        T: The type of the matrix elements.

    Args:
        token: The text to parse.
        function: The caller\'s name, for the error message.

    Returns:
        The parsed value.

    Raises:
        ValueError: If `token` is not a literal the element type accepts.
    """
    try:
        return T.from_string(token)
    except e:
        raise ValueError(
            function=function,
            message=String("Cannot parse '", token, "' as a number: ", e),
        )


def _tokenize_rows[
    T: Copyable & Deinitable,
    //,
    parse: def(String, String) raises thin -> T,
](text: String, function: String) raises -> List[List[T]]:
    """Splits a bracketed matrix literal into rows of parsed elements.

    Elements are separated by whitespace or by commas; a nested `[...]` opens
    a row. Text with no nesting is one row, so `"[1, 2, 3]"` and `"1 2 3"`
    both parse as a single row of three.

    The bracket walk is the same whatever the elements are, so the element type
    enters only through `parse`. That is what lets one literal syntax serve
    `Float64` and `BigDecimal` alike.

    Parameters:
        T: The type of the matrix elements.
        parse: Turns one token into an element, given the caller's name for
            the error message.

    Args:
        text: The literal to split.
        function: The caller's name, for the error message.

    Returns:
        One list of elements per row.

    Raises:
        ValueError: If the brackets are unbalanced or nested more than two
            deep, if a token is not a number, or if no elements are found.
    """
    comptime LBRACKET = ord("[")
    comptime RBRACKET = ord("]")

    var bytes = text.as_bytes()
    var rows = List[List[T]]()
    var current = List[T]()
    var depth = 0
    var token_start = -1

    for i in range(len(bytes)):
        var c = Int(bytes[i])
        var opens = c == LBRACKET
        var closes = c == RBRACKET
        # Anything that is not a digit, sign, dot or exponent letter ends the
        # token. Being liberal here means `_parse_element` reports the bad
        # token rather than this loop guessing at what a number looks like.
        var separates = (
            c == ord(" ")
            or c == ord("\t")
            or c == ord("\n")
            or c == ord("\r")
            or c == ord(",")
            or c == ord(";")
        )

        if (opens or closes or separates) and token_start >= 0:
            current.append(parse(String(text[byte=token_start:i]), function))
            token_start = -1

        if opens:
            depth += 1
            if depth > 2:
                raise ValueError(
                    function=function,
                    message=(
                        "A matrix literal nests at most two levels deep,"
                        " as in '[[1, 2], [3, 4]]'."
                    ),
                )
        elif closes:
            if depth == 0:
                raise ValueError(
                    function=function,
                    message="Unbalanced brackets: a ']' has no matching '['.",
                )
            if depth == 2:
                rows.append(current^)
                current = List[T]()
            depth -= 1
        elif not separates and token_start < 0:
            token_start = i

    if token_start >= 0:
        current.append(
            parse(String(text[byte = token_start : len(bytes)]), function)
        )
    if depth != 0:
        raise ValueError(
            function=function,
            message="Unbalanced brackets: a '[' has no matching ']'.",
        )
    if len(current) > 0:
        # A literal with no nesting, such as "1 2 3", is a single row.
        rows.append(current^)
    if len(rows) == 0:
        raise ValueError(
            function=function,
            message=String("No elements found in '", text, "'."),
        )
    return rows^


def from_string[
    dtype: DType = DType.float64, //, T: Copyable & Deinitable = Scalar[dtype]
](text: String, order: String = "C") raises -> Matrix[Scalar[dtype]] where (
    T == Scalar[dtype]
):
    """Creates a matrix from a bracketed literal, deducing its shape.

    Rows are written as nested brackets and elements are separated by commas
    or whitespace, so `"[[1, 2, 3], [4, 5, 6]]"` gives a 2x3 matrix. A literal
    with no nesting, such as `"[1, 2, 3]"` or `"1 2 3"`, is a single row.

    Parameters:
        dtype: The dtype behind `T`, deduced rather than written.
        T: The type of the matrix elements. Defaults to `Float64`.

    Args:
        text: The literal to parse.
        order: The memory layout of the result, "C" or "F". Defaults to "C".

    Returns:
        A new matrix holding the parsed elements.

    Raises:
        ValueError: If the brackets are unbalanced, if a token is not a
            number, if the rows have different lengths, or if `order` is
            neither "C" nor "F".
    """
    comptime fn_name = "from_string(text, order)"
    return matrix[Scalar[dtype]](
        _tokenize_rows[parse=_parse_element[dtype]](text, fn_name), order
    )


def from_string[
    dtype: DType = DType.float64, //, T: Copyable & Deinitable = Scalar[dtype]
](text: String, nrows: Int, ncols: Int, order: String = "C") raises -> Matrix[
    Scalar[dtype]
] where (T == Scalar[dtype]):
    """Creates a matrix of the given shape from a literal of `nrows * ncols`
    elements.

    The bracket structure of `text` is ignored: every element found is read in
    `order` into the requested shape, so `"1 2 3 4"` and `"[[1, 2], [3, 4]]"`
    give the same 2x2 result.

    Parameters:
        dtype: The dtype behind `T`, deduced rather than written.
        T: The type of the matrix elements. Defaults to `Float64`.

    Args:
        text: The literal to parse.
        nrows: The number of rows in the matrix.
        ncols: The number of columns in the matrix.
        order: The index order in which elements are read, "C" or "F".
            Defaults to "C".

    Returns:
        A new `nrows x ncols` matrix holding the parsed elements.

    Raises:
        ValueError: If a token is not a number, if the element count is not
            `nrows * ncols`, or if `order` is neither "C" nor "F".
    """
    comptime fn_name = "from_string(text, nrows, ncols, order)"
    var rows = _tokenize_rows[parse=_parse_element[dtype]](text, fn_name)
    var flat = List[Scalar[dtype]]()
    for row in rows:
        for element in row:
            flat.append(element)
    return matrix[Scalar[dtype]](
        flat_list=flat^, nrows=nrows, ncols=ncols, order=order
    )


def from_string[
    T: Copyable & Deinitable & Parsable
](text: String, order: String = "C") raises -> Matrix[T]:
    """Creates an arbitrary-precision matrix from a bracketed literal.

    The twin of the routine above, for elements that parse themselves. It is
    the short spelling of a matrix the caller could build element by element;
    what it is not is a conversion from `Float64`. `BigDecimal` and
    `Decimal128` have no implicit constructor from one, and going through one
    would round the literal to a binary float before the element ever saw it.
    `"0.1"` read this way is a tenth; `Float64(0.1)` is not.

    Parameters:
        T: The type of the matrix elements, such as `BDec` or `Dec128`.

    Args:
        text: The literal to parse.
        order: The memory layout of the result, "C" or "F". Defaults to "C".

    Returns:
        A new matrix holding the parsed elements.

    Raises:
        ValueError: If the brackets are unbalanced, if a token is not a
            literal the element type accepts, if the rows have different
            lengths, or if `order` is neither "C" nor "F".
    """
    comptime fn_name = "from_string(text, order)"
    return matrix[T](
        _tokenize_rows[parse=_parse_parsable[T]](text, fn_name), order
    )


def from_string[
    T: Copyable & Deinitable & Parsable
](text: String, nrows: Int, ncols: Int, order: String = "C") raises -> Matrix[
    T
]:
    """Creates an arbitrary-precision matrix of the given shape from a literal.

    The bracket structure of `text` is ignored: every element found is read in
    `order` into the requested shape.

    Parameters:
        T: The type of the matrix elements, such as `BDec` or `Dec128`.

    Args:
        text: The literal to parse.
        nrows: The number of rows in the matrix.
        ncols: The number of columns in the matrix.
        order: The index order in which elements are read, "C" or "F".
            Defaults to "C".

    Returns:
        A new `nrows x ncols` matrix holding the parsed elements.

    Raises:
        ValueError: If a token is not a literal the element type accepts, if
            the element count is not `nrows * ncols`, or if `order` is neither
            "C" nor "F".
    """
    comptime fn_name = "from_string(text, nrows, ncols, order)"
    var rows = _tokenize_rows[parse=_parse_parsable[T]](text, fn_name)
    var flat = List[T]()
    for row in rows:
        for element in row:
            flat.append(element.copy())
    return matrix[T](flat_list=flat^, nrows=nrows, ncols=ncols, order=order)
