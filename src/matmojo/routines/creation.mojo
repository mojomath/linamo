"""
This module defines routines for creating matrices and matrix views in MatMojo.
"""

from matmojo.types.errors import ValueError
from matmojo.types.matrix import Matrix
from matmojo.types.matrix_view import MatrixView
from matmojo.types.static_matrix import StaticMatrix
from matmojo.utils.indexing import get_offset

# ===---------------------------------------------------------------------- ===#
# Matrix creation routines
# - Create matrices from nested lists.
# - Create matrices from flat lists with specified shapes.
# - Create static matrices from nested lists with compile-time dimensions.
# - Create static matrices from flat lists with compile-time dimensions.
# ===---------------------------------------------------------------------- ===#


def matrix[
    dtype: DType = DType.float64
](list: List[List[Scalar[dtype]]], order: String = "C") raises -> Matrix[dtype]:
    """Initializes the matrix with a list of lists.

    Parameters:
        dtype: The data type of the matrix elements.

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
            file="src/matmojo/routines/creation.mojo",
            function="matrix()",
            message="Data cannot be an empty list.",
            previous_error=None,
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
            file="src/matmojo/routines/creation.mojo",
            function="matrix()",
            message="Invalid order. Must be 'C' or 'F'.",
            previous_error=None,
        )

    for row in list:
        if len(row) != ncols:
            raise ValueError(
                file="src/matmojo/routines/creation.mojo",
                function="matrix()",
                message="All rows must have the same length.",
                previous_error=None,
            )

    var data = List[Scalar[dtype]](unsafe_uninit_length=nrows * ncols)
    var row_index = 0
    for row in list:
        var col_index = 0
        for element in row:
            data[row_index * row_stride + col_index * col_stride] = element
            col_index += 1
        row_index += 1
    return Matrix[dtype](
        data=data^,
        nrows=nrows,
        ncols=ncols,
        row_stride=row_stride,
        col_stride=col_stride,
    )


def matrix[
    dtype: DType = DType.float64
](
    *,
    var flat_list: List[Scalar[dtype]],
    nrows: Int,
    ncols: Int,
    order: String = "C",
) raises -> Matrix[dtype]:
    """Initializes the matrix with a list and shape.

    Parameters:
        dtype: The data type of the matrix elements.

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
            file="src/matmojo/routines/creation.mojo",
            function="matrix()",
            message="Data cannot be an empty list.",
            previous_error=None,
        )
    if len(flat_list) != nrows * ncols:
        raise ValueError(
            file="src/matmojo/routines/creation.mojo",
            function="matrix()",
            message="Data length does not match the specified shape.",
            previous_error=None,
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
            file="src/matmojo/routines/creation.mojo",
            function="matrix()",
            message="Invalid order. Must be 'C' or 'F'.",
            previous_error=None,
        )
    return Matrix[dtype](
        data=flat_list^,
        nrows=nrows,
        ncols=ncols,
        row_stride=row_stride,
        col_stride=col_stride,
    )


def smatrix[
    nrows: Int, ncols: Int, dtype: DType = DType.float64
](var list: List[List[Scalar[dtype]]]) raises -> StaticMatrix[
    dtype, nrows, ncols
]:
    """Initializes the static matrix with a list of lists.

    Parameters:
        nrows: The number of rows in the matrix.
        ncols: The number of columns in the matrix.
        dtype: The data type of the matrix elements. Defaults to `DType.float64`.

    Args:
        list: A list of lists representing the rows of the matrix.
    """

    if len(list) != nrows:
        raise ValueError(
            file="src/matmojo/routines/creation.mojo",
            function="matrix()",
            message="Number of rows in list does not match nrows.",
            previous_error=None,
        )
    var result = StaticMatrix[dtype, nrows, ncols]()  # Initialize with zeros
    for row_index in range(len(list)):
        if len(list[row_index]) != ncols:
            raise ValueError(
                file="src/matmojo/routines/creation.mojo",
                function="matrix()",
                message="All rows must have the same length as ncols.",
                previous_error=None,
            )
        for col_index in range(ncols):
            result.data[
                row_index * result.row_stride + col_index * result.col_stride
            ] = list[row_index][col_index]
    return result^


def smatrix[
    nrows: Int, ncols: Int, dtype: DType = DType.float64
](*, var flat_list: List[Scalar[dtype]]) raises -> StaticMatrix[
    dtype, nrows, ncols
]:
    """Initializes the static matrix with a list of values.

    Parameters:
        nrows: The number of rows in the matrix.
        ncols: The number of columns in the matrix.
        dtype: The data type of the matrix elements.

    Args:
        flat_list: A list of values.
    """
    if len(flat_list) != nrows * ncols:
        raise ValueError(
            file="src/matmojo/routines/creation.mojo",
            function="matrix()",
            message="Number of rows in list does not match nrows.",
            previous_error=None,
        )
    var result = StaticMatrix[dtype, nrows, ncols]()  # Initialize with zeros
    var offset = 0
    for i in range(nrows):
        for j in range(ncols):
            result.data[
                i * result.row_stride + j * result.col_stride
            ] = flat_list[offset]
            offset += 1
    return result^


# ===---------------------------------------------------------------------- ===#
# Convenience constructors
# - zeros(), ones(), full()
# - eye(), identity()
# - diag()
# ===---------------------------------------------------------------------- ===#


def zeros[
    dtype: DType = DType.float64
](nrows: Int, ncols: Int) -> Matrix[dtype]:
    """Creates a matrix filled with zeros.

    Parameters:
        dtype: The data type of the matrix elements. Defaults to `DType.float64`.

    Args:
        nrows: The number of rows in the matrix.
        ncols: The number of columns in the matrix.

    Returns:
        A new matrix of shape (nrows, ncols) filled with zeros.
    """
    return Matrix[dtype](
        data=List[Scalar[dtype]](length=nrows * ncols, fill=0),
        nrows=nrows,
        ncols=ncols,
        row_stride=ncols,
        col_stride=1,
    )


def ones[dtype: DType = DType.float64](nrows: Int, ncols: Int) -> Matrix[dtype]:
    """Creates a matrix filled with ones.

    Parameters:
        dtype: The data type of the matrix elements. Defaults to `DType.float64`.

    Args:
        nrows: The number of rows in the matrix.
        ncols: The number of columns in the matrix.

    Returns:
        A new matrix of shape (nrows, ncols) filled with ones.
    """
    return Matrix[dtype](
        data=List[Scalar[dtype]](length=nrows * ncols, fill=1),
        nrows=nrows,
        ncols=ncols,
        row_stride=ncols,
        col_stride=1,
    )


def full[
    dtype: DType = DType.float64
](nrows: Int, ncols: Int, fill_value: Scalar[dtype]) -> Matrix[dtype]:
    """Creates a matrix filled with a specified value.

    Parameters:
        dtype: The data type of the matrix elements. Defaults to `DType.float64`.

    Args:
        nrows: The number of rows in the matrix.
        ncols: The number of columns in the matrix.
        fill_value: The value to fill the matrix with.

    Returns:
        A new matrix of shape (nrows, ncols) filled with fill_value.
    """
    return Matrix[dtype](
        data=List[Scalar[dtype]](length=nrows * ncols, fill=fill_value),
        nrows=nrows,
        ncols=ncols,
        row_stride=ncols,
        col_stride=1,
    )


def eye[dtype: DType = DType.float64](n: Int) -> Matrix[dtype]:
    """Creates an n×n identity matrix.

    Parameters:
        dtype: The data type of the matrix elements. Defaults to `DType.float64`.

    Args:
        n: The number of rows and columns in the identity matrix.

    Returns:
        A new n×n identity matrix with ones on the diagonal and zeros elsewhere.
    """
    var data = List[Scalar[dtype]](length=n * n, fill=0)
    for i in range(n):
        data[i * n + i] = 1
    return Matrix[dtype](
        data=data^,
        nrows=n,
        ncols=n,
        row_stride=n,
        col_stride=1,
    )


def identity[dtype: DType = DType.float64](n: Int) -> Matrix[dtype]:
    """Creates an n×n identity matrix. Alias for `eye()`.

    Parameters:
        dtype: The data type of the matrix elements. Defaults to `DType.float64`.

    Args:
        n: The number of rows and columns in the identity matrix.

    Returns:
        A new n×n identity matrix with ones on the diagonal and zeros elsewhere.
    """
    return eye[dtype](n)


def diag[dtype: DType](var values: List[Scalar[dtype]]) -> Matrix[dtype]:
    """Creates a square diagonal matrix from a list of values.

    Parameters:
        dtype: The data type of the matrix elements.

    Args:
        values: A list of diagonal values.

    Returns:
        A new n×n matrix with the given values on the diagonal and zeros
        elsewhere, where n is the length of `values`.
    """
    var n = len(values)
    var data = List[Scalar[dtype]](length=n * n, fill=0)
    for i in range(n):
        data[i * n + i] = values[i]
    return Matrix[dtype](
        data=data^,
        nrows=n,
        ncols=n,
        row_stride=n,
        col_stride=1,
    )


def diag[dtype: DType](mat: Matrix[dtype]) raises -> List[Scalar[dtype]]:
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
    if mat.nrows != mat.ncols:
        raise ValueError(
            file="src/matmojo/routines/creation.mojo",
            function="diag()",
            message="Matrix must be square to extract diagonal.",
            previous_error=None,
        )
    var n = mat.nrows
    var result = List[Scalar[dtype]](length=n, fill=0)
    for i in range(n):
        result[i] = mat.data[get_offset(i, i, mat.row_stride, mat.col_stride)]
    return result^
