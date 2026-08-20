"""
Creation routines for matrices of arbitrary-precision elements.
"""

from decimo import Numeric

from linamo.types.errors import ValueError
from linamo.types.matrix import Matrix


def zeros[T: Numeric](nrows: Int, ncols: Int) -> Matrix[T]:
    """Creates a matrix filled with zeros.

    The counterpart of `linamo.zeros`, which asks its element type for a `0`
    that only a scalar has. This asks `Numeric` for one instead.

    Parameters:
        T: The type of the matrix elements.

    Args:
        nrows: The number of rows in the matrix.
        ncols: The number of columns in the matrix.

    Returns:
        A new C-contiguous `nrows x ncols` matrix of zeros.
    """
    return Matrix[T](
        List[T](length=nrows * ncols, fill=T.zero()), nrows, ncols, ncols, 1
    )


def ones[T: Numeric](nrows: Int, ncols: Int) -> Matrix[T]:
    """Creates a matrix filled with ones.

    Parameters:
        T: The type of the matrix elements.

    Args:
        nrows: The number of rows in the matrix.
        ncols: The number of columns in the matrix.

    Returns:
        A new C-contiguous `nrows x ncols` matrix of ones.
    """
    return Matrix[T](
        List[T](length=nrows * ncols, fill=T.one()), nrows, ncols, ncols, 1
    )


def eye[T: Numeric](n: Int) -> Matrix[T]:
    """Creates an `n x n` identity matrix.

    Parameters:
        T: The type of the matrix elements.

    Args:
        n: The number of rows and columns.

    Returns:
        A new `n x n` matrix with ones on the diagonal and zeros elsewhere.
    """
    var data = List[T](length=n * n, fill=T.zero())
    for i in range(n):
        data[i * n + i] = T.one()
    return Matrix[T](data^, n, n, n, 1)


def identity[T: Numeric](n: Int) -> Matrix[T]:
    """Creates an `n x n` identity matrix. Alias for `eye()`.

    Parameters:
        T: The type of the matrix elements.

    Args:
        n: The number of rows and columns.

    Returns:
        A new `n x n` matrix with ones on the diagonal and zeros elsewhere.
    """
    return eye[T](n)


def diag[T: Numeric](var values: List[T]) -> Matrix[T]:
    """Creates a square diagonal matrix from a list of values.

    Parameters:
        T: The type of the matrix elements.

    Args:
        values: The diagonal values.

    Returns:
        A new `n x n` matrix with `values` on the diagonal and zeros
        elsewhere, where `n` is the length of `values`.
    """
    var n = len(values)
    var data = List[T](length=n * n, fill=T.zero())
    for i in range(n):
        data[i * n + i] = values[i].copy()
    return Matrix[T](data^, n, n, n, 1)
