"""
Numpy interoperability for Linamo.

Provides functions to convert between numpy ndarrays and Linamo matrices:

- `matrix_from_numpy()`: Create a Matrix from a 2D numpy ndarray (data copy).
- `to_numpy()`: Export a Matrix to a numpy ndarray (data copy).
"""

from std.python import Python, PythonObject
from std.memory import unsafe_memcpy

from linamo.types.matrix import Matrix


# ===----------------------------------------------------------------------===#
# numpy → Matrix
# ===----------------------------------------------------------------------===#


def matrix_from_numpy[
    dtype: DType = DType.float64
](data: PythonObject) raises -> Matrix[dtype]:
    """Create a Matrix from a numpy ndarray.

    The numpy array must be 2D. Data is copied from numpy memory into
    a new Matrix (the Matrix owns its own data). The resulting Matrix
    is always C-contiguous regardless of the numpy array's memory layout.

    Parameters:
        dtype: The data type of the matrix elements. Defaults to float64.

    Args:
        data: A 2D numpy ndarray (PythonObject).

    Raises:
        Error: If the array is not 2D or is empty.

    Returns:
        A Matrix[dtype] with the same shape and data as the numpy array.

    Example:
        ```mojo
        from std.python import Python
        from linamo.routines.numpy_interop import matrix_from_numpy

        def main() raises:
            var np = Python.import_module("numpy")
            var np_arr = np.arange(6.0).reshape(2, 3)
            var mat = matrix_from_numpy(np_arr)
        ```
    """
    var np = Python.import_module("numpy")

    # Validate dimensionality
    var ndim = Int(py=data.ndim)
    if ndim != 2:
        raise Error(
            "matrix_from_numpy: expected 2D array, got " + String(ndim) + "D"
        )

    var nrows = Int(py=data.shape[0])
    var ncols = Int(py=data.shape[1])

    if nrows == 0 or ncols == 0:
        raise Error("matrix_from_numpy: array must not be empty")

    # Map Mojo DType to numpy dtype for correct interpretation
    var np_dtype = np.float64

    comptime if dtype == DType.float32:
        np_dtype = np.float32
    elif dtype == DType.float16:
        np_dtype = np.float16
    elif dtype == DType.int64:
        np_dtype = np.int64
    elif dtype == DType.int32:
        np_dtype = np.int32
    elif dtype == DType.int16:
        np_dtype = np.int16
    elif dtype == DType.int8:
        np_dtype = np.int8

    # Ensure the array is C-contiguous and the correct dtype
    var np_arr = np.ascontiguousarray(data, dtype=np_dtype)

    # Get the data pointer from numpy's __array_interface__
    var pointer = np_arr.__array_interface__["data"][0].unsafe_get_as_pointer[
        dtype
    ]()

    # Create Matrix data buffer and copy from numpy memory
    var mat_data = List[Scalar[dtype]](length=nrows * ncols, fill=0)
    unsafe_memcpy(dest=mat_data._data, src=pointer, count=nrows * ncols)

    return Matrix[dtype](
        data=mat_data^,
        nrows=nrows,
        ncols=ncols,
        row_stride=ncols,
        col_stride=1,
    )


# ===----------------------------------------------------------------------===#
# Matrix → numpy
# ===----------------------------------------------------------------------===#


def to_numpy[dtype: DType](mat: Matrix[dtype]) raises -> PythonObject:
    """Export a Matrix to a numpy ndarray.

    Data is always copied. The resulting numpy array is C-contiguous.

    Parameters:
        dtype: The data type of the matrix elements.

    Args:
        mat: The Matrix to export.

    Returns:
        A numpy ndarray (PythonObject) with the same shape and data.

    Example:
        ```mojo
        from linamo import matrix
        from linamo.routines.numpy_interop import to_numpy

        def main() raises:
            var mat = matrix[DType.float64]([[1.0, 2.0], [3.0, 4.0]])
            var np_arr = to_numpy(mat)
        ```
    """
    var np = Python.import_module("numpy")

    # Map Mojo DType to numpy dtype
    var np_dtype = np.float64

    comptime if dtype == DType.float32:
        np_dtype = np.float32
    elif dtype == DType.float16:
        np_dtype = np.float16
    elif dtype == DType.int64:
        np_dtype = np.int64
    elif dtype == DType.int32:
        np_dtype = np.int32
    elif dtype == DType.int16:
        np_dtype = np.int16
    elif dtype == DType.int8:
        np_dtype = np.int8

    # Create a numpy array and copy data into it
    var nrows = mat.nrows
    var ncols = mat.ncols
    var result = np.zeros(nrows * ncols, dtype=np_dtype).reshape(nrows, ncols)

    # Get target pointer
    var np_ptr = result.__array_interface__["data"][0].unsafe_get_as_pointer[
        dtype
    ]()

    # For C-contiguous matrix, straight memcpy
    if mat.is_c_contiguous():
        unsafe_memcpy(dest=np_ptr, src=mat.data._data, count=nrows * ncols)
    else:
        # General case: copy element by element
        for i in range(nrows):
            for j in range(ncols):
                np_ptr[unsafe_offset=i * ncols + j] = mat.data[
                    i * mat.row_stride + j * mat.col_stride
                ]

    return result^
