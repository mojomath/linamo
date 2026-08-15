"""
This module defines the `MatrixView` type, which is a view on a `Matrix`.
"""

import std.math as builtin_math
import matmojo.routines.math

from matmojo.traits.matrix_like import MatrixLike
from matmojo.types.errors import ValueError
from matmojo.types.matrix import Matrix
from matmojo.utils.indexing import get_offset, indices_within_bounds
from std.memory import Pointer


struct MatrixView[mut: Bool, //, dtype: DType, origin: Origin[mut=mut]](
    MatrixLike, Writable
):
    """A 2D matrix view type that references another Matrix.

    Parameters:
        mut: Whether the reference to the matrix is mutable.
        dtype: The data type of the matrix elements. Defaults to `DType.float64`.
        origin: The origin of the matrix.
    """

    comptime ElementType = Scalar[Self.dtype]
    """The type of the elements in the matrix view, derived from the dtype."""

    var data: Span[Self.ElementType, Self.origin]
    """A span representing the data of the matrix view."""
    var nrows: Int
    """The number of rows in the matrix view."""
    var ncols: Int
    """The number of columns in the matrix view."""
    var row_stride: Int
    """The row stride of the matrix view."""
    var col_stride: Int
    """The column stride of the matrix view."""
    var offset: Int
    """The offset in the base matrix data where the view starts."""

    # ===--------------------------------------------------------------------===#
    # Retrieve attributes
    # ===--------------------------------------------------------------------===#

    def get_data(self) -> Span[Self.ElementType, Self.origin]:
        """Returns the underlying data of the matrix."""
        return self.data

    def get_nrows(self) -> Int:
        """Returns the number of rows in the matrix."""
        return self.nrows

    def get_ncols(self) -> Int:
        """Returns the number of columns in the matrix."""
        return self.ncols

    def get_row_stride(self) -> Int:
        """Returns the row stride of the matrix."""
        return self.row_stride

    def get_col_stride(self) -> Int:
        """Returns the column stride of the matrix."""
        return self.col_stride

    def get_offset(self) -> Int:
        """Returns the offset in the underlying data buffer for the matrix."""
        return self.offset

    def get_size(self) -> Int:
        """Returns the total number of elements in the matrix."""
        return self.nrows * self.ncols

    def is_c_contiguous(self) -> Bool:
        """Returns True if the view is C-contiguous (row-major, dense)."""
        return self.col_stride == 1 and self.row_stride == self.ncols

    def is_f_contiguous(self) -> Bool:
        """Returns True if the view is F-contiguous (column-major, dense)."""
        return self.row_stride == 1 and self.col_stride == self.nrows

    def is_row_contiguous(self) -> Bool:
        """Returns True if elements within each row are contiguous (col_stride == 1).

        Allows padding between rows (row_stride >= ncols).
        """
        return self.col_stride == 1

    def is_col_contiguous(self) -> Bool:
        """Returns True if elements within each column are contiguous (row_stride == 1).

        Allows padding between columns (col_stride >= nrows).
        """
        return self.row_stride == 1

    # ===--------------------------------------------------------------------===#
    # Life Cycle Management
    # ===--------------------------------------------------------------------===#

    def __init__(
        out self,
        data: Span[Self.ElementType, Self.origin],
        *,
        nrows: Int,
        ncols: Int,
        row_stride: Int,
        col_stride: Int,
        offset: Int,
    ):
        """Initializes a MatrixView instance that references a Matrix.

        Args:
            data: A span representing the matrix data.
            nrows: The number of rows in the view.
            ncols: The number of columns in the view.
            row_stride: The row stride for accessing elements.
            col_stride: The column stride for accessing elements.
            offset: The starting offset in the matrix data.
        """
        self.data = data
        self.nrows = nrows
        self.ncols = ncols
        self.row_stride = row_stride
        self.col_stride = col_stride
        self.offset = offset

    def __init__(
        out self,
        data: Span[Self.ElementType, Self.origin],
        *,
        slice_x: Slice,
        slice_y: Slice,
        initial_nrows: Int,
        initial_ncols: Int,
        initial_row_stride: Int,
        initial_col_stride: Int,
        initial_offset: Int,
    ):
        """Initializes a MatrixView instance using slicing parameters."""
        self.data = data
        var start_x, end_x, step_x = slice_x.indices(initial_nrows)
        var start_y, end_y, step_y = slice_y.indices(initial_ncols)
        self.offset = (
            initial_offset
            + start_x * initial_row_stride
            + start_y * initial_col_stride
        )
        self.nrows = builtin_math.ceildiv(end_x - start_x, step_x)
        self.ncols = builtin_math.ceildiv(end_y - start_y, step_y)
        self.row_stride = initial_row_stride * step_x
        self.col_stride = initial_col_stride * step_y

    # ===--------------------------------------------------------------------===#
    # Element Access and Mutation
    # ===--------------------------------------------------------------------===#

    def __getitem__(self, row: Int, col: Int) -> Self.ElementType:
        """Accesses an element of the matrix view using row and column indices.
        """
        var index = self.offset + row * self.row_stride + col * self.col_stride
        return self.data[index]

    # [Mojo Miji]
    # The return type can also be written as:
    # `MatrixView[Self.dtype, Self.origin]`
    # It means that the view on view has the same data type and origin as the
    # original view.
    def __getitem__(
        self, rows: Slice, cols: Slice
    ) raises -> MatrixView[Self.dtype, Self.origin]:
        """Gets a view of the specified row with a slice of columns."""
        return Self(
            data=self.data,
            slice_x=rows,
            slice_y=cols,
            initial_nrows=self.nrows,
            initial_ncols=self.ncols,
            initial_row_stride=self.row_stride,
            initial_col_stride=self.col_stride,
            initial_offset=self.offset,
        )

    def get_unsafe(self, row: Int, col: Int) -> Scalar[Self.dtype]:
        """Gets the element at the specified indices without bounds checking.

        This method is unsafe because it does not perform bounds checking on
        the provided indices. It should only be used when the caller can
        guarantee that the indices are valid.

        Args:
            row: The row index.
            col: The column index.

        Returns:
            The element at the specified indices.
        """
        debug_assert(
            indices_within_bounds(row, col, self.nrows, self.ncols),
            "Debug assertion failed: Indices out of bounds in `unsafe_load`",
        )
        var offset = get_offset(
            row, col, self.row_stride, self.col_stride, self.offset
        )
        return self.data[offset]

    # ===--------------------------------------------------------------------===#
    # String Representation and Writing
    # ===--------------------------------------------------------------------===#

    def __str__(self) -> String:
        """Returns a string representation of the matrix."""
        var result = String("")
        for i in range(self.nrows):
            for j in range(self.ncols):
                result += (
                    String(
                        self.data[
                            self.offset
                            + i * self.row_stride
                            + j * self.col_stride
                        ]
                    )
                    + "\t"
                )
            if i < self.nrows - 1:
                result += "\n"
        return result

    def write_to[W: Writer](self, mut writer: W):
        """Writes the matrix view to a writer."""
        writer.write("MatrixView, ")
        writer.write(self.dtype)
        writer.write(", ")
        writer.write(self.nrows)
        writer.write("x")
        writer.write(self.ncols)
        writer.write(", strides: ")
        writer.write(self.row_stride)
        writer.write("-")
        writer.write(self.col_stride)
        writer.write(", offset: ")
        writer.write(self.offset)
        writer.write(":\n")
        for i in range(self.nrows):
            if i == 0:
                writer.write("[[\t")
            else:
                writer.write(" [\t")
            for j in range(self.ncols):
                writer.write(
                    self.data[
                        self.offset + i * self.row_stride + j * self.col_stride
                    ]
                )
                writer.write("\t")
            writer.write("]")
            if i < self.nrows - 1:
                writer.write("\n")
            else:
                writer.write("]\n")

    # ===--------------------------------------------------------------------===#
    # Basic math dunders
    # ===--------------------------------------------------------------------===#

    def __add__[
        origin_b: Origin
    ](self, other: MatrixView[Self.dtype, origin_b]) raises -> Matrix[
        Self.dtype
    ]:
        """Performs element-wise addition of two matrix views."""
        return matmojo.routines.math.add(self, other)

    def __add__(self, other: Matrix[Self.dtype]) raises -> Matrix[Self.dtype]:
        """Performs element-wise addition of a matrix view and a matrix."""
        return matmojo.routines.math.add(self, other)

    def __sub__[
        origin_b: Origin
    ](self, other: MatrixView[Self.dtype, origin_b]) raises -> Matrix[
        Self.dtype
    ]:
        """Performs element-wise subtraction of two matrix views."""
        return matmojo.routines.math.sub(self, other)

    def __sub__(self, other: Matrix[Self.dtype]) raises -> Matrix[Self.dtype]:
        """Performs element-wise subtraction of a matrix view and a matrix."""
        return matmojo.routines.math.sub(self, other)

    def __mul__[
        origin_b: Origin
    ](self, other: MatrixView[Self.dtype, origin_b]) raises -> Matrix[
        Self.dtype
    ]:
        """Performs element-wise multiplication of two matrix views."""
        return matmojo.routines.math.mul(self, other)

    def __mul__(self, other: Matrix[Self.dtype]) raises -> Matrix[Self.dtype]:
        """Performs element-wise multiplication of a matrix view and a matrix.
        """
        return matmojo.routines.math.mul(self, other)

    def __truediv__[
        origin_b: Origin
    ](self, other: MatrixView[Self.dtype, origin_b]) raises -> Matrix[
        Self.dtype
    ]:
        """Performs element-wise division of two matrix views."""
        return matmojo.routines.math.div(self, other)

    def __truediv__(
        self, other: Matrix[Self.dtype]
    ) raises -> Matrix[Self.dtype]:
        """Performs element-wise division of a matrix view and a matrix."""
        return matmojo.routines.math.div(self, other)

    def __matmul__[
        origin_b: Origin
    ](self, other: MatrixView[Self.dtype, origin_b]) raises -> Matrix[
        Self.dtype
    ]:
        """Performs matrix multiplication of two matrix views."""
        return matmojo.routines.math.matmul(self, other)

    def __matmul__(
        self, other: Matrix[Self.dtype]
    ) raises -> Matrix[Self.dtype]:
        """Performs matrix multiplication of a matrix view and a matrix."""
        return matmojo.routines.math.matmul(self, other)
