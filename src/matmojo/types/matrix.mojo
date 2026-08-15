"""
This module defines the `Matrix` type, which is a dynamically sized 2D matrix.
"""

from matmojo.traits.matrix_like import MatrixLike
from matmojo.types.errors import IndexError, ValueError
from matmojo.types.matrix_view import MatrixView
import matmojo.routines.math
from matmojo.utils.indexing import (
    get_offset,
    indices_within_bounds,
)


struct Matrix[dtype: DType](Copyable, MatrixLike, Movable, Writable):
    """A 2D matrix type.
    A matrix owns its data and can write to it. The elements are stored in a
    contiguous block of memory in either row-major (C-contiguous) or
    column-major (Fortran-contiguous) order.

    Parameters:
        dtype: The data type of the matrix elements. Defaults to `DType.float64`.
    """

    # [Mojo Miji]
    # `comptime` can be used to define a type alias that can be translated back
    # to the original type at compile time. We do this for convenience.
    comptime ElementType = Scalar[Self.dtype]
    """The type of the elements in the matrix, derived from the dtype."""

    # [Mojo Miji]
    # If we want to implement a simple 2D matrix type,
    # the following three attributes are essential:
    # - data: A contiguous block of memory that holds the elements of the matrix.
    # - shape: A tuple that specifies the dimensions of the matrix (rows, cols).
    # - strides: A tuple that specifies the number of bytes to step in each dimension.
    # The size attribute can be derived from the shape (size = rows * cols) and
    # is not necessary to store separately.
    #
    # About the "data" attribute:
    # We use a single list to store the elements of the matrix in a contiguous
    # block of memory. This is a "safe" way to manage memory in Mojo, as it
    # avoids the complexities of manual memory management while still providing
    # efficient access to the elements. It is also aligned with our philosophy
    # of "using safe Mojo as much as possible".
    # The disadvantage of this approach is that you cannot easily design a
    # shared-memory model where multiple matrices share the same underlying data
    # without defining different data types. In MatMojo, we have to define both
    # a "Matrix" type that owns its data and a "MatrixView" type that references
    # the data of another matrix. Thanks to the generic programming capabilities
    # of Mojo, we can still achieve a high level of code reuse between these
    # types.
    #
    # About the shape and strides of the matrix:
    # We use integers to store the shape and the strides of the matrix, which
    # is an efficient way to store the dimensions. For n-D arrays, we have to
    # use the list type to store the shape because the dimension is not fixed at
    # compile time. This also applies to the strides.
    #
    # CORE ATTRIBUTES
    var data: List[Self.ElementType]
    """The elements of the matrix stored in a contiguous block of memory."""
    var nrows: Int
    """The number of rows in the matrix."""
    var ncols: Int
    """The number of columns in the matrix."""
    var row_stride: Int
    """The stride (in number of elements) to move to the next row."""
    var col_stride: Int
    """The stride (in number of elements) to move to the next column."""

    # ===--------------------------------------------------------------------===#
    # Retrieve attributes
    # ===--------------------------------------------------------------------===#
    def get_data(self) -> Span[Self.ElementType, origin_of(self.data)]:
        """Returns the underlying data of the matrix."""
        return Span(self.data)

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
        return 0

    def get_size(self) -> Int:
        """Returns the total number of elements in the matrix."""
        return self.nrows * self.ncols

    def is_c_contiguous(self) -> Bool:
        """Returns True if the matrix is C-contiguous (row-major, dense)."""
        return self.col_stride == 1 and self.row_stride == self.ncols

    def is_f_contiguous(self) -> Bool:
        """Returns True if the matrix is F-contiguous (column-major, dense)."""
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
        var data: List[Self.ElementType],
        nrows: Int,
        ncols: Int,
        row_stride: Int,
        col_stride: Int,
    ):
        self.data = data^
        self.nrows = nrows
        self.ncols = ncols
        self.row_stride = row_stride
        self.col_stride = col_stride

    def __init__(
        out self,
        nrows: Int,
        ncols: Int,
        row_stride: Int,
        col_stride: Int,
    ):
        self.data = List[Self.ElementType](length=nrows * ncols, fill=0)
        self.nrows = nrows
        self.ncols = ncols
        self.row_stride = row_stride
        self.col_stride = col_stride

    def __init__(out self, *, copy: Self):
        """Initializes the matrix by copying another matrix."""
        self.data = copy.data.copy()
        self.nrows = copy.nrows
        self.ncols = copy.ncols
        self.row_stride = copy.row_stride
        self.col_stride = copy.col_stride

    def __init__(out self, *, deinit move: Self):
        """Initializes the matrix by moving another matrix."""
        self.data = move.data^
        self.nrows = move.nrows
        self.ncols = move.ncols
        self.row_stride = move.row_stride
        self.col_stride = move.col_stride

    # ===--------------------------------------------------------------------===#
    # Element Access and Mutation
    # View Access
    # ===--------------------------------------------------------------------===#

    # [Mojo Miji]
    # This method returns a reference to the element at the specified indices.
    # The mutability of the reference is determined by the mutability of the
    # underlying data (self.data). Since self.data is a mutable list, the
    # reference returned by __getitem__ is mutable, allowing for both reading
    # and writing to the matrix elements.
    # Thus, `__setitem__` is not needed as a separate method.
    def __getitem__(
        ref self, row: Int, col: Int
    ) raises -> ref[
        self.data[row * self.row_stride + col * self.col_stride]
    ] Self.ElementType:
        """Gets the element at the specified indices.

        Args:
            row: The row index.
            col: The column index.

        Raises:
            IndexError: If the indices are out of bounds.

        Returns:
            The element at the specified indices.
        """
        if row < 0 or row >= self.nrows or col < 0 or col >= self.ncols:
            raise IndexError(
                file="src/matmojo/types/matrix.mojo",
                function=(
                    "Matrix.__getitem__(self, row: Int, col: Int) ->"
                    " Self.ElementType"
                ),
                message="Index out of bounds.",
                previous_error=None,
            )
        return self.data[row * self.row_stride + col * self.col_stride]

    # [Mojo Miji]
    # When you pass `Self.dtype` and `origin_of(self)` as parameters to the
    # `MatrixView` type, you are creating a new, specific instantiation of the
    # generic `MatrixView` type that is tailored to the certain data type and
    # the origin of the current matrix instance.
    # In another word, if you have a matrix of type `int64` and is called `a`,
    # then this method will create a specific `MatrixView_int64_origin_a` type
    # at compile time, and then return an instance of this type.
    # Mojo compiler will ensure that `a` will not be destroyed as long as the
    # matrix view is still alive.
    # The approach of recording the origin, which is `a`, into the parameter of
    # the `MatrixView` type is called "parameterized origin".
    def __getitem__(
        self, x: Slice, y: Slice
    ) raises -> MatrixView[dtype=Self.dtype, origin=origin_of(self.data)]:
        """Gets a view of the specified row with a slice of columns."""
        return MatrixView(
            data=self.data,
            slice_x=x,
            slice_y=y,
            initial_nrows=self.nrows,
            initial_ncols=self.ncols,
            initial_row_stride=self.row_stride,
            initial_col_stride=self.col_stride,
            initial_offset=0,
        )

    def get_unsafe(self, row: Int, col: Int) -> Self.ElementType:
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
        var offset = get_offset(row, col, self.row_stride, self.col_stride)
        return self.data._data.unsafe_offset(offset)[]

    def view(self) -> MatrixView[Self.dtype, origin_of(self.data)]:
        """Gets a view of the entire matrix."""
        return MatrixView(
            data=Span(self.data),
            nrows=self.nrows,
            ncols=self.ncols,
            row_stride=self.row_stride,
            col_stride=self.col_stride,
            offset=0,
        )

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
                            get_offset(i, j, self.row_stride, self.col_stride)
                        ]
                    )
                    + "\t"
                )
            if i < self.nrows - 1:
                result += "\n"
        return result

    def write_to[W: Writer](self, mut writer: W):
        """Writes the matrix to a writer."""
        writer.write("Matrix, ")
        writer.write(self.dtype)
        writer.write(", ")
        writer.write(self.nrows)
        writer.write("x")
        writer.write(self.ncols)
        writer.write(", strides: ")
        writer.write(self.row_stride)
        writer.write("-")
        writer.write(self.col_stride)
        writer.write(":\n")
        for i in range(self.nrows):
            if i == 0:
                writer.write("[[\t")
            else:
                writer.write(" [\t")
            for j in range(self.ncols):
                writer.write(
                    self.data[
                        get_offset(i, j, self.row_stride, self.col_stride)
                    ]
                )
                writer.write("\t")
            writer.write("]")
            if i < self.nrows - 1:
                writer.write("\n")
            else:
                writer.write("]\n")

    # ===------------------------------------------------------------------ ===#
    # Basic math dunders
    # ===------------------------------------------------------------------ ===#

    def __add__(self, other: Self) raises -> Self:
        """Performs element-wise addition of two matrices."""
        return matmojo.routines.math.add(self, other)

    def __add__[
        origin: Origin
    ](self, other: MatrixView[Self.dtype, origin]) raises -> Self:
        """Performs element-wise addition of a matrix and a matrix view."""
        return matmojo.routines.math.add(self, other)

    def __sub__(self, other: Self) raises -> Self:
        """Performs element-wise subtraction of two matrices."""
        return matmojo.routines.math.sub(self, other)

    def __sub__[
        origin: Origin
    ](self, other: MatrixView[Self.dtype, origin]) raises -> Self:
        """Performs element-wise subtraction of a matrix and a matrix view."""
        return matmojo.routines.math.sub(self, other)

    def __mul__(self, other: Self) raises -> Self:
        """Performs element-wise multiplication of two matrices."""
        return matmojo.routines.math.mul(self, other)

    def __mul__[
        origin: Origin
    ](self, other: MatrixView[Self.dtype, origin]) raises -> Self:
        """Performs element-wise multiplication of a matrix and a matrix view.
        """
        return matmojo.routines.math.mul(self, other)

    def __truediv__(self, other: Self) raises -> Self:
        """Performs element-wise division of two matrices."""
        return matmojo.routines.math.div(self, other)

    def __truediv__[
        origin: Origin
    ](self, other: MatrixView[Self.dtype, origin]) raises -> Self:
        """Performs element-wise division of a matrix and a matrix view."""
        return matmojo.routines.math.div(self, other)

    def __matmul__(self, other: Self) raises -> Self:
        """Performs matrix multiplication of two matrices."""
        return matmojo.routines.math.matmul(self, other)

    def __matmul__[
        origin: Origin
    ](self, other: MatrixView[Self.dtype, origin]) raises -> Self:
        """Performs matrix multiplication of a matrix and a matrix view."""
        return matmojo.routines.math.matmul(self, other)
