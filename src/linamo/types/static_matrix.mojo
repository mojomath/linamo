"""
This module defines the `StaticMatrix` type which is a statically sized 2D matrix.
"""

from linamo.types.errors import IndexError, ValueError
from linamo.types.matrix import Matrix
from linamo.types.matrix_view import MatrixView
import linamo.routines.math
from linamo.utils.indexing import (
    indices_within_bounds,
)


def next_power_of_two(x: Int) -> Int:
    """Returns the next power of two greater than or equal to x."""
    if x <= 1:
        return 1
    var v = x - 1
    v |= v >> 1
    v |= v >> 2
    v |= v >> 4
    v |= v >> 8
    v |= v >> 16
    v |= v >> 32
    return v + 1


struct StaticMatrix[dtype: DType, nrows: Int, ncols: Int](Copyable, Writable):

    """A statically sized 2D matrix type.

    Parameters:
        dtype: The data type of the matrix elements.
        nrows: The number of rows in the matrix.
        ncols: The number of columns in the matrix.
    """

    comptime BUFFER_ROW_LEN = next_power_of_two(Self.nrows)
    comptime BUFFER_COL_LEN = next_power_of_two(Self.ncols)
    comptime BUFFER_SIZE = Self.BUFFER_ROW_LEN * Self.BUFFER_COL_LEN
    comptime row_stride = Self.BUFFER_COL_LEN
    comptime col_stride = 1

    comptime ElementType = Scalar[Self.dtype]
    """The type of the elements in the matrix, derived from the dtype."""

    var _data: SIMD[Self.dtype, Self.BUFFER_SIZE]
    """A SIMD array representing the data of the matrix."""

    # ===--------------------------------------------------------------------===#
    # Retrieve attributes
    # ===--------------------------------------------------------------------===#
    # `nrows` and `ncols` are struct parameters and the two strides are
    # `comptime` aliases, so `m.nrows` already reads as a compile-time constant
    # and nothing can assign to it. Only `_data` needs an accessor, which is
    # why this type carries fewer of them than `Matrix` and `MatrixView`.

    @always_inline
    def data(self) -> SIMD[Self.dtype, Self.BUFFER_SIZE]:
        """Returns the padded SIMD buffer backing the matrix."""
        return self._data

    @always_inline
    def offset(self) -> Int:
        """Returns the offset in the underlying data buffer for the matrix."""
        return 0

    @always_inline
    def size(self) -> Int:
        """Returns the total number of elements in the matrix."""
        return self.nrows * self.ncols

    def is_c_contiguous(self) -> Bool:
        """Returns True if the matrix is C-contiguous (row-major, dense).

        StaticMatrix uses power-of-2 padding for SIMD alignment, so it is
        only C-contiguous when ncols equals the padded buffer column length.
        """
        return self.row_stride == self.ncols

    def is_f_contiguous(self) -> Bool:
        """Returns True if the matrix is F-contiguous (column-major, dense).

        StaticMatrix is always row-major with col_stride=1, so it is never
        F-contiguous (unless it is a single row or column).
        """
        return self.row_stride == 1 and self.col_stride == self.nrows

    def is_row_contiguous(self) -> Bool:
        """Returns True if elements within each row are contiguous.

        StaticMatrix always has col_stride=1, so rows are always contiguous.
        """
        return True

    def is_col_contiguous(self) -> Bool:
        """Returns True if elements within each column are contiguous.

        StaticMatrix has row_stride = BUFFER_COL_LEN (power-of-2 padded),
        which equals 1 only when ncols <= 1.
        """
        return Self.row_stride == 1

    # ===--------------------------------------------------------------------===#
    # Life Cycle Management
    # ===--------------------------------------------------------------------===#

    def __init__(out self):
        """Initializes the matrix with all zeros."""
        # [Mojo Miji]
        # SIMD() initializes the buffer with zeros at compile time, so we don't
        # need to explicitly fill it with zeros here.
        self._data = SIMD[Self.dtype, Self.BUFFER_SIZE]()

    def __init__(out self, simd: SIMD[Self.dtype, Self.BUFFER_SIZE]):
        """Initializes the matrix with SIMD that match the size of buffer."""
        self._data = simd

    def __init__(out self, *, copy: Self):
        """Initializes the matrix by copying another matrix."""
        self._data = copy._data

    # ===--------------------------------------------------------------------===#
    # Element Access and Mutation
    # ===--------------------------------------------------------------------===#

    def __getitem__(self, row: Int, col: Int) -> Self.ElementType:
        """Accesses an element of the matrix view using row and column indices.
        """
        return self._data[row * self.row_stride + col * self.col_stride]

    # ===--------------------------------------------------------------------===#
    # Conversion
    # ===--------------------------------------------------------------------===#

    def to_matrix(self) -> Matrix[Self.dtype]:
        """Copies the matrix into a new owning, C-contiguous `Matrix`.

        This is the only bridge between `StaticMatrix` and the rest of the
        library, which works in `Matrix` and `MatrixView` throughout. The
        buffer is padded to a power of two in each dimension, so the copy
        walks the strides rather than taking the buffer wholesale; the result
        is dense.

        Returns:
            A new `Matrix` holding a dense copy of the elements.
        """
        # [Mojo Miji]
        # An `@implicit` constructor would not serve here. Mojo applies a
        # single implicit conversion and never chains two, so one landing on
        # `Matrix` would still not reach the `MatrixView` signatures the
        # operators and routines are written against. One reaching them
        # directly would cost the compile-time shape check that is the point
        # of this type: `nrows` and `ncols` are parameters, so `a + b` on a
        # 2x2 and a 3x3 is a compile error, and a converting operand would
        # instead carry the mismatch into a dynamic kernel and raise at run
        # time. Naming the hop keeps the error where the compiler can see it.
        var result = Matrix[Self.dtype](
            nrows=Self.nrows,
            ncols=Self.ncols,
            row_stride=Self.ncols,
            col_stride=1,
        )
        for i in range(Self.nrows):
            for j in range(Self.ncols):
                result._data[i * Self.ncols + j] = self[i, j]
        return result^

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
                        self._data[i * self.row_stride + j * self.col_stride]
                    )
                    + "\t"
                )
            if i < self.nrows - 1:
                result += "\n"
        return result

    def write_to[W: Writer](self, mut writer: W):
        """Writes the matrix to a writer."""
        writer.write("StaticMatrix, ")
        writer.write(self.dtype)
        writer.write(", ")
        writer.write(self.nrows)
        writer.write("x")
        writer.write(self.ncols)
        writer.write(":\n")
        for i in range(self.nrows):
            if i == 0:
                writer.write("[[\t")
            else:
                writer.write(" [\t")
            for j in range(self.ncols):
                writer.write(
                    round(
                        self._data[i * self.row_stride + j * self.col_stride], 4
                    )
                )
                writer.write("\t")
            writer.write("]")
            if i < self.nrows - 1:
                writer.write("\n")
            else:
                writer.write("]")

    # ===------------------------------------------------------------------ ===#
    # Basic math dunders
    # ===------------------------------------------------------------------ ===#

    def __add__(self, other: Self) -> Self:
        """Performs element-wise addition of two matrices."""
        return linamo.routines.math.add(self, other)

    def __matmul__[
        other_ncols: Int
    ](
        self, other: StaticMatrix[Self.dtype, Self.ncols, other_ncols]
    ) -> StaticMatrix[Self.dtype, Self.nrows, other_ncols]:
        """Performs matrix multiplication of two matrices."""
        return linamo.routines.math.matmul(self, other)
