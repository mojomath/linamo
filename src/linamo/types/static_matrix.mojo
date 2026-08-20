"""
This module defines the `StaticMatrix` type which is a statically sized 2D matrix.
"""

from linamo.utils.str import element_type_name
from linamo.utils.element import dtype_of
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


# [Mojo Miji]
# The parameter is an element type, spelled as on `Matrix` and `MatrixView`,
# even though this type can only ever hold a scalar: its buffer is a single
# SIMD register, so the dtype has to be known where the field is declared.
# `dtype_of` recovers it from `T` at compile time and rejects anything that is
# not a `Scalar[d]` --- which is the right answer rather than a limitation. A
# `StaticMatrix[BigInt]` would have no register to live in, and an
# arbitrary-precision element has nothing to gain from static shapes anyway;
# that is what `Matrix` is for.
struct StaticMatrix[T: Copyable & Deinitable, num_rows: Int, num_cols: Int](
    Copyable, Writable
):

    """A statically sized 2D matrix type.

    Parameters:
        T: The type of the matrix elements. Must be a scalar.
        num_rows: The number of rows in the matrix.
        num_cols: The number of columns in the matrix.
    """

    comptime BUFFER_ROW_LEN = next_power_of_two(Self.num_rows)
    comptime BUFFER_COL_LEN = next_power_of_two(Self.num_cols)
    comptime BUFFER_SIZE = Self.BUFFER_ROW_LEN * Self.BUFFER_COL_LEN
    comptime ROW_STRIDE = Self.BUFFER_COL_LEN
    comptime COL_STRIDE = 1

    comptime dtype = dtype_of[Self.T]()
    """The dtype behind `T`, recovered at compile time for the SIMD buffer."""
    comptime ElementType = Self.T
    """The type of the elements in the matrix. A second name for `T`."""

    var _data: SIMD[Self.dtype, Self.BUFFER_SIZE]
    """A SIMD array representing the data of the matrix."""

    # [Mojo Miji]
    # `Self.T` and `Scalar[Self.dtype]` are the same type --- that is what the
    # `where` clause on the struct says --- but `dtype_of` is an opaque
    # compile-time call, so the compiler will not equate them on its own. These
    # two are the only places that gap is bridged; everything else in the file,
    # and every caller outside it, works in whichever of the two names is
    # natural there.
    @always_inline
    def _flat(self, index: Int) -> Self.ElementType:
        """Reads one element out of the padded buffer.

        Args:
            index: The offset into the padded buffer.

        Returns:
            The element at that offset.
        """
        return rebind[Self.ElementType](self._data[index]).copy()

    @always_inline
    def _set_flat(mut self, index: Int, value: Self.ElementType):
        """Writes one element into the padded buffer.

        Args:
            index: The offset into the padded buffer.
            value: The element to write.
        """
        self._data[index] = rebind[Scalar[Self.dtype]](value)

    @always_inline
    def _as_simd(
        self,
    ) -> StaticMatrix[Scalar[Self.dtype], Self.num_rows, Self.num_cols]:
        """Returns `self` with its element type restated for the routines."""
        return rebind[
            StaticMatrix[Scalar[Self.dtype], Self.num_rows, Self.num_cols]
        ](self).copy()

    # ===--------------------------------------------------------------------===#
    # Retrieve attributes
    # ===--------------------------------------------------------------------===#
    # The shape lives in the parameters and the strides in `comptime` aliases,
    # so none of this is state and nothing can assign to it. The accessors are
    # spelled as methods anyway, and named as on `Matrix` and `MatrixView`, so
    # that a single piece of read-only code reads a shape the same way whichever
    # of the three it was handed --- which is what `traits/matrix_like.mojo`
    # would need. Each is `@always_inline` over a compile-time constant, so the
    # call costs nothing over naming the parameter directly.

    @always_inline
    def nrows(self) -> Int:
        """Returns the number of rows in the matrix."""
        return Self.num_rows

    @always_inline
    def ncols(self) -> Int:
        """Returns the number of columns in the matrix."""
        return Self.num_cols

    @always_inline
    def row_stride(self) -> Int:
        """Returns the row stride of the matrix.

        The buffer is padded to a power of two in each dimension, so this is
        `BUFFER_COL_LEN` rather than `num_cols`.
        """
        return Self.ROW_STRIDE

    @always_inline
    def col_stride(self) -> Int:
        """Returns the column stride of the matrix. Always 1."""
        return Self.COL_STRIDE

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
        return Self.num_rows * Self.num_cols

    def is_c_contiguous(self) -> Bool:
        """Returns True if the matrix is C-contiguous (row-major, dense).

        StaticMatrix uses power-of-2 padding for SIMD alignment, so it is
        only C-contiguous when `num_cols` equals the padded buffer column
        length.
        """
        return Self.ROW_STRIDE == Self.num_cols

    def is_f_contiguous(self) -> Bool:
        """Returns True if the matrix is F-contiguous (column-major, dense).

        StaticMatrix is always row-major with `COL_STRIDE == 1`, so it is never
        F-contiguous (unless it is a single row or column).
        """
        return Self.ROW_STRIDE == 1 and Self.COL_STRIDE == Self.num_rows

    def is_row_contiguous(self) -> Bool:
        """Returns True if elements within each row are contiguous.

        StaticMatrix always has `COL_STRIDE == 1`, so rows are always
        contiguous.
        """
        return True

    def is_col_contiguous(self) -> Bool:
        """Returns True if elements within each column are contiguous.

        StaticMatrix has `ROW_STRIDE == BUFFER_COL_LEN` (power-of-2 padded),
        which equals 1 only when `num_cols <= 1`.
        """
        return Self.ROW_STRIDE == 1

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
        return self._flat(row * Self.ROW_STRIDE + col * Self.COL_STRIDE)

    # ===--------------------------------------------------------------------===#
    # Conversion
    # ===--------------------------------------------------------------------===#

    def to_matrix(self) -> Matrix[Self.T]:
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
        # of this type: the shape lives in the parameters, so `a + b` on a
        # 2x2 and a 3x3 is a compile error, and a converting operand would
        # instead carry the mismatch into a dynamic kernel and raise at run
        # time. Naming the hop keeps the error where the compiler can see it.
        var buffer = List[Self.ElementType](
            capacity=Self.num_rows * Self.num_cols
        )
        for i in range(Self.num_rows):
            for j in range(Self.num_cols):
                buffer.append(self[i, j])
        return Matrix[Self.T](
            buffer^, Self.num_rows, Self.num_cols, Self.num_cols, 1
        )

    # ===--------------------------------------------------------------------===#
    # String Representation and Writing
    # ===--------------------------------------------------------------------===#

    def __str__(self) -> String:
        """Returns a string representation of the matrix."""
        var result = String("")
        for i in range(Self.num_rows):
            for j in range(Self.num_cols):
                result += (
                    String(
                        self._data[i * Self.ROW_STRIDE + j * Self.COL_STRIDE]
                    )
                    + "\t"
                )
            if i < Self.num_rows - 1:
                result += "\n"
        return result

    def write_to[W: Writer](self, mut writer: W):
        """Writes the matrix to a writer."""
        writer.write("StaticMatrix, ")
        writer.write(element_type_name[Self.ElementType]())
        writer.write(", ")
        writer.write(Self.num_rows)
        writer.write("x")
        writer.write(Self.num_cols)
        writer.write(":\n")
        for i in range(Self.num_rows):
            if i == 0:
                writer.write("[[\t")
            else:
                writer.write(" [\t")
            for j in range(Self.num_cols):
                writer.write(
                    round(
                        self._data[i * Self.ROW_STRIDE + j * Self.COL_STRIDE], 4
                    )
                )
                writer.write("\t")
            writer.write("]")
            if i < Self.num_rows - 1:
                writer.write("\n")
            else:
                writer.write("]")

    # ===------------------------------------------------------------------ ===#
    # Basic math dunders
    # ===------------------------------------------------------------------ ===#

    def __add__(
        self, other: Self
    ) -> StaticMatrix[Scalar[Self.dtype], Self.num_rows, Self.num_cols]:
        """Performs element-wise addition of two matrices."""
        return linamo.routines.math.add(self._as_simd(), other._as_simd())

    def __matmul__[
        other_num_cols: Int
    ](
        self, other: StaticMatrix[Self.T, Self.num_cols, other_num_cols]
    ) -> StaticMatrix[Scalar[Self.dtype], Self.num_rows, other_num_cols]:
        """Performs matrix multiplication of two matrices."""
        return linamo.routines.math.matmul(self._as_simd(), other._as_simd())
