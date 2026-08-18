"""
This module defines the `MatrixView` type, which is a view on a `Matrix`.
"""

import std.math as builtin_math
import linamo.routines.math
import linamo.routines.logic
import linamo.routines.manipulation

from linamo.traits.matrix_like import MatrixLike
from linamo.types.errors import IndexError
from linamo.types.matrix import Matrix
from linamo.types.matrix_iter import MatrixAxisIter
from linamo.utils.indexing import get_offset, indices_within_bounds
from std.memory import Pointer


struct MatrixView[mut: Bool, //, dtype: DType, origin: Origin[mut=mut]](
    ImplicitlyCopyable, MatrixLike, Movable, Sized, Writable
):
    """A 2D matrix view type that references another Matrix.

    Parameters:
        mut: Whether the reference to the matrix is mutable.
        dtype: The data type of the matrix elements.
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

    # [Mojo Miji]
    # This is what lets one signature stand in for four. A routine declared as
    # `def add(a: MatrixView[dtype, oa], b: MatrixView[dtype, ob])` now accepts
    # a `Matrix` in either position, because the compiler inserts this
    # conversion. See the note in `linamo.routines.math`.
    #
    # Two details make it work. The argument is `ref m`: only `ref` binds the
    # origin to the caller's storage. Under `imm`, `read` or the default
    # convention, `origin_of(m.data)` names the callee's own parameter slot, so
    # the conversion is one no caller can ever satisfy and every call site
    # fails to compile. And the result is wrapped in `ImmOrigin(...)`, so a
    # `var` matrix yields a
    # *read-only* view: without that, `add(a, a)` would be two mutable borrows
    # of one matrix and would not compile, which is the same wall 5.2 hit.
    @implicit
    def __init__[
        d: DType
    ](out self: MatrixView[d, ImmOrigin(origin_of(m.data))], ref m: Matrix[d]):
        """Converts a `Matrix` into a read-only view of the whole matrix.

        Parameters:
            d: The data type of the matrix elements.

        Args:
            m: The matrix to view.
        """
        self.data = Span(m.data).as_imm()
        self.nrows = m.nrows
        self.ncols = m.ncols
        self.row_stride = m.row_stride
        self.col_stride = m.col_stride
        self.offset = 0

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
        # `ceildiv` alone goes negative when the slice selects nothing - the
        # `3:1` and `1:4:-1` forms both give a negative count - and a view with
        # `nrows = -2` reports `len(v) == -2` and silently disagrees with every
        # loop written as `range(start, end, step)`. Python calls these empty,
        # so clamp. Genuine negative steps are untouched: `4:0:-1` is
        # `ceildiv(-4, -1) == 4`.
        self.nrows = max(0, builtin_math.ceildiv(end_x - start_x, step_x))
        self.ncols = max(0, builtin_math.ceildiv(end_y - start_y, step_y))
        self.row_stride = initial_row_stride * step_x
        self.col_stride = initial_col_stride * step_y

    # ===--------------------------------------------------------------------===#
    # Element Access and Mutation
    # ===--------------------------------------------------------------------===#

    def __getitem__(
        self, row: Int, col: Int
    ) -> ref[Self.origin] Self.ElementType:
        """Accesses an element of the matrix view using row and column indices.
        """
        var index = self.offset + row * self.row_stride + col * self.col_stride
        return self.data[index]

    # [Mojo Miji]
    # `ImmOrigin(Self.origin)` demotes the origin to a read-only one, so a view
    # of a view is always read-only even when the parent view is mutable. This
    # matches `Matrix.__getitem__` and for the same reason: two mutable views
    # of one matrix cannot both be passed to a single call, which would make
    # `v[0:1, :] - v[1:2, :]` illegal. Use `routines.mutation` on the parent
    # view when you need to write.
    def __getitem__(
        self, rows: Slice, cols: Slice
    ) raises -> MatrixView[Self.dtype, ImmOrigin(Self.origin)]:
        """Gets a read-only view of the specified rows and columns."""
        return MatrixView[Self.dtype, ImmOrigin(Self.origin)](
            data=self.data.as_imm(),
            slice_x=rows,
            slice_y=cols,
            initial_nrows=self.nrows,
            initial_ncols=self.ncols,
            initial_row_stride=self.row_stride,
            initial_col_stride=self.col_stride,
            initial_offset=self.offset,
        )

    # [Mojo Miji]
    # The mirror of `Span.as_imm()`. A mutable view is an exclusive borrow, so
    # it cannot appear twice in one expression; demoting it to a read-only view
    # lifts that restriction, exactly as `&mut T` to `&T` does in Rust. There
    # is no inverse: nothing in the library promotes a read-only view back to a
    # mutable one.
    def as_imm(self) -> MatrixView[Self.dtype, ImmOrigin(Self.origin)]:
        """Returns a read-only view over the same elements.

        Returns:
            A view with the same shape, strides and offset, but a read-only
            origin, so that it may be combined with other views of the same
            matrix.
        """
        return MatrixView[Self.dtype, ImmOrigin(Self.origin)](
            data=self.data.as_imm(),
            nrows=self.nrows,
            ncols=self.ncols,
            row_stride=self.row_stride,
            col_stride=self.col_stride,
            offset=self.offset,
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
    # Length and iteration
    # ===--------------------------------------------------------------------===#

    def __len__(self) -> Int:
        """Returns the number of rows.

        This is the row count rather than the element count so that `len()`
        agrees with what `__iter__` yields, the way it does for any Python
        sequence. Use `get_size()` for `nrows * ncols`.
        """
        return self.nrows

    def rows[
        forward: Bool = True
    ](self) -> MatrixAxisIter[Self.dtype, Self.origin, 0, forward]:
        """Iterates over the rows, yielding each as a `1 x ncols` view.

        Nothing is copied: each row borrows the parent buffer and inherits its
        mutability, so rows of a mutable view can be written through.

        Parameters:
            forward: True for first-to-last, False for last-to-first.
        """
        return MatrixAxisIter[axis=0, forward=forward](self)

    def cols[
        forward: Bool = True
    ](self) -> MatrixAxisIter[Self.dtype, Self.origin, 1, forward]:
        """Iterates over the columns, yielding each as an `nrows x 1` view.

        Parameters:
            forward: True for first-to-last, False for last-to-first.
        """
        return MatrixAxisIter[axis=1, forward=forward](self)

    def __iter__(self) -> MatrixAxisIter[Self.dtype, Self.origin, 0, True]:
        """Iterates over the rows, so `for row in view` walks row views."""
        return self.rows()

    # Mojo 1.0's builtin `reversed()` only accepts specific stdlib containers,
    # so it will not route here. Call `view.__reversed__()` or the clearer
    # `view.rows[False]()` instead; this stays for when a protocol hook lands.
    def __reversed__(
        self,
    ) -> MatrixAxisIter[Self.dtype, Self.origin, 0, False]:
        """Iterates over the rows from last to first."""
        return self.rows[False]()

    # ===--------------------------------------------------------------------===#
    # SIMD access
    # ===--------------------------------------------------------------------===#

    def load[
        width: Int = 1
    ](self, row: Int, col: Int) raises -> SIMD[Self.dtype, width]:
        """Loads `width` elements along row `row`, starting at column `col`.

        When the row is contiguous (`col_stride == 1`) this is a single vector
        load. Otherwise - which is what slicing with a step produces - it
        falls back to gathering element by element, so the call is always
        correct and only the speed changes.

        Parameters:
            width: How many elements to load.

        Args:
            row: The row to read from.
            col: The column at which the run starts.

        Raises:
            IndexError: If the run would leave the view.

        Returns:
            The `width` elements as a SIMD vector.
        """
        if row < 0 or row >= self.nrows or col < 0 or col + width > self.ncols:
            raise IndexError(
                function="MatrixView.load[width](self, row: Int, col: Int)",
                message="SIMD load runs past the end of the view.",
            )
        var base = get_offset(
            row, col, self.row_stride, self.col_stride, self.offset
        )
        if self.col_stride == 1:
            return (
                self.data.unsafe_ptr()
                .unsafe_offset(base)
                .unsafe_load[width=width]()
            )
        var result = SIMD[Self.dtype, width]()
        for i in range(width):
            result[i] = self.data[base + i * self.col_stride]
        return result

    # ===--------------------------------------------------------------------===#
    # Materialisation
    # ===--------------------------------------------------------------------===#

    def to_matrix(self) raises -> Matrix[Self.dtype]:
        """Copies the view into a new owning, C-contiguous `Matrix`.

        This is the one deliberate allocation in the view API. `copy()` returns
        another view of the same data, which is an O(1) handle copy; this walks
        the (possibly strided) view and produces dense owned storage.

        Returns:
            A new `Matrix` holding a dense copy of the viewed elements.
        """
        var result = Matrix[Self.dtype](
            nrows=self.nrows,
            ncols=self.ncols,
            row_stride=self.ncols,
            col_stride=1,
        )
        for i in range(self.nrows):
            for j in range(self.ncols):
                result.data[i * self.ncols + j] = self[i, j]
        return result^

    def astype[target: DType](self) raises -> Matrix[target]:
        """Returns a C-contiguous copy of this view cast to `target`.

        Like `to_matrix()`, this materialises: the result owns its elements.

        Parameters:
            target: The data type of the result elements.

        Returns:
            A new `Matrix[target]` with the same shape.
        """
        return linamo.routines.manipulation.astype[target](self)

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
        return linamo.routines.math.add(self, other)

    def __add__(self, other: Matrix[Self.dtype]) raises -> Matrix[Self.dtype]:
        """Performs element-wise addition of a matrix view and a matrix."""
        return linamo.routines.math.add(self, other)

    def __sub__[
        origin_b: Origin
    ](self, other: MatrixView[Self.dtype, origin_b]) raises -> Matrix[
        Self.dtype
    ]:
        """Performs element-wise subtraction of two matrix views."""
        return linamo.routines.math.sub(self, other)

    def __sub__(self, other: Matrix[Self.dtype]) raises -> Matrix[Self.dtype]:
        """Performs element-wise subtraction of a matrix view and a matrix."""
        return linamo.routines.math.sub(self, other)

    def __mul__[
        origin_b: Origin
    ](self, other: MatrixView[Self.dtype, origin_b]) raises -> Matrix[
        Self.dtype
    ]:
        """Performs element-wise multiplication of two matrix views."""
        return linamo.routines.math.mul(self, other)

    def __mul__(self, other: Matrix[Self.dtype]) raises -> Matrix[Self.dtype]:
        """Performs element-wise multiplication of a matrix view and a matrix.
        """
        return linamo.routines.math.mul(self, other)

    def __truediv__[
        origin_b: Origin
    ](self, other: MatrixView[Self.dtype, origin_b]) raises -> Matrix[
        Self.dtype
    ]:
        """Performs element-wise division of two matrix views."""
        return linamo.routines.math.div(self, other)

    def __truediv__(
        self, other: Matrix[Self.dtype]
    ) raises -> Matrix[Self.dtype]:
        """Performs element-wise division of a matrix view and a matrix."""
        return linamo.routines.math.div(self, other)

    def __matmul__[
        origin_b: Origin
    ](self, other: MatrixView[Self.dtype, origin_b]) raises -> Matrix[
        Self.dtype
    ]:
        """Performs matrix multiplication of two matrix views."""
        return linamo.routines.math.matmul(self, other)

    def __matmul__(
        self, other: Matrix[Self.dtype]
    ) raises -> Matrix[Self.dtype]:
        """Performs matrix multiplication of a matrix view and a matrix."""
        return linamo.routines.math.matmul(self, other)

    # ===--------------------------------------------------------------------===#
    # Scalar operands for the arithmetic dunders
    # ===--------------------------------------------------------------------===#

    def __add__(self, other: Scalar[Self.dtype]) -> Matrix[Self.dtype]:
        """Adds a scalar to every element of the view."""
        return linamo.routines.math.scalar_add(self, other)

    def __sub__(self, other: Scalar[Self.dtype]) -> Matrix[Self.dtype]:
        """Subtracts a scalar from every element of the view."""
        return linamo.routines.math.scalar_sub(self, other)

    def __mul__(self, other: Scalar[Self.dtype]) -> Matrix[Self.dtype]:
        """Multiplies every element by a scalar of the view."""
        return linamo.routines.math.scalar_mul(self, other)

    def __truediv__(self, other: Scalar[Self.dtype]) -> Matrix[Self.dtype]:
        """Divides every element by a scalar of the view."""
        return linamo.routines.math.scalar_div(self, other)

    def __floordiv__(self, other: Scalar[Self.dtype]) -> Matrix[Self.dtype]:
        """Floor-divides every element by a scalar of the view."""
        return linamo.routines.math.scalar_floordiv(self, other)

    def __mod__(self, other: Scalar[Self.dtype]) -> Matrix[Self.dtype]:
        """Takes every element modulo a scalar of the view."""
        return linamo.routines.math.scalar_mod(self, other)

    def __pow__(self, other: Scalar[Self.dtype]) -> Matrix[Self.dtype]:
        """Raises every element to a scalar power of the view."""
        return linamo.routines.math.scalar_pow(self, other)

    # ===--------------------------------------------------------------------===#
    # floordiv, mod, pow
    # ===--------------------------------------------------------------------===#
    # `__pow__` is element-wise, matching NumPy's `**`.

    def __floordiv__[
        origin_b: Origin
    ](self, other: MatrixView[Self.dtype, origin_b]) raises -> Matrix[
        Self.dtype
    ]:
        """Performs element-wise floor division of two matrix views."""
        return linamo.routines.math.floordiv(self, other)

    def __floordiv__(
        self, other: Matrix[Self.dtype]
    ) raises -> Matrix[Self.dtype]:
        """Performs element-wise floor division of a matrix view and a matrix.
        """
        return linamo.routines.math.floordiv(self, other)

    def __mod__[
        origin_b: Origin
    ](self, other: MatrixView[Self.dtype, origin_b]) raises -> Matrix[
        Self.dtype
    ]:
        """Performs element-wise modulo of two matrix views."""
        return linamo.routines.math.mod(self, other)

    def __mod__(self, other: Matrix[Self.dtype]) raises -> Matrix[Self.dtype]:
        """Performs element-wise modulo of a matrix view and a matrix."""
        return linamo.routines.math.mod(self, other)

    def __pow__[
        origin_b: Origin
    ](self, other: MatrixView[Self.dtype, origin_b]) raises -> Matrix[
        Self.dtype
    ]:
        """Performs element-wise exponentiation of two matrix views."""
        return linamo.routines.math.pow(self, other)

    def __pow__(self, other: Matrix[Self.dtype]) raises -> Matrix[Self.dtype]:
        """Performs element-wise exponentiation of a matrix view and a matrix.
        """
        return linamo.routines.math.pow(self, other)

    # ===--------------------------------------------------------------------===#
    # Reflected scalar operators
    # ===--------------------------------------------------------------------===#
    # There are no in-place counterparts (`+=`, `-=`, ...) on a view: the type
    # is generic over `origin` and Mojo checks a method body against the
    # read-only instantiation as well, so nothing writing through `self.data`
    # can be defined here. See `routines/mutation.mojo`.

    def __radd__(self, other: Scalar[Self.dtype]) -> Matrix[Self.dtype]:
        """Adds every element of the view to a scalar (`scalar + view`)."""
        return linamo.routines.math.scalar_add(self, other)

    def __rmul__(self, other: Scalar[Self.dtype]) -> Matrix[Self.dtype]:
        """Multiplies a scalar by every element (`scalar * view`)."""
        return linamo.routines.math.scalar_mul(self, other)

    def __rsub__(self, other: Scalar[Self.dtype]) -> Matrix[Self.dtype]:
        """Subtracts every element of the view from a scalar (`scalar - view`).
        """
        return linamo.routines.math.scalar_rsub(self, other)

    def __rtruediv__(self, other: Scalar[Self.dtype]) -> Matrix[Self.dtype]:
        """Divides a scalar by every element of the view (`scalar / view`)."""
        return linamo.routines.math.scalar_rdiv(self, other)

    # ===--------------------------------------------------------------------===#
    # Comparison operators
    # ===--------------------------------------------------------------------===#
    # Element-wise `Matrix[DType.bool]` masks, as on `Matrix`.

    def __lt__[
        origin_b: Origin
    ](self, other: MatrixView[Self.dtype, origin_b]) raises -> Matrix[
        DType.bool
    ]:
        """Element-wise less-than comparison with another matrix view."""
        return linamo.routines.logic.less(self, other)

    def __lt__(self, other: Matrix[Self.dtype]) raises -> Matrix[DType.bool]:
        """Element-wise less-than comparison with a matrix."""
        return linamo.routines.logic.less(self, other)

    def __lt__(self, other: Scalar[Self.dtype]) -> Matrix[DType.bool]:
        """Element-wise less-than comparison against a scalar."""
        return linamo.routines.logic.scalar_less(self, other)

    def __le__[
        origin_b: Origin
    ](self, other: MatrixView[Self.dtype, origin_b]) raises -> Matrix[
        DType.bool
    ]:
        """Element-wise less-than-or-equal comparison with another matrix view.
        """
        return linamo.routines.logic.less_equal(self, other)

    def __le__(self, other: Matrix[Self.dtype]) raises -> Matrix[DType.bool]:
        """Element-wise less-than-or-equal comparison with a matrix."""
        return linamo.routines.logic.less_equal(self, other)

    def __le__(self, other: Scalar[Self.dtype]) -> Matrix[DType.bool]:
        """Element-wise less-than-or-equal comparison against a scalar."""
        return linamo.routines.logic.scalar_less_equal(self, other)

    def __gt__[
        origin_b: Origin
    ](self, other: MatrixView[Self.dtype, origin_b]) raises -> Matrix[
        DType.bool
    ]:
        """Element-wise greater-than comparison with another matrix view."""
        return linamo.routines.logic.greater(self, other)

    def __gt__(self, other: Matrix[Self.dtype]) raises -> Matrix[DType.bool]:
        """Element-wise greater-than comparison with a matrix."""
        return linamo.routines.logic.greater(self, other)

    def __gt__(self, other: Scalar[Self.dtype]) -> Matrix[DType.bool]:
        """Element-wise greater-than comparison against a scalar."""
        return linamo.routines.logic.scalar_greater(self, other)

    def __ge__[
        origin_b: Origin
    ](self, other: MatrixView[Self.dtype, origin_b]) raises -> Matrix[
        DType.bool
    ]:
        """Element-wise greater-than-or-equal comparison with another matrix view.
        """
        return linamo.routines.logic.greater_equal(self, other)

    def __ge__(self, other: Matrix[Self.dtype]) raises -> Matrix[DType.bool]:
        """Element-wise greater-than-or-equal comparison with a matrix."""
        return linamo.routines.logic.greater_equal(self, other)

    def __ge__(self, other: Scalar[Self.dtype]) -> Matrix[DType.bool]:
        """Element-wise greater-than-or-equal comparison against a scalar."""
        return linamo.routines.logic.scalar_greater_equal(self, other)

    def __eq__[
        origin_b: Origin
    ](self, other: MatrixView[Self.dtype, origin_b]) raises -> Matrix[
        DType.bool
    ]:
        """Element-wise equality comparison with another matrix view."""
        return linamo.routines.logic.equal(self, other)

    def __eq__(self, other: Matrix[Self.dtype]) raises -> Matrix[DType.bool]:
        """Element-wise equality comparison with a matrix."""
        return linamo.routines.logic.equal(self, other)

    def __eq__(self, other: Scalar[Self.dtype]) -> Matrix[DType.bool]:
        """Element-wise equality comparison against a scalar."""
        return linamo.routines.logic.scalar_equal(self, other)

    def __ne__[
        origin_b: Origin
    ](self, other: MatrixView[Self.dtype, origin_b]) raises -> Matrix[
        DType.bool
    ]:
        """Element-wise inequality comparison with another matrix view."""
        return linamo.routines.logic.not_equal(self, other)

    def __ne__(self, other: Matrix[Self.dtype]) raises -> Matrix[DType.bool]:
        """Element-wise inequality comparison with a matrix."""
        return linamo.routines.logic.not_equal(self, other)

    def __ne__(self, other: Scalar[Self.dtype]) -> Matrix[DType.bool]:
        """Element-wise inequality comparison against a scalar."""
        return linamo.routines.logic.scalar_not_equal(self, other)
