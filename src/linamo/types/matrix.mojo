"""
This module defines the `Matrix` type, which is a dynamically sized 2D matrix.
"""

import std.math as builtin_math

from linamo.traits.matrix_like import MatrixLike
from linamo.types.errors import IndexError, ValueError
from linamo.types.matrix_iter import MatrixAxisIter
from linamo.types.matrix_view import MatrixView
import linamo.routines.math
import linamo.routines.logic
from linamo.utils.indexing import (
    get_offset,
    indices_within_bounds,
)


struct Matrix[dtype: DType](Copyable, MatrixLike, Movable, Sized, Writable):
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
    # without defining different data types. In Linamo, we have to define both
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
    #
    # [Mojo Miji]
    # The returned origin is `origin_of(self.data)` - the whole buffer - and
    # not the finer `self.data[row * row_stride + col * col_stride]`, which is
    # what this method used to name. A per-element origin sounds more precise,
    # and it is, but forming a second one invalidated the first, so
    #
    #     var s = a[0, 0] + a[1, 1]
    #
    # did not compile on a `var` matrix. Naming the buffer instead lets any
    # number of element references coexist. The reference has to be built from
    # a pointer because `self.data[i]` would re-derive the narrow origin.
    #
    # `__setitem__` is deliberately absent, and not only because the mutable
    # reference above makes it unnecessary. Defining `__setitem__` on this type
    # makes the compiler pass `self` to `__getitem__` as a temporary copy in
    # some positions, so a view sliced from it carries the origin of a dead
    # temporary and `a[0:1, :] - a[1:2, :]` stops compiling. Region writes are
    # spelled `assign(...)` from `routines.mutation` for the same reason.
    def __getitem__(
        ref self, row: Int, col: Int
    ) raises -> ref[origin_of(self.data)] Self.ElementType:
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
                function=(
                    "Matrix.__getitem__(self, row: Int, col: Int) ->"
                    " Self.ElementType"
                ),
                message="Index out of bounds.",
            )
        return self.data._data.unsafe_offset(
            row * self.row_stride + col * self.col_stride
        )[]

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
    # [Mojo Miji]
    # Note that `self` is taken by `read` and not by `ref` here. That single
    # word decides the mutability of every view slicing produces: with `ref`,
    # `origin_of(self.data)` would inherit the caller's mutability, so slicing
    # a `var` matrix would hand back a *mutable* view. Two mutable views of the
    # same matrix cannot be passed to one call, which would make the most
    # ordinary expressions in linear algebra illegal:
    #
    #     var d = a[0:1, :] - a[1:2, :]   # would not compile
    #
    # Reading two blocks of one matrix at the same time is always safe, so
    # slicing yields a read-only view and the expression above compiles. When
    # you do want to write through a sub-matrix, ask for it explicitly with
    # `view(x, y)`, which does inherit mutability.
    def __getitem__(
        self, x: Slice, y: Slice
    ) raises -> MatrixView[dtype=Self.dtype, origin=origin_of(self.data)]:
        """Gets a read-only view of the specified rows and columns."""
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

    # [Mojo Miji]
    # `read self`, not `ref self`. This method is the Matrix -> MatrixView
    # conversion the routine layer uses about forty times to feed kernels, and
    # every one of those uses is a read. Taking `ref self` made it hand back a
    # *mutable* view whenever the receiver was a `var`, which meant
    # `m.view() - m.view()` was rejected as two mutable borrows of one matrix,
    # and meant an innocuous-looking call was a write door. It is now a
    # shorthand for `m[:, :]` and nothing more.
    def view(self) -> MatrixView[Self.dtype, origin_of(self.data)]:
        """Gets a read-only view of the entire matrix.

        This is the same thing `m[:, :]` produces, and is the conversion the
        named routines use to accept a `Matrix` wherever a `MatrixView` is
        expected. To write through a sub-matrix, use `view_mut` from
        `linamo.routines.mutation`.
        """
        return MatrixView(
            data=Span(self.data),
            nrows=self.nrows,
            ncols=self.ncols,
            row_stride=self.row_stride,
            col_stride=self.col_stride,
            offset=0,
        )

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
    ](self) -> MatrixAxisIter[Self.dtype, origin_of(self.data), 0, forward]:
        """Iterates over the rows, yielding each as a `1 x ncols` view.

        Nothing is copied. The rows are read-only regardless of how the
        matrix was bound: iteration is an implicit path, and implicit paths do
        not grant write access. Use `rows_mut` from `linamo.routines.mutation`
        to walk a matrix writably.

        Parameters:
            forward: True for first-to-last, False for last-to-first.
        """
        return MatrixAxisIter[axis=0, forward=forward](self.view())

    def cols[
        forward: Bool = True
    ](self) -> MatrixAxisIter[Self.dtype, origin_of(self.data), 1, forward]:
        """Iterates over the columns, yielding each as an `nrows x 1` view.

        Parameters:
            forward: True for first-to-last, False for last-to-first.
        """
        return MatrixAxisIter[axis=1, forward=forward](self.view())

    def __iter__(
        self,
    ) -> MatrixAxisIter[Self.dtype, origin_of(self.data), 0, True]:
        """Iterates over the rows, so `for row in matrix` walks row views."""
        return self.rows()

    # Mojo 1.0's builtin `reversed()` only accepts specific stdlib containers,
    # so it will not route here. Call `matrix.__reversed__()` or the clearer
    # `matrix.rows[False]()` instead; this stays for when a protocol hook
    # lands.
    def __reversed__(
        self,
    ) -> MatrixAxisIter[Self.dtype, origin_of(self.data), 0, False]:
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
        load; otherwise it gathers element by element, so the call is always
        correct and only the speed changes.

        Parameters:
            width: How many elements to load.

        Args:
            row: The row to read from.
            col: The column at which the run starts.

        Raises:
            IndexError: If the run would leave the matrix.

        Returns:
            The `width` elements as a SIMD vector.
        """
        if row < 0 or row >= self.nrows or col < 0 or col + width > self.ncols:
            raise IndexError(
                function="Matrix.load[width](self, row: Int, col: Int)",
                message="SIMD load runs past the end of the matrix.",
            )
        var base = get_offset(row, col, self.row_stride, self.col_stride)
        if self.col_stride == 1:
            return (
                Span(self.data)
                .unsafe_ptr()
                .unsafe_offset(base)
                .unsafe_load[width=width]()
            )
        var result = SIMD[Self.dtype, width]()
        for i in range(width):
            result[i] = self.data[base + i * self.col_stride]
        return result

    def store[
        width: Int = 1
    ](mut self, row: Int, col: Int, value: SIMD[Self.dtype, width]) raises:
        """Stores `width` elements along row `row`, starting at column `col`.

        Unlike the view equivalent this can be a method, because a `Matrix`
        owns a concretely mutable `List` - there is no generic origin whose
        read-only instantiation the body would also have to satisfy.

        Parameters:
            width: How many elements to store.

        Args:
            row: The row to write to.
            col: The column at which the run starts.
            value: The elements to write.

        Raises:
            IndexError: If the run would leave the matrix.
        """
        if row < 0 or row >= self.nrows or col < 0 or col + width > self.ncols:
            raise IndexError(
                function="Matrix.store[width](mut self, row, col, value)",
                message="SIMD store runs past the end of the matrix.",
            )
        var base = get_offset(row, col, self.row_stride, self.col_stride)
        if self.col_stride == 1:
            Span(self.data).unsafe_ptr().unsafe_offset(base).unsafe_store(value)
        else:
            for i in range(width):
                self.data[base + i * self.col_stride] = value[i]

    # ===--------------------------------------------------------------------===#
    # Region assignment
    # ===--------------------------------------------------------------------===#

    # Spelled as named methods rather than `__setitem__`: Mojo 1.0 routes
    # `a[i:j, k:l] = rhs` through `__getitem__`, which would force `rhs` to be
    # a view with this matrix's own origin. See `routines/mutation.mojo`.
    def fill(
        mut self, rows: Slice, cols: Slice, value: Self.ElementType
    ) raises:
        """Writes one scalar into every element of the selected region.

        Args:
            rows: The rows to fill.
            cols: The columns to fill.
            value: The scalar written to every selected element.
        """
        var start_row, end_row, step_row = rows.indices(self.nrows)
        var start_col, end_col, step_col = cols.indices(self.ncols)
        for i in range(start_row, end_row, step_row):
            for j in range(start_col, end_col, step_col):
                self.data[
                    get_offset(i, j, self.row_stride, self.col_stride)
                ] = value

    def assign[
        mut_b: Bool, //, origin_b: Origin[mut=mut_b]
    ](
        mut self,
        rows: Slice,
        cols: Slice,
        src: MatrixView[Self.dtype, origin_b],
    ) raises:
        """Copies `src` into the region selected by `rows` and `cols`.

        Args:
            rows: The rows to assign into.
            cols: The columns to assign into.
            src: The source, which must match the target shape exactly.

        Raises:
            ValueError: If the shapes do not match.
        """
        var start_row, end_row, step_row = rows.indices(self.nrows)
        var start_col, end_col, step_col = cols.indices(self.ncols)
        var target_nrows = builtin_math.ceildiv(end_row - start_row, step_row)
        var target_ncols = builtin_math.ceildiv(end_col - start_col, step_col)
        if target_nrows != src.nrows or target_ncols != src.ncols:
            raise ValueError(
                function="Matrix.assign(mut self, rows, cols, src)",
                message="Shape mismatch in region assignment.",
            )
        for i in range(target_nrows):
            for j in range(target_ncols):
                self.data[
                    get_offset(
                        start_row + i * step_row,
                        start_col + j * step_col,
                        self.row_stride,
                        self.col_stride,
                    )
                ] = src[i, j]

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
        return linamo.routines.math.add(self, other)

    def __add__[
        origin: Origin
    ](self, other: MatrixView[Self.dtype, origin]) raises -> Self:
        """Performs element-wise addition of a matrix and a matrix view."""
        return linamo.routines.math.add(self, other)

    def __sub__(self, other: Self) raises -> Self:
        """Performs element-wise subtraction of two matrices."""
        return linamo.routines.math.sub(self, other)

    def __sub__[
        origin: Origin
    ](self, other: MatrixView[Self.dtype, origin]) raises -> Self:
        """Performs element-wise subtraction of a matrix and a matrix view."""
        return linamo.routines.math.sub(self, other)

    def __mul__(self, other: Self) raises -> Self:
        """Performs element-wise multiplication of two matrices."""
        return linamo.routines.math.mul(self, other)

    def __mul__[
        origin: Origin
    ](self, other: MatrixView[Self.dtype, origin]) raises -> Self:
        """Performs element-wise multiplication of a matrix and a matrix view.
        """
        return linamo.routines.math.mul(self, other)

    def __truediv__(self, other: Self) raises -> Self:
        """Performs element-wise division of two matrices."""
        return linamo.routines.math.div(self, other)

    def __truediv__[
        origin: Origin
    ](self, other: MatrixView[Self.dtype, origin]) raises -> Self:
        """Performs element-wise division of a matrix and a matrix view."""
        return linamo.routines.math.div(self, other)

    def __matmul__(self, other: Self) raises -> Self:
        """Performs matrix multiplication of two matrices."""
        return linamo.routines.math.matmul(self, other)

    def __matmul__[
        origin: Origin
    ](self, other: MatrixView[Self.dtype, origin]) raises -> Self:
        """Performs matrix multiplication of a matrix and a matrix view."""
        return linamo.routines.math.matmul(self, other)

    # ===------------------------------------------------------------------ ===#
    # Scalar operands for the arithmetic dunders
    # ===------------------------------------------------------------------ ===#
    # `A + 2.0` and friends. The matrix-matrix overloads live above; these add
    # the scalar right-hand side, and the reflected forms below cover the
    # `2.0 + A` direction.

    def __add__(self, other: Self.ElementType) -> Self:
        """Adds a scalar to every element."""
        return linamo.routines.math.scalar_add(self, other)

    def __sub__(self, other: Self.ElementType) -> Self:
        """Subtracts a scalar from every element."""
        return linamo.routines.math.scalar_sub(self, other)

    def __mul__(self, other: Self.ElementType) -> Self:
        """Multiplies every element by a scalar."""
        return linamo.routines.math.scalar_mul(self, other)

    def __truediv__(self, other: Self.ElementType) -> Self:
        """Divides every element by a scalar."""
        return linamo.routines.math.scalar_div(self, other)

    def __floordiv__(self, other: Self.ElementType) -> Self:
        """Floor-divides every element by a scalar."""
        return linamo.routines.math.scalar_floordiv(self, other)

    def __mod__(self, other: Self.ElementType) -> Self:
        """Takes every element modulo a scalar."""
        return linamo.routines.math.scalar_mod(self, other)

    def __pow__(self, other: Self.ElementType) -> Self:
        """Raises every element to a scalar power."""
        return linamo.routines.math.scalar_pow(self, other)

    # ===------------------------------------------------------------------ ===#
    # floordiv, mod, pow
    # ===------------------------------------------------------------------ ===#
    # `__pow__` is element-wise, matching NumPy's `**`. Matrix exponentiation
    # is a different operation and is spelled as a named routine, not `**`.

    def __floordiv__(self, other: Self) raises -> Self:
        """Performs element-wise floor division of two matrices."""
        return linamo.routines.math.floordiv(self, other)

    def __floordiv__[
        origin: Origin
    ](self, other: MatrixView[Self.dtype, origin]) raises -> Self:
        """Performs element-wise floor division of a matrix and a matrix view.
        """
        return linamo.routines.math.floordiv(self, other)

    def __mod__(self, other: Self) raises -> Self:
        """Performs element-wise modulo of two matrices."""
        return linamo.routines.math.mod(self, other)

    def __mod__[
        origin: Origin
    ](self, other: MatrixView[Self.dtype, origin]) raises -> Self:
        """Performs element-wise modulo of a matrix and a matrix view."""
        return linamo.routines.math.mod(self, other)

    def __pow__(self, other: Self) raises -> Self:
        """Performs element-wise exponentiation of two matrices."""
        return linamo.routines.math.pow(self, other)

    def __pow__[
        origin: Origin
    ](self, other: MatrixView[Self.dtype, origin]) raises -> Self:
        """Performs element-wise exponentiation of a matrix and a matrix view.
        """
        return linamo.routines.math.pow(self, other)

    # ===------------------------------------------------------------------ ===#
    # Reflected scalar operators
    # ===------------------------------------------------------------------ ===#
    # Reached when the scalar is on the left: `2.0 + A`. Addition and
    # multiplication commute, so they reuse the forward routine; subtraction
    # and division need the operands the other way round.
    #
    # `__rtruediv__` is not in the 5.2 table but is included anyway: without
    # it `2.0 - A` would work while `2.0 / A` silently failed to compile,
    # which is a worse API than either having all four or none.

    def __radd__(self, other: Self.ElementType) -> Self:
        """Adds every element of the matrix to a scalar (`scalar + mat`)."""
        return linamo.routines.math.scalar_add(self, other)

    def __rmul__(self, other: Self.ElementType) -> Self:
        """Multiplies a scalar by every element (`scalar * mat`)."""
        return linamo.routines.math.scalar_mul(self, other)

    def __rsub__(self, other: Self.ElementType) -> Self:
        """Subtracts every element of the matrix from a scalar (`scalar - mat`).
        """
        return linamo.routines.math.scalar_rsub(self, other)

    def __rtruediv__(self, other: Self.ElementType) -> Self:
        """Divides a scalar by every element of the matrix (`scalar / mat`)."""
        return linamo.routines.math.scalar_rdiv(self, other)

    # ===------------------------------------------------------------------ ===#
    # In-place operators
    # ===------------------------------------------------------------------ ===#
    # These write back through the matrix's own strides rather than allocating
    # a result, so a transposed or otherwise non-contiguous matrix keeps its
    # layout. There is no `MatrixView` counterpart: a view is generic over
    # `origin` and Mojo checks the body against the read-only instantiation
    # too, so no method writing through `self.data` can be defined on it. Use
    # the free functions in `routines/mutation.mojo` for views.
    #
    # Aliasing (`a += a[:, :]`) is rejected by the borrow checker, which will
    # not hand out a mutable reference to `a` while a view of it is live.

    def __iadd__(mut self, other: Self) raises:
        """In-place element-wise addition with another matrix."""
        linamo.routines.math._elementwise_inplace[
            func=Scalar[Self.dtype].__add__
        ](self, other.view())

    def __iadd__[
        origin: Origin
    ](mut self, other: MatrixView[Self.dtype, origin]) raises:
        """In-place element-wise addition with a matrix view."""
        linamo.routines.math._elementwise_inplace[
            func=Scalar[Self.dtype].__add__
        ](self, other)

    def __iadd__(mut self, other: Self.ElementType):
        """In-place element-wise addition with a scalar."""
        linamo.routines.math._scalar_elementwise_inplace[
            func=Scalar[Self.dtype].__add__
        ](self, other)

    def __isub__(mut self, other: Self) raises:
        """In-place element-wise subtraction with another matrix."""
        linamo.routines.math._elementwise_inplace[
            func=Scalar[Self.dtype].__sub__
        ](self, other.view())

    def __isub__[
        origin: Origin
    ](mut self, other: MatrixView[Self.dtype, origin]) raises:
        """In-place element-wise subtraction with a matrix view."""
        linamo.routines.math._elementwise_inplace[
            func=Scalar[Self.dtype].__sub__
        ](self, other)

    def __isub__(mut self, other: Self.ElementType):
        """In-place element-wise subtraction with a scalar."""
        linamo.routines.math._scalar_elementwise_inplace[
            func=Scalar[Self.dtype].__sub__
        ](self, other)

    def __imul__(mut self, other: Self) raises:
        """In-place element-wise multiplication with another matrix."""
        linamo.routines.math._elementwise_inplace[
            func=Scalar[Self.dtype].__mul__
        ](self, other.view())

    def __imul__[
        origin: Origin
    ](mut self, other: MatrixView[Self.dtype, origin]) raises:
        """In-place element-wise multiplication with a matrix view."""
        linamo.routines.math._elementwise_inplace[
            func=Scalar[Self.dtype].__mul__
        ](self, other)

    def __imul__(mut self, other: Self.ElementType):
        """In-place element-wise multiplication with a scalar."""
        linamo.routines.math._scalar_elementwise_inplace[
            func=Scalar[Self.dtype].__mul__
        ](self, other)

    def __itruediv__(mut self, other: Self) raises:
        """In-place element-wise division with another matrix."""
        linamo.routines.math._elementwise_inplace[
            func=Scalar[Self.dtype].__truediv__
        ](self, other.view())

    def __itruediv__[
        origin: Origin
    ](mut self, other: MatrixView[Self.dtype, origin]) raises:
        """In-place element-wise division with a matrix view."""
        linamo.routines.math._elementwise_inplace[
            func=Scalar[Self.dtype].__truediv__
        ](self, other)

    def __itruediv__(mut self, other: Self.ElementType):
        """In-place element-wise division with a scalar."""
        linamo.routines.math._scalar_elementwise_inplace[
            func=Scalar[Self.dtype].__truediv__
        ](self, other)

    def __ifloordiv__(mut self, other: Self) raises:
        """In-place element-wise floor division with another matrix."""
        linamo.routines.math._elementwise_inplace[
            func=Scalar[Self.dtype].__floordiv__
        ](self, other.view())

    def __ifloordiv__[
        origin: Origin
    ](mut self, other: MatrixView[Self.dtype, origin]) raises:
        """In-place element-wise floor division with a matrix view."""
        linamo.routines.math._elementwise_inplace[
            func=Scalar[Self.dtype].__floordiv__
        ](self, other)

    def __ifloordiv__(mut self, other: Self.ElementType):
        """In-place element-wise floor division with a scalar."""
        linamo.routines.math._scalar_elementwise_inplace[
            func=Scalar[Self.dtype].__floordiv__
        ](self, other)

    def __imod__(mut self, other: Self) raises:
        """In-place element-wise modulo with another matrix."""
        linamo.routines.math._elementwise_inplace[
            func=Scalar[Self.dtype].__mod__
        ](self, other.view())

    def __imod__[
        origin: Origin
    ](mut self, other: MatrixView[Self.dtype, origin]) raises:
        """In-place element-wise modulo with a matrix view."""
        linamo.routines.math._elementwise_inplace[
            func=Scalar[Self.dtype].__mod__
        ](self, other)

    def __imod__(mut self, other: Self.ElementType):
        """In-place element-wise modulo with a scalar."""
        linamo.routines.math._scalar_elementwise_inplace[
            func=Scalar[Self.dtype].__mod__
        ](self, other)

    # ===------------------------------------------------------------------ ===#
    # Comparison operators
    # ===------------------------------------------------------------------ ===#
    # These return an element-wise `Matrix[DType.bool]` mask, as in NumPy —
    # not a single `Bool`. `Matrix` therefore deliberately does not conform to
    # `EqualityComparable`; use `utils/test_utils.assert_matrices_equal` to ask
    # whether two matrices are wholly identical.

    def __lt__(self, other: Self) raises -> Matrix[DType.bool]:
        """Element-wise less-than comparison with another matrix."""
        return linamo.routines.logic.less(self, other)

    def __lt__[
        origin: Origin
    ](self, other: MatrixView[Self.dtype, origin]) raises -> Matrix[DType.bool]:
        """Element-wise less-than comparison with a matrix view."""
        return linamo.routines.logic.less(self, other)

    def __lt__(self, other: Self.ElementType) -> Matrix[DType.bool]:
        """Element-wise less-than comparison against a scalar."""
        return linamo.routines.logic.scalar_less(self, other)

    def __le__(self, other: Self) raises -> Matrix[DType.bool]:
        """Element-wise less-than-or-equal comparison with another matrix."""
        return linamo.routines.logic.less_equal(self, other)

    def __le__[
        origin: Origin
    ](self, other: MatrixView[Self.dtype, origin]) raises -> Matrix[DType.bool]:
        """Element-wise less-than-or-equal comparison with a matrix view."""
        return linamo.routines.logic.less_equal(self, other)

    def __le__(self, other: Self.ElementType) -> Matrix[DType.bool]:
        """Element-wise less-than-or-equal comparison against a scalar."""
        return linamo.routines.logic.scalar_less_equal(self, other)

    def __gt__(self, other: Self) raises -> Matrix[DType.bool]:
        """Element-wise greater-than comparison with another matrix."""
        return linamo.routines.logic.greater(self, other)

    def __gt__[
        origin: Origin
    ](self, other: MatrixView[Self.dtype, origin]) raises -> Matrix[DType.bool]:
        """Element-wise greater-than comparison with a matrix view."""
        return linamo.routines.logic.greater(self, other)

    def __gt__(self, other: Self.ElementType) -> Matrix[DType.bool]:
        """Element-wise greater-than comparison against a scalar."""
        return linamo.routines.logic.scalar_greater(self, other)

    def __ge__(self, other: Self) raises -> Matrix[DType.bool]:
        """Element-wise greater-than-or-equal comparison with another matrix."""
        return linamo.routines.logic.greater_equal(self, other)

    def __ge__[
        origin: Origin
    ](self, other: MatrixView[Self.dtype, origin]) raises -> Matrix[DType.bool]:
        """Element-wise greater-than-or-equal comparison with a matrix view."""
        return linamo.routines.logic.greater_equal(self, other)

    def __ge__(self, other: Self.ElementType) -> Matrix[DType.bool]:
        """Element-wise greater-than-or-equal comparison against a scalar."""
        return linamo.routines.logic.scalar_greater_equal(self, other)

    def __eq__(self, other: Self) raises -> Matrix[DType.bool]:
        """Element-wise equality comparison with another matrix."""
        return linamo.routines.logic.equal(self, other)

    def __eq__[
        origin: Origin
    ](self, other: MatrixView[Self.dtype, origin]) raises -> Matrix[DType.bool]:
        """Element-wise equality comparison with a matrix view."""
        return linamo.routines.logic.equal(self, other)

    def __eq__(self, other: Self.ElementType) -> Matrix[DType.bool]:
        """Element-wise equality comparison against a scalar."""
        return linamo.routines.logic.scalar_equal(self, other)

    def __ne__(self, other: Self) raises -> Matrix[DType.bool]:
        """Element-wise inequality comparison with another matrix."""
        return linamo.routines.logic.not_equal(self, other)

    def __ne__[
        origin: Origin
    ](self, other: MatrixView[Self.dtype, origin]) raises -> Matrix[DType.bool]:
        """Element-wise inequality comparison with a matrix view."""
        return linamo.routines.logic.not_equal(self, other)

    def __ne__(self, other: Self.ElementType) -> Matrix[DType.bool]:
        """Element-wise inequality comparison against a scalar."""
        return linamo.routines.logic.scalar_not_equal(self, other)
