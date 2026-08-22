"""
This module defines the `MatrixView` type, which is a view on a `Matrix`.
"""

import std.math as builtin_math

from linamo.errors import IndexError
import linamo.routines.linalg
import linamo.routines.math
import linamo.routines.logic
import linamo.routines.manipulation

from decimo import Numeric

from linamo.types.matrix import Matrix
from linamo.types.matrix_iter import MatrixAxisIter
from linamo.utils.indexing import get_offset, indices_within_bounds
from linamo.utils.str import element_type_name
from linamo.utils.formatting import (
    ELISION,
    elides,
    gap_position,
    plan_indices,
    trim_fraction,
    write_grid,
    write_header,
)
from std.memory import Pointer


struct MatrixView[
    mut: Bool, //, T: Copyable & Deinitable, origin: Origin[mut=mut]
](
    ImplicitlyCopyable,
    Movable,
    Sized,
    Writable where conforms_to(T, Writable),
):
    """A 2D matrix view type that references another Matrix.

    Parameters:
        mut: Whether the reference to the matrix is mutable.
        T: The type of the matrix elements.
        origin: The origin of the matrix.
    """

    comptime ElementType = Self.T
    """The type of the elements in the matrix view. A second name for `T`,
    used in signatures where `T` alone would read as a stray letter."""

    var _data: Span[Self.ElementType, Self.origin]
    """A span representing the data of the matrix view."""
    var _nrows: Int
    """The number of rows in the matrix view."""
    var _ncols: Int
    """The number of columns in the matrix view."""
    var _row_stride: Int
    """The row stride of the matrix view."""
    var _col_stride: Int
    """The column stride of the matrix view."""
    var _offset: Int
    """The offset in the base matrix data where the view starts."""

    # ===--------------------------------------------------------------------===#
    # Retrieve attributes
    # ===--------------------------------------------------------------------===#

    @always_inline
    def data(self) -> Span[Self.ElementType, Self.origin]:
        """Returns the underlying data of the matrix."""
        return self._data

    @always_inline
    def nrows(self) -> Int:
        """Returns the number of rows in the matrix."""
        return self._nrows

    @always_inline
    def ncols(self) -> Int:
        """Returns the number of columns in the matrix."""
        return self._ncols

    @always_inline
    def row_stride(self) -> Int:
        """Returns the row stride of the matrix."""
        return self._row_stride

    @always_inline
    def col_stride(self) -> Int:
        """Returns the column stride of the matrix."""
        return self._col_stride

    @always_inline
    def offset(self) -> Int:
        """Returns the offset in the underlying data buffer for the matrix."""
        return self._offset

    @always_inline
    def size(self) -> Int:
        """Returns the total number of elements in the matrix."""
        return self._nrows * self._ncols

    def is_c_contiguous(self) -> Bool:
        """Returns True if the view is C-contiguous (row-major, dense)."""
        return self._col_stride == 1 and self._row_stride == self._ncols

    def is_f_contiguous(self) -> Bool:
        """Returns True if the view is F-contiguous (column-major, dense)."""
        return self._row_stride == 1 and self._col_stride == self._nrows

    def is_row_contiguous(self) -> Bool:
        """Returns True if elements within each row are contiguous (col_stride == 1).

        Allows padding between rows (row_stride >= ncols).
        """
        return self._col_stride == 1

    def is_col_contiguous(self) -> Bool:
        """Returns True if elements within each column are contiguous (row_stride == 1).

        Allows padding between columns (col_stride >= nrows).
        """
        return self._row_stride == 1

    # ===--------------------------------------------------------------------===#
    # Life Cycle Management
    # ===--------------------------------------------------------------------===#

    def __init__(
        out self,
        buffer: Span[Self.ElementType, Self.origin],
        *,
        nrows: Int,
        ncols: Int,
        row_stride: Int,
        col_stride: Int,
        offset: Int,
    ):
        """Initializes a MatrixView instance that references a Matrix.

        Args:
            buffer: A span representing the matrix data.
            nrows: The number of rows in the view.
            ncols: The number of columns in the view.
            row_stride: The row stride for accessing elements.
            col_stride: The column stride for accessing elements.
            offset: The starting offset in the matrix data.
        """
        self._data = buffer
        self._nrows = nrows
        self._ncols = ncols
        self._row_stride = row_stride
        self._col_stride = col_stride
        self._offset = offset

    # [Mojo Miji]
    # This is what lets one signature stand in for four. A parameter declared
    # as `MatrixView[T, o]` accepts a `Matrix` too, because the compiler
    # inserts this conversion. It applies to operator dunders as much as to
    # plain routines: `A + B`, `A + V`, `V + B` and `V + V` all resolve to the
    # single view-operand overload.
    #
    # Two details make it work. The argument is `ref m`: only `ref` binds the
    # origin to the caller's storage. Under `imm`, `read` or the default
    # convention, `origin_of(m._data)` names the callee's own parameter slot, so
    # the conversion is one no caller can satisfy and every call site fails to
    # compile. And the result is wrapped in `ImmOrigin(...)`, so a `var` matrix
    # yields a *read-only* view; without that, `add(a, a)` would be two mutable
    # borrows of one matrix and would not compile.
    @implicit
    def __init__[
        E: Copyable & Deinitable, //
    ](out self: MatrixView[E, ImmOrigin(origin_of(m._data))], ref m: Matrix[E]):
        """Converts a `Matrix` into a read-only view of the whole matrix.

        Parameters:
            E: The type of the matrix elements.

        Args:
            m: The matrix to view.
        """
        self._data = Span(m._data).as_imm()
        self._nrows = m.nrows()
        self._ncols = m.ncols()
        self._row_stride = m.row_stride()
        self._col_stride = m.col_stride()
        self._offset = 0

    def __init__(
        out self,
        buffer: Span[Self.ElementType, Self.origin],
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
        self._data = buffer
        var start_x, end_x, step_x = slice_x.indices(initial_nrows)
        var start_y, end_y, step_y = slice_y.indices(initial_ncols)
        self._offset = (
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
        self._nrows = max(0, builtin_math.ceildiv(end_x - start_x, step_x))
        self._ncols = max(0, builtin_math.ceildiv(end_y - start_y, step_y))
        self._row_stride = initial_row_stride * step_x
        self._col_stride = initial_col_stride * step_y

    # ===--------------------------------------------------------------------===#
    # Element Access and Mutation
    # ===--------------------------------------------------------------------===#

    def __getitem__(
        self, row: Int, col: Int
    ) -> ref[Self.origin] Self.ElementType:
        """Accesses an element of the matrix view using row and column indices.
        """
        var index = (
            self._offset + row * self._row_stride + col * self._col_stride
        )
        return self._data[index]

    # [Mojo Miji]
    # `ImmOrigin(Self.origin)` demotes the origin to a read-only one, so a view
    # of a view is always read-only even when the parent view is mutable. This
    # matches `Matrix.__getitem__` and for the same reason: two mutable views
    # of one matrix cannot both be passed to a single call, which would make
    # `v[0:1, :] - v[1:2, :]` illegal. Use `routines.mutation` on the parent
    # view when you need to write.
    def __getitem__(
        self, rows: Slice, cols: Slice
    ) raises -> MatrixView[Self.T, ImmOrigin(Self.origin)]:
        """Gets a read-only view of the specified rows and columns."""
        return MatrixView[Self.T, ImmOrigin(Self.origin)](
            buffer=self._data.as_imm(),
            slice_x=rows,
            slice_y=cols,
            initial_nrows=self._nrows,
            initial_ncols=self._ncols,
            initial_row_stride=self._row_stride,
            initial_col_stride=self._col_stride,
            initial_offset=self._offset,
        )

    # [Mojo Miji]
    # The mirror of `Span.as_imm()`. A mutable view is an exclusive borrow, so
    # it cannot appear twice in one expression; demoting it to a read-only view
    # lifts that restriction, exactly as `&mut T` to `&T` does in Rust. There
    # is no inverse: nothing in the library promotes a read-only view back to a
    # mutable one.
    def as_imm(self) -> MatrixView[Self.T, ImmOrigin(Self.origin)]:
        """Returns a read-only view over the same elements.

        Returns:
            A view with the same shape, strides and offset, but a read-only
            origin, so that it may be combined with other views of the same
            matrix.
        """
        return MatrixView[Self.T, ImmOrigin(Self.origin)](
            buffer=self._data.as_imm(),
            nrows=self._nrows,
            ncols=self._ncols,
            row_stride=self._row_stride,
            col_stride=self._col_stride,
            offset=self._offset,
        )

    def unsafe_get(self, row: Int, col: Int) -> Self.ElementType:
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
            indices_within_bounds(row, col, self._nrows, self._ncols),
            "Debug assertion failed: Indices out of bounds in `unsafe_load`",
        )
        var offset = get_offset(
            row, col, self._row_stride, self._col_stride, self._offset
        )
        return self._data[offset].copy()

    # ===--------------------------------------------------------------------===#
    # Length and iteration
    # ===--------------------------------------------------------------------===#

    def __len__(self) -> Int:
        """Returns the number of rows.

        This is the row count rather than the element count so that `len()`
        agrees with what `__iter__` yields, the way it does for any Python
        sequence. Use `size()` for `nrows * ncols`.
        """
        return self._nrows

    def rows[
        forward: Bool = True
    ](self) -> MatrixAxisIter[Self.T, Self.origin, 0, forward]:
        """Iterates over the rows, yielding each as a `1 x ncols` view.

        Nothing is copied: each row borrows the parent buffer and inherits its
        mutability, so rows of a mutable view can be written through.

        Parameters:
            forward: True for first-to-last, False for last-to-first.
        """
        return MatrixAxisIter[axis=0, forward=forward](self)

    def cols[
        forward: Bool = True
    ](self) -> MatrixAxisIter[Self.T, Self.origin, 1, forward]:
        """Iterates over the columns, yielding each as an `nrows x 1` view.

        Parameters:
            forward: True for first-to-last, False for last-to-first.
        """
        return MatrixAxisIter[axis=1, forward=forward](self)

    def __iter__(self) -> MatrixAxisIter[Self.T, Self.origin, 0, True]:
        """Iterates over the rows, so `for row in view` walks row views."""
        return self.rows()

    # Mojo 1.0's builtin `reversed()` only accepts specific stdlib containers,
    # so it will not route here. Call `view.__reversed__()` or the clearer
    # `view.rows[False]()` instead; this stays for when a protocol hook lands.
    def __reversed__(
        self,
    ) -> MatrixAxisIter[Self.T, Self.origin, 0, False]:
        """Iterates over the rows from last to first."""
        return self.rows[False]()

    # ===--------------------------------------------------------------------===#
    # SIMD access
    # ===--------------------------------------------------------------------===#

    # [Mojo Miji]
    # `where Self.T == Scalar[d]` decides whether a method exists, but it does
    # not *refine* `T` inside the body: the compiler still type-checks the body
    # against the opaque `T`, so `self` will not bind to a parameter written as
    # `MatrixView[Scalar[d], ...]`. This is the one place that gap is bridged.
    # Every dtype-gated method below routes through it, which is why none of
    # them carries a `rebind` of its own.
    @always_inline
    def _as_simd[
        d: DType
    ](self) -> MatrixView[Scalar[d], Self.origin] where Self.T == Scalar[d]:
        """Returns `self` with its element type restated as `Scalar[d]`.

        Parameters:
            d: The dtype of the elements, deduced from `Self.T`.

        Returns:
            The same view, typed so that the SIMD routines accept it.
        """
        return rebind[MatrixView[Scalar[d], Self.origin]](self)

    def load[
        d: DType, //, width: Int = 1
    ](self, row: Int, col: Int) raises -> SIMD[d, width] where (
        Self.T == Scalar[d]
    ):
        """Loads `width` elements along row `row`, starting at column `col`.

        When the row is contiguous (`col_stride == 1`) this is a single vector
        load. Otherwise - which is what slicing with a step produces - it
        falls back to gathering element by element, so the call is always
        correct and only the speed changes.

        Parameters:
            d: The dtype of the elements, deduced from `Self.T`.
            width: How many elements to load.

        Args:
            row: The row to read from.
            col: The column at which the run starts.

        Raises:
            IndexError: If the run would leave the view.

        Returns:
            The `width` elements as a SIMD vector.
        """
        if (
            row < 0
            or row >= self._nrows
            or col < 0
            or col + width > self._ncols
        ):
            raise IndexError(
                function="MatrixView.load[width](self, row: Int, col: Int)",
                message="SIMD load runs past the end of the view.",
            )
        var base = get_offset(
            row, col, self._row_stride, self._col_stride, self._offset
        )
        if self._col_stride == 1:
            return (
                self._as_simd[d]()
                ._data.unsafe_ptr()
                .unsafe_offset(base)
                .unsafe_load[width=width]()
            )
        var result = SIMD[d, width]()
        for i in range(width):
            result[i] = rebind[Scalar[d]](
                self._data[base + i * self._col_stride]
            )
        return result

    # ===--------------------------------------------------------------------===#
    # Materialisation
    # ===--------------------------------------------------------------------===#

    def to_matrix(self) raises -> Matrix[Self.T]:
        """Copies the view into a new owning, C-contiguous `Matrix`.

        This is the one deliberate allocation in the view API. `copy()` returns
        another view of the same data, which is an O(1) handle copy; this walks
        the (possibly strided) view and produces dense owned storage.

        Returns:
            A new `Matrix` holding a dense copy of the viewed elements.
        """
        # The buffer is built by walking the view rather than allocated
        # zero-filled and then overwritten: it is one pass instead of two, and
        # it asks nothing of the element type beyond being copyable, so this
        # works for an element that has no zero.
        var buffer = List[Self.ElementType](capacity=self._nrows * self._ncols)
        for i in range(self._nrows):
            for j in range(self._ncols):
                buffer.append(self[i, j].copy())
        return Matrix[Self.T](
            buffer^,
            self._nrows,
            self._ncols,
            self._ncols,
            1,
        )

    def transpose(self) -> Matrix[Self.T]:
        """Returns the transpose of this view as a new owning matrix.

        The result is C-contiguous regardless of this view's layout.
        Transposing only moves elements, so this exists for every element type.

        There is no `.T` spelling to go with it: `T` is this struct's element
        type parameter, and a parameter and a method cannot share a name.

        Returns:
            A new matrix with the rows and columns exchanged.
        """
        return linamo.routines.linalg.transpose(self)

    def astype[
        d: DType, target: DType, //, Target: Copyable & Deinitable
    ](self) raises -> Matrix[Scalar[target]] where (
        Self.T == Scalar[d] and Target == Scalar[target]
    ):
        """Returns a C-contiguous copy of this view cast to `target`.

        Like `to_matrix()`, this materialises: the result owns its elements.

        Parameters:
            d: The dtype of this view's elements, deduced from `Self.T`.
            target: The dtype behind `Target`, deduced rather than
                written.
            Target: The type of the result elements.

        Returns:
            A new `Matrix[Scalar[target]]` with the same shape.
        """
        return linamo.routines.manipulation.astype[Scalar[target]](
            self._as_simd[d]()
        )

    # ===--------------------------------------------------------------------===#
    # String Representation and Writing
    # ===--------------------------------------------------------------------===#

    # [Mojo Miji]
    # Both of these read elements into text, so they exist only when the
    # element type can be written. The struct's `Writable` conformance carries
    # the same condition, which is what keeps `print(view)` available for
    # `MatrixView[Float64]` and absent for a view whose element has no
    # `write_to`.
    def _write_grid[
        W: Writer, //
    ](self, mut writer: W) where conforms_to(Self.T, Writable):
        """Writes the bracketed grid of elements, without a header.

        A row `plan_indices` leaves out is an empty list, which `write_grid`
        prints as `ELISION`; a column it leaves out carries the mark in every
        row, so it is measured and padded like any other column.
        """
        var elide = elides(self._nrows, self._ncols)
        var rows = plan_indices(self._nrows, elide)
        var cols = plan_indices(self._ncols, elide)
        var cells = List[List[String]]()
        for i in rows:
            var row = List[String]()
            if i >= 0:
                for j in cols:
                    if j >= 0:
                        row.append(
                            trim_fraction(
                                String(
                                    self._data[
                                        self._offset
                                        + i * self._row_stride
                                        + j * self._col_stride
                                    ]
                                )
                            )
                        )
                    else:
                        row.append(String(ELISION))
            cells.append(row^)
        write_grid(writer, cells, gap_position(cols))

    def __str__(self) -> String where conforms_to(Self.T, Writable):
        """Returns the grid of elements, without the header line."""
        var text = String("")
        self._write_grid(text)
        return text^

    def write_to[
        W: Writer, //
    ](self, mut writer: W) where conforms_to(Self.T, Writable):
        """Writes the matrix view to a writer, header line first."""
        write_header(
            writer,
            "MatrixView",
            element_type_name[Self.T](),
            self._nrows,
            self._ncols,
            self._row_stride,
            self._col_stride,
            self._offset,
        )
        writer.write("\n")
        self._write_grid(writer)

    # ===--------------------------------------------------------------------===#
    # Basic math dunders
    # ===--------------------------------------------------------------------===#
    # One overload per operation, taking a view. A `Matrix` right-hand side
    # converts implicitly, so no `Matrix`-operand twin is needed; see the
    # `@implicit` constructor above. The result is always an owning `Matrix`,
    # since an element-wise operation has to allocate somewhere.

    def __add__[
        d: DType, origin_b: Origin, //
    ](self, other: MatrixView[Self.T, origin_b]) raises -> Matrix[
        Scalar[d]
    ] where (Self.T == Scalar[d]):
        """Performs element-wise addition."""
        return linamo.routines.math.add(self._as_simd[d](), other._as_simd[d]())

    def __sub__[
        d: DType, origin_b: Origin, //
    ](self, other: MatrixView[Self.T, origin_b]) raises -> Matrix[
        Scalar[d]
    ] where (Self.T == Scalar[d]):
        """Performs element-wise subtraction."""
        return linamo.routines.math.sub(self._as_simd[d](), other._as_simd[d]())

    def __mul__[
        d: DType, origin_b: Origin, //
    ](self, other: MatrixView[Self.T, origin_b]) raises -> Matrix[
        Scalar[d]
    ] where (Self.T == Scalar[d]):
        """Performs matrix multiplication, the same as `@`."""
        return linamo.routines.math.matmul(
            self._as_simd[d](), other._as_simd[d]()
        )

    def __matmul__[
        d: DType, origin_b: Origin, //
    ](self, other: MatrixView[Self.T, origin_b]) raises -> Matrix[
        Scalar[d]
    ] where (Self.T == Scalar[d]):
        """Performs matrix multiplication."""
        return linamo.routines.math.matmul(
            self._as_simd[d](), other._as_simd[d]()
        )

    def __neg__[
        d: DType, //
    ](self) -> Matrix[Scalar[d]] where Self.T == Scalar[d]:
        """Negates every element."""
        return linamo.routines.math.scalar_rsub(
            self._as_simd[d](), Scalar[d](0)
        )

    # ===--------------------------------------------------------------------===#
    # Element-wise product, quotient and power
    # ===--------------------------------------------------------------------===#
    # `*` multiplies matrices and `**` raises one to a power, so the
    # element-wise forms are spelled out, exactly as on `Matrix`.

    def mul[
        d: DType, origin_b: Origin, //
    ](self, other: MatrixView[Self.T, origin_b]) raises -> Matrix[
        Scalar[d]
    ] where (Self.T == Scalar[d]):
        """Multiplies element by element (the Hadamard product)."""
        return linamo.routines.math.mul(self._as_simd[d](), other._as_simd[d]())

    def mul[
        d: DType, //
    ](self, other: Self.ElementType) -> Matrix[Scalar[d]] where (
        Self.T == Scalar[d]
    ):
        """Multiplies every element by a scalar, the same as `*`."""
        return linamo.routines.math.scalar_mul(
            self._as_simd[d](), rebind[Scalar[d]](other)
        )

    def div[
        d: DType, origin_b: Origin, //
    ](self, other: MatrixView[Self.T, origin_b]) raises -> Matrix[
        Scalar[d]
    ] where (Self.T == Scalar[d]):
        """Divides element by element."""
        return linamo.routines.math.div(self._as_simd[d](), other._as_simd[d]())

    def div[
        d: DType, //
    ](self, other: Self.ElementType) -> Matrix[Scalar[d]] where (
        Self.T == Scalar[d]
    ):
        """Divides every element by a scalar, the same as `/`."""
        return linamo.routines.math.scalar_div(
            self._as_simd[d](), rebind[Scalar[d]](other)
        )

    def pow[
        d: DType, origin_b: Origin, //
    ](self, other: MatrixView[Self.T, origin_b]) raises -> Matrix[
        Scalar[d]
    ] where (Self.T == Scalar[d]):
        """Raises each element to the matching element of `other`."""
        return linamo.routines.math.pow(self._as_simd[d](), other._as_simd[d]())

    def pow[
        d: DType, //
    ](self, other: Self.ElementType) -> Matrix[Scalar[d]] where (
        Self.T == Scalar[d]
    ):
        """Raises every element to a scalar power."""
        return linamo.routines.math.scalar_pow(
            self._as_simd[d](), rebind[Scalar[d]](other)
        )

    # ===--------------------------------------------------------------------===#
    # Scalar operands for the arithmetic dunders
    # ===--------------------------------------------------------------------===#

    def __add__[
        d: DType, //
    ](self, other: Self.ElementType) -> Matrix[Scalar[d]] where (
        Self.T == Scalar[d]
    ):
        """Adds a scalar to every element of the view."""
        return linamo.routines.math.scalar_add(
            self._as_simd[d](), rebind[Scalar[d]](other)
        )

    def __sub__[
        d: DType, //
    ](self, other: Self.ElementType) -> Matrix[Scalar[d]] where (
        Self.T == Scalar[d]
    ):
        """Subtracts a scalar from every element of the view."""
        return linamo.routines.math.scalar_sub(
            self._as_simd[d](), rebind[Scalar[d]](other)
        )

    def __mul__[
        d: DType, //
    ](self, other: Self.ElementType) -> Matrix[Scalar[d]] where (
        Self.T == Scalar[d]
    ):
        """Multiplies every element by a scalar of the view."""
        return linamo.routines.math.scalar_mul(
            self._as_simd[d](), rebind[Scalar[d]](other)
        )

    def __truediv__[
        d: DType, //
    ](self, other: Self.ElementType) -> Matrix[Scalar[d]] where (
        Self.T == Scalar[d]
    ):
        """Divides every element by a scalar of the view."""
        return linamo.routines.math.scalar_div(
            self._as_simd[d](), rebind[Scalar[d]](other)
        )

    def __floordiv__[
        d: DType, //
    ](self, other: Self.ElementType) -> Matrix[Scalar[d]] where (
        Self.T == Scalar[d]
    ):
        """Floor-divides every element by a scalar of the view."""
        return linamo.routines.math.scalar_floordiv(
            self._as_simd[d](), rebind[Scalar[d]](other)
        )

    def __mod__[
        d: DType, //
    ](self, other: Self.ElementType) -> Matrix[Scalar[d]] where (
        Self.T == Scalar[d]
    ):
        """Takes every element modulo a scalar of the view."""
        return linamo.routines.math.scalar_mod(
            self._as_simd[d](), rebind[Scalar[d]](other)
        )

    def __pow__[
        d: DType, //
    ](self, exponent: Int) raises -> Matrix[Scalar[d]] where (
        Self.T == Scalar[d]
    ):
        """Raises the viewed matrix to an integer power, as on `Matrix`."""
        return linamo.routines.linalg.matrix_power(self._as_simd[d](), exponent)

    # ===--------------------------------------------------------------------===#
    # floordiv and mod
    # ===--------------------------------------------------------------------===#
    # Element-wise, and unambiguously so: neither has a linear-algebra reading
    # that `*` and `**` could be confused with.

    def __floordiv__[
        d: DType, origin_b: Origin, //
    ](self, other: MatrixView[Self.T, origin_b]) raises -> Matrix[
        Scalar[d]
    ] where (Self.T == Scalar[d]):
        """Performs element-wise floor division."""
        return linamo.routines.math.floordiv(
            self._as_simd[d](), other._as_simd[d]()
        )

    def __mod__[
        d: DType, origin_b: Origin, //
    ](self, other: MatrixView[Self.T, origin_b]) raises -> Matrix[
        Scalar[d]
    ] where (Self.T == Scalar[d]):
        """Performs element-wise modulo."""
        return linamo.routines.math.mod(self._as_simd[d](), other._as_simd[d]())

    # ===--------------------------------------------------------------------===#
    # Reflected scalar operators
    # ===--------------------------------------------------------------------===#
    # There are no in-place counterparts (`+=`, `-=`, ...) on a view: the type
    # is generic over `origin` and Mojo checks a method body against the
    # read-only instantiation as well, so nothing writing through `self._data`
    # can be defined here. See `routines/mutation.mojo`.

    def __radd__[
        d: DType, //
    ](self, other: Self.ElementType) -> Matrix[Scalar[d]] where (
        Self.T == Scalar[d]
    ):
        """Adds every element of the view to a scalar (`scalar + view`)."""
        return linamo.routines.math.scalar_add(
            self._as_simd[d](), rebind[Scalar[d]](other)
        )

    def __rmul__[
        d: DType, //
    ](self, other: Self.ElementType) -> Matrix[Scalar[d]] where (
        Self.T == Scalar[d]
    ):
        """Multiplies a scalar by every element (`scalar * view`)."""
        return linamo.routines.math.scalar_mul(
            self._as_simd[d](), rebind[Scalar[d]](other)
        )

    def __rsub__[
        d: DType, //
    ](self, other: Self.ElementType) -> Matrix[Scalar[d]] where (
        Self.T == Scalar[d]
    ):
        """Subtracts every element of the view from a scalar (`scalar - view`).
        """
        return linamo.routines.math.scalar_rsub(
            self._as_simd[d](), rebind[Scalar[d]](other)
        )

    def __rtruediv__[
        d: DType, //
    ](self, other: Self.ElementType) -> Matrix[Scalar[d]] where (
        Self.T == Scalar[d]
    ):
        """Divides a scalar by every element of the view (`scalar / view`)."""
        return linamo.routines.math.scalar_rdiv(
            self._as_simd[d](), rebind[Scalar[d]](other)
        )

    # ===--------------------------------------------------------------------===#
    # Arbitrary-precision operands
    # ===--------------------------------------------------------------------===#
    # The counterpart of the block of the same name on `Matrix`: one more
    # overload of each operator, selected by `conforms_to(Self.T, Numeric)`
    # where the ones above are selected by `Self.T == Scalar[d]`.

    def __add__[
        origin_b: Origin, //
    ](self, other: MatrixView[Self.T, origin_b]) raises -> Matrix[
        Self.T
    ] where conforms_to(Self.T, Numeric):
        """Performs element-wise addition."""
        return linamo.routines.math.add(self, other)

    def __sub__[
        origin_b: Origin, //
    ](self, other: MatrixView[Self.T, origin_b]) raises -> Matrix[
        Self.T
    ] where conforms_to(Self.T, Numeric):
        """Performs element-wise subtraction."""
        return linamo.routines.math.sub(self, other)

    def __mul__[
        origin_b: Origin, //
    ](self, other: MatrixView[Self.T, origin_b]) raises -> Matrix[
        Self.T
    ] where conforms_to(Self.T, Numeric):
        """Performs matrix multiplication, the same as `@`."""
        return linamo.routines.math.matmul(self, other)

    def __matmul__[
        origin_b: Origin, //
    ](self, other: MatrixView[Self.T, origin_b]) raises -> Matrix[
        Self.T
    ] where conforms_to(Self.T, Numeric):
        """Performs matrix multiplication."""
        return linamo.routines.math.matmul(self, other)

    def __neg__(
        self,
    ) raises -> Matrix[Self.T] where conforms_to(Self.T, Numeric):
        """Negates every element."""
        return linamo.routines.math.neg(self)

    def __pow__(
        self, exponent: Int
    ) raises -> Matrix[Self.T] where conforms_to(
        Self.T, Numeric
    ) and conforms_to(Self.T, Comparable):
        """Raises the viewed matrix to an integer power, as on `Matrix`."""
        return linamo.routines.linalg.matrix_power(self, exponent)

    def mul[
        origin_b: Origin, //
    ](self, other: MatrixView[Self.T, origin_b]) raises -> Matrix[
        Self.T
    ] where conforms_to(Self.T, Numeric):
        """Multiplies element by element (the Hadamard product)."""
        return linamo.routines.math.mul(self, other)

    def mul(
        self, other: Self.ElementType
    ) raises -> Matrix[Self.T] where conforms_to(Self.T, Numeric):
        """Multiplies every element by a value, the same as `*`."""
        return linamo.routines.math.scalar_mul(self, other)

    def div[
        origin_b: Origin, //
    ](self, other: MatrixView[Self.T, origin_b]) raises -> Matrix[
        Self.T
    ] where conforms_to(Self.T, Numeric):
        """Divides element by element."""
        return linamo.routines.math.div(self, other)

    def div(
        self, other: Self.ElementType
    ) raises -> Matrix[Self.T] where conforms_to(Self.T, Numeric):
        """Divides every element by a value, the same as `/`."""
        return linamo.routines.math.scalar_div(self, other)

    def __add__(
        self, other: Self.ElementType
    ) raises -> Matrix[Self.T] where conforms_to(Self.T, Numeric):
        """Adds a value to every element of the view."""
        return linamo.routines.math.scalar_add(self, other)

    def __sub__(
        self, other: Self.ElementType
    ) raises -> Matrix[Self.T] where conforms_to(Self.T, Numeric):
        """Subtracts a value from every element of the view."""
        return linamo.routines.math.scalar_sub(self, other)

    def __mul__(
        self, other: Self.ElementType
    ) raises -> Matrix[Self.T] where conforms_to(Self.T, Numeric):
        """Multiplies every element of the view by a value."""
        return linamo.routines.math.scalar_mul(self, other)

    def __truediv__(
        self, other: Self.ElementType
    ) raises -> Matrix[Self.T] where conforms_to(Self.T, Numeric):
        """Divides every element of the view by a value."""
        return linamo.routines.math.scalar_div(self, other)

    def __radd__(
        self, other: Self.ElementType
    ) raises -> Matrix[Self.T] where conforms_to(Self.T, Numeric):
        """Adds every element of the view to a value (`value + view`)."""
        return linamo.routines.math.scalar_add(self, other)

    def __rmul__(
        self, other: Self.ElementType
    ) raises -> Matrix[Self.T] where conforms_to(Self.T, Numeric):
        """Multiplies a value by every element (`value * view`)."""
        return linamo.routines.math.scalar_mul(self, other)

    def __rsub__(
        self, other: Self.ElementType
    ) raises -> Matrix[Self.T] where conforms_to(Self.T, Numeric):
        """Subtracts every element of the view from a value (`value - view`)."""
        return linamo.routines.math.scalar_rsub(self, other)

    def __rtruediv__(
        self, other: Self.ElementType
    ) raises -> Matrix[Self.T] where conforms_to(Self.T, Numeric):
        """Divides a value by every element of the view (`value / view`)."""
        return linamo.routines.math.scalar_rdiv(self, other)

    # ===--------------------------------------------------------------------===#
    # Comparison operators
    # ===--------------------------------------------------------------------===#
    # Element-wise `Matrix[DType.bool]` masks, as on `Matrix`.

    def __lt__[
        d: DType, origin_b: Origin, //
    ](self, other: MatrixView[Self.T, origin_b]) raises -> Matrix[
        Scalar[DType.bool]
    ] where (Self.T == Scalar[d]):
        """Element-wise less-than comparison with another matrix or view."""
        return linamo.routines.logic.less(
            self._as_simd[d](), other._as_simd[d]()
        )

    def __lt__[
        d: DType, //
    ](self, other: Self.ElementType) -> Matrix[Scalar[DType.bool]] where (
        Self.T == Scalar[d]
    ):
        """Element-wise less-than comparison against a scalar."""
        return linamo.routines.logic.scalar_less(
            self._as_simd[d](), rebind[Scalar[d]](other)
        )

    def __le__[
        d: DType, origin_b: Origin, //
    ](self, other: MatrixView[Self.T, origin_b]) raises -> Matrix[
        Scalar[DType.bool]
    ] where (Self.T == Scalar[d]):
        """Element-wise less-than-or-equal comparison with another matrix or view.
        """
        return linamo.routines.logic.less_equal(
            self._as_simd[d](), other._as_simd[d]()
        )

    def __le__[
        d: DType, //
    ](self, other: Self.ElementType) -> Matrix[Scalar[DType.bool]] where (
        Self.T == Scalar[d]
    ):
        """Element-wise less-than-or-equal comparison against a scalar."""
        return linamo.routines.logic.scalar_less_equal(
            self._as_simd[d](), rebind[Scalar[d]](other)
        )

    def __gt__[
        d: DType, origin_b: Origin, //
    ](self, other: MatrixView[Self.T, origin_b]) raises -> Matrix[
        Scalar[DType.bool]
    ] where (Self.T == Scalar[d]):
        """Element-wise greater-than comparison with another matrix or view."""
        return linamo.routines.logic.greater(
            self._as_simd[d](), other._as_simd[d]()
        )

    def __gt__[
        d: DType, //
    ](self, other: Self.ElementType) -> Matrix[Scalar[DType.bool]] where (
        Self.T == Scalar[d]
    ):
        """Element-wise greater-than comparison against a scalar."""
        return linamo.routines.logic.scalar_greater(
            self._as_simd[d](), rebind[Scalar[d]](other)
        )

    def __ge__[
        d: DType, origin_b: Origin, //
    ](self, other: MatrixView[Self.T, origin_b]) raises -> Matrix[
        Scalar[DType.bool]
    ] where (Self.T == Scalar[d]):
        """Element-wise greater-than-or-equal comparison with another matrix or view.
        """
        return linamo.routines.logic.greater_equal(
            self._as_simd[d](), other._as_simd[d]()
        )

    def __ge__[
        d: DType, //
    ](self, other: Self.ElementType) -> Matrix[Scalar[DType.bool]] where (
        Self.T == Scalar[d]
    ):
        """Element-wise greater-than-or-equal comparison against a scalar."""
        return linamo.routines.logic.scalar_greater_equal(
            self._as_simd[d](), rebind[Scalar[d]](other)
        )

    def __eq__[
        d: DType, origin_b: Origin, //
    ](self, other: MatrixView[Self.T, origin_b]) raises -> Matrix[
        Scalar[DType.bool]
    ] where (Self.T == Scalar[d]):
        """Element-wise equality comparison with another matrix or view."""
        return linamo.routines.logic.equal(
            self._as_simd[d](), other._as_simd[d]()
        )

    def __eq__[
        d: DType, //
    ](self, other: Self.ElementType) -> Matrix[Scalar[DType.bool]] where (
        Self.T == Scalar[d]
    ):
        """Element-wise equality comparison against a scalar."""
        return linamo.routines.logic.scalar_equal(
            self._as_simd[d](), rebind[Scalar[d]](other)
        )

    def __ne__[
        d: DType, origin_b: Origin, //
    ](self, other: MatrixView[Self.T, origin_b]) raises -> Matrix[
        Scalar[DType.bool]
    ] where (Self.T == Scalar[d]):
        """Element-wise inequality comparison with another matrix or view."""
        return linamo.routines.logic.not_equal(
            self._as_simd[d](), other._as_simd[d]()
        )

    def __ne__[
        d: DType, //
    ](self, other: Self.ElementType) -> Matrix[Scalar[DType.bool]] where (
        Self.T == Scalar[d]
    ):
        """Element-wise inequality comparison against a scalar."""
        return linamo.routines.logic.scalar_not_equal(
            self._as_simd[d](), rebind[Scalar[d]](other)
        )
