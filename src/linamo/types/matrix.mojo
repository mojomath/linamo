"""
This module defines the `Matrix` type, which is a dynamically sized 2D matrix.
"""


from decimo import Numeric

from linamo.types.errors import IndexError, ValueError
from linamo.types.matrix_iter import MatrixAxisIter
from linamo.types.matrix_view import MatrixView
import linamo.routines.linalg
import linamo.routines.math
import linamo.routines.logic
import linamo.routines.manipulation
import linamo.routines.mutation as mutation
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
from linamo.utils.indexing import (
    get_offset,
    indices_within_bounds,
    layout_fits_buffer,
    layout_is_dense,
)


struct Matrix[T: Copyable & Deinitable](
    Copyable, Movable, Sized, Writable where conforms_to(T, Writable)
):
    """A 2D matrix type.
    A matrix owns its data and can write to it. The elements are stored in a
    contiguous block of memory in either row-major (C-contiguous) or
    column-major (Fortran-contiguous) order.

    Parameters:
        T: The type of the matrix elements.
    """

    # [Mojo Miji]
    # The parameter is the element *type*, not a `DType`, so a matrix is
    # spelled the way every other container is: `Matrix[Float64]` beside
    # `List[Float64]`. `Float64` is itself `Scalar[DType.float64]`, so nothing
    # about the scalar case is lost --- the routines recover the dtype by
    # matching `Matrix[Scalar[d]]`, which infers `d` --- and an element type
    # that has no `DType` at all, such as an arbitrary-precision integer, is
    # now expressible.
    #
    # The bound is `List`'s own bound. A matrix is a container first, so it
    # accepts exactly what its storage accepts, and the arithmetic
    # requirements are asked for per method rather than up front.
    comptime ElementType = Self.T
    """The type of the elements in the matrix. A second name for `T`, used in
    signatures where `T` alone would read as a stray letter."""

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
    var _data: List[Self.ElementType]
    """The elements of the matrix stored in a contiguous block of memory."""
    var _nrows: Int
    """The number of rows in the matrix."""
    var _ncols: Int
    """The number of columns in the matrix."""
    var _row_stride: Int
    """The stride (in number of elements) to move to the next row."""
    var _col_stride: Int
    """The stride (in number of elements) to move to the next column."""

    # ===--------------------------------------------------------------------===#
    # Retrieve attributes
    # ===--------------------------------------------------------------------===#
    @always_inline
    def data(self) -> Span[Self.ElementType, origin_of(self._data)]:
        """Returns the underlying data of the matrix."""
        return Span(self._data)

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
        return 0

    @always_inline
    def size(self) -> Int:
        """Returns the total number of elements in the matrix."""
        return self._nrows * self._ncols

    def is_c_contiguous(self) -> Bool:
        """Returns True if the matrix is C-contiguous (row-major, dense)."""
        return self._col_stride == 1 and self._row_stride == self._ncols

    def is_f_contiguous(self) -> Bool:
        """Returns True if the matrix is F-contiguous (column-major, dense)."""
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

    # [Mojo Miji]
    # The shape, the two strides and the length of `data` are one invariant
    # bundle, not five independent numbers: indexing computes
    # `row * row_stride + col * col_stride`, so together they decide which
    # buffer slots `m[i, j]` can reach. Two properties have to hold, and the
    # constructors below are the only place they can be established.
    #
    # `layout_fits_buffer` is the easier one --- `row_stride = 100` over a
    # six-element buffer indexes past the end, which `List` catches under
    # `-D ASSERT=all` and is undefined in release. `layout_is_dense` is the
    # subtler one: `row_stride = 0` makes every row the same row, so
    # `m[0, 0] = 5` silently also writes `m[1, 0]`. That is a legitimate state
    # for a `MatrixView` --- it is what `broadcast_to` produces --- and never
    # for a matrix that owns its buffer, which is why the check lives here and
    # not on the view.
    #
    # `debug_assert` costs nothing in release, so this is a statement of the
    # invariant that the test suite executes rather than a runtime tax. The
    # copying and moving constructors below take their layout from a matrix
    # that already satisfies both, so they have nothing to establish.

    def __init__(
        out self,
        var buffer: List[Self.ElementType],
        nrows: Int,
        ncols: Int,
        row_stride: Int,
        col_stride: Int,
    ):
        debug_assert(
            layout_is_dense(nrows, ncols, row_stride, col_stride),
            "Debug assertion failed: `Matrix` layout is not C- or F-major",
        )
        debug_assert(
            layout_fits_buffer(
                nrows, ncols, row_stride, col_stride, len(buffer)
            ),
            "Debug assertion failed: `Matrix` layout overruns its buffer",
        )
        self._data = buffer^
        self._nrows = nrows
        self._ncols = ncols
        self._row_stride = row_stride
        self._col_stride = col_stride

    def __init__[
        d: DType, //
    ](
        out self,
        nrows: Int,
        ncols: Int,
        row_stride: Int,
        col_stride: Int,
    ) where (Self.T == Scalar[d]):
        debug_assert(
            layout_is_dense(nrows, ncols, row_stride, col_stride),
            "Debug assertion failed: `Matrix` layout is not C- or F-major",
        )
        debug_assert(
            layout_fits_buffer(
                nrows, ncols, row_stride, col_stride, nrows * ncols
            ),
            "Debug assertion failed: `Matrix` layout overruns its buffer",
        )
        # `0` is a value only a numeric type has, so it is spelled as a
        # `Scalar[d]` and restated as `Self.T` --- a rebind of one element,
        # not of the buffer.
        self._data = List[Self.ElementType](
            length=nrows * ncols, fill=rebind[Self.ElementType](Scalar[d](0))
        )
        self._nrows = nrows
        self._ncols = ncols
        self._row_stride = row_stride
        self._col_stride = col_stride

    def __init__(out self, *, copy: Self):
        """Initializes the matrix by copying another matrix."""
        self._data = copy._data.copy()
        self._nrows = copy._nrows
        self._ncols = copy._ncols
        self._row_stride = copy._row_stride
        self._col_stride = copy._col_stride

    def __init__(out self, *, deinit move: Self):
        """Initializes the matrix by moving another matrix."""
        self._data = move._data^
        self._nrows = move._nrows
        self._ncols = move._ncols
        self._row_stride = move._row_stride
        self._col_stride = move._col_stride

    # ===--------------------------------------------------------------------===#
    # Element Access and Mutation
    # View Access
    # ===--------------------------------------------------------------------===#

    # [Mojo Miji]
    # This method returns a reference to the element at the specified indices.
    # The mutability of the reference is determined by the mutability of the
    # underlying data (self._data). Since self._data is a mutable list, the
    # reference returned by __getitem__ is mutable, allowing for both reading
    # and writing to the matrix elements.
    # Thus, `__setitem__` is not needed as a separate method.
    #
    # [Mojo Miji]
    # The returned origin is `origin_of(self._data)` - the whole buffer - and
    # not the finer `self._data[row * row_stride + col * col_stride]`, which is
    # what this method used to name. A per-element origin sounds more precise,
    # and it is, but forming a second one invalidated the first, so
    #
    #     var s = a[0, 0] + a[1, 1]
    #
    # did not compile on a `var` matrix. Naming the buffer instead lets any
    # number of element references coexist. The reference has to be built from
    # a pointer because `self._data[i]` would re-derive the narrow origin.
    #
    # `__setitem__` is deliberately absent, and not only because the mutable
    # reference above makes it unnecessary. Defining `__setitem__` on this type
    # makes the compiler pass `self` to `__getitem__` as a temporary copy in
    # some positions, so a view sliced from it carries the origin of a dead
    # temporary and `a[0:1, :] - a[1:2, :]` stops compiling. Region writes are
    # spelled `set(...)` for the same reason.
    def __getitem__(
        ref self, row: Int, col: Int
    ) raises -> ref[origin_of(self._data)] Self.ElementType:
        """Gets the element at the specified indices.

        Args:
            row: The row index.
            col: The column index.

        Raises:
            IndexError: If the indices are out of bounds.

        Returns:
            The element at the specified indices.
        """
        if row < 0 or row >= self._nrows or col < 0 or col >= self._ncols:
            raise IndexError(
                function=(
                    "Matrix.__getitem__(self, row: Int, col: Int) ->"
                    " Self.ElementType"
                ),
                message="Index out of bounds.",
            )
        return self._data._data.unsafe_offset(
            row * self._row_stride + col * self._col_stride
        )[]

    # [Mojo Miji]
    # When you pass `Self.T` and `origin_of(self)` as parameters to the
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
    # `origin_of(self._data)` would inherit the caller's mutability, so slicing
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
    ) raises -> MatrixView[T=Self.T, origin=origin_of(self._data)]:
        """Gets a read-only view of the specified rows and columns."""
        return MatrixView(
            buffer=self._data,
            slice_x=x,
            slice_y=y,
            initial_nrows=self._nrows,
            initial_ncols=self._ncols,
            initial_row_stride=self._row_stride,
            initial_col_stride=self._col_stride,
            initial_offset=0,
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
        var offset = get_offset(row, col, self._row_stride, self._col_stride)
        return self._data._data.unsafe_offset(offset)[].copy()

    # [Mojo Miji]
    # `read self`, not `ref self`. This method is the Matrix -> MatrixView
    # conversion the routine layer uses about forty times to feed kernels, and
    # every one of those uses is a read. Taking `ref self` made it hand back a
    # *mutable* view whenever the receiver was a `var`, which meant
    # `m.view() - m.view()` was rejected as two mutable borrows of one matrix,
    # and meant an innocuous-looking call was a write door. It is now a
    # shorthand for `m[:, :]` and nothing more.
    def view(imm self) -> MatrixView[Self.T, origin_of(self._data)]:
        """Gets a read-only view of the entire matrix.

        This is the same thing `m[:, :]` produces, and is the conversion the
        named routines use to accept a `Matrix` wherever a `MatrixView` is
        expected. To write through a sub-matrix, use `view_mut` from
        `linamo.routines.mutation`.
        """
        return MatrixView(
            buffer=Span(self._data),
            nrows=self._nrows,
            ncols=self._ncols,
            row_stride=self._row_stride,
            col_stride=self._col_stride,
            offset=0,
        )

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
    ](self) -> MatrixAxisIter[Self.T, origin_of(self._data), 0, forward]:
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
    ](self) -> MatrixAxisIter[Self.T, origin_of(self._data), 1, forward]:
        """Iterates over the columns, yielding each as an `nrows x 1` view.

        Parameters:
            forward: True for first-to-last, False for last-to-first.
        """
        return MatrixAxisIter[axis=1, forward=forward](self.view())

    def __iter__(
        self,
    ) -> MatrixAxisIter[Self.T, origin_of(self._data), 0, True]:
        """Iterates over the rows, so `for row in matrix` walks row views."""
        return self.rows()

    # Mojo 1.0's builtin `reversed()` only accepts specific stdlib containers,
    # so it will not route here. Call `matrix.__reversed__()` or the clearer
    # `matrix.rows[False]()` instead; this stays for when a protocol hook
    # lands.
    def __reversed__(
        self,
    ) -> MatrixAxisIter[Self.T, origin_of(self._data), 0, False]:
        """Iterates over the rows from last to first."""
        return self.rows[False]()

    # ===--------------------------------------------------------------------===#
    # SIMD access
    # ===--------------------------------------------------------------------===#

    # [Mojo Miji]
    # `where Self.T == Scalar[d]` decides whether a method exists, but it does
    # not *refine* `T` inside the body: the body is still checked against the
    # opaque `T`, so `self` will not bind to a parameter written in terms of
    # `Scalar[d]`. This restates the element type for the routines that need
    # it, and does so as a *view*, which is an O(1) metadata copy rather than a
    # copy of the buffer. Every dtype-gated method below routes through it.
    @always_inline
    def _simd_view[
        d: DType
    ](imm self) -> MatrixView[Scalar[d], origin_of(self._data)] where (
        Self.T == Scalar[d]
    ):
        """Returns a read-only view with the element type restated.

        Parameters:
            d: The dtype of the elements, deduced from `Self.T`.

        Returns:
            The same elements, typed so that the SIMD routines accept them.
        """
        return rebind[MatrixView[Scalar[d], origin_of(self._data)]](self.view())

    def load[
        d: DType, //, width: Int = 1
    ](self, row: Int, col: Int) raises -> SIMD[d, width] where (
        Self.T == Scalar[d]
    ):
        """Loads `width` elements along row `row`, starting at column `col`.

        When the row is contiguous (`col_stride == 1`) this is a single vector
        load; otherwise it gathers element by element, so the call is always
        correct and only the speed changes.

        Parameters:
            d: The dtype of the elements, deduced from `Self.T`.
            width: How many elements to load.

        Args:
            row: The row to read from.
            col: The column at which the run starts.

        Raises:
            IndexError: If the run would leave the matrix.

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
                function="Matrix.load[width](self, row: Int, col: Int)",
                message="SIMD load runs past the end of the matrix.",
            )
        var base = get_offset(row, col, self._row_stride, self._col_stride)
        if self._col_stride == 1:
            return (
                self._simd_view[d]()
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

    def store[
        d: DType, //, width: Int = 1
    ](mut self, row: Int, col: Int, value: SIMD[d, width]) raises where (
        Self.T == Scalar[d]
    ):
        """Stores `width` elements along row `row`, starting at column `col`.

        Unlike the view equivalent this can be a method, because a `Matrix`
        owns a concretely mutable `List` - there is no generic origin whose
        read-only instantiation the body would also have to satisfy.

        Parameters:
            d: The dtype of the elements, deduced from `Self.T`.
            width: How many elements to store.

        Args:
            row: The row to write to.
            col: The column at which the run starts.
            value: The elements to write.

        Raises:
            IndexError: If the run would leave the matrix.
        """
        if (
            row < 0
            or row >= self._nrows
            or col < 0
            or col + width > self._ncols
        ):
            raise IndexError(
                function="Matrix.store[width](mut self, row, col, value)",
                message="SIMD store runs past the end of the matrix.",
            )
        var base = get_offset(row, col, self._row_stride, self._col_stride)
        # A `mut` rebind of the buffer would copy it, so the element type is
        # restated on the pointer instead --- a value, and one the write goes
        # straight through.
        var ptr = self._data.unsafe_ptr().unsafe_bitcast[Scalar[d]]()
        if self._col_stride == 1:
            ptr.unsafe_offset(base).unsafe_store(value)
        else:
            for i in range(width):
                ptr.unsafe_offset(base + i * self._col_stride)[] = value[i]

    # ===--------------------------------------------------------------------===#
    # Region writes
    # ===--------------------------------------------------------------------===#

    # Every write is spelled `set`, and which one runs is decided by the
    # arguments: a `Self.ElementType` fills, a `MatrixView` copies. There is no
    # ambiguity to worry about, because nothing in the library converts a
    # scalar to a matrix, and a `Matrix` source is accepted through the
    # implicit `Matrix` -> `MatrixView` conversion.
    #
    # These are named methods rather than `__setitem__`: Mojo 1.0 routes
    # `a[i:j, k:l] = rhs` through `__getitem__`, which would force `rhs` to be
    # a view carrying this matrix's own origin. See `routines/mutation.mojo`.
    #
    # Each one delegates to `routines.mutation` rather than looping over
    # `self._data` itself. Writing the loop twice is what let the two copies
    # drift: the version that lived here used `Slice.indices()` and agreed with
    # Python on `m.set(3:1, ..)`, while the view constructor computed a
    # negative extent for the same slice. One loop cannot disagree with itself.
    #
    # The whole-matrix scalar write is the one exception, and it does not
    # duplicate anything: a `Matrix` owns exactly `nrows * ncols` elements at
    # offset zero, so every slot in `data` belongs to it under either layout.
    # That makes the write a flat walk of the buffer with no index arithmetic,
    # no bounds to check and nothing to raise - both simpler and faster than
    # the strided region path it would otherwise borrow.
    def set(mut self, value: Self.ElementType):
        """Writes one scalar into every element of the matrix.

        Args:
            value: The scalar written to every element.
        """
        for i in range(len(self._data)):
            self._data[i] = value.copy()

    def set(mut self, rows: Slice, cols: Slice, value: Self.ElementType) raises:
        """Writes one scalar into every element of the selected region.

        Args:
            rows: The rows to write to.
            cols: The columns to write to.
            value: The scalar written to every selected element.
        """
        var target = mutation.view_mut(self, rows, cols)
        mutation.fill(
            target, Slice(0, target.nrows()), Slice(0, target.ncols()), value
        )

    def set[
        mut_b: Bool, //, origin_b: Origin[mut=mut_b]
    ](mut self, src: MatrixView[Self.T, origin_b]) raises:
        """Copies `src` into the whole matrix.

        Parameters:
            mut_b: Whether the source view is mutable.
            origin_b: The origin of the source view.

        Args:
            src: The source, which must match this matrix's shape exactly.

        Raises:
            ValueError: If the shapes do not match.
        """
        self.set(Slice(0, self._nrows), Slice(0, self._ncols), src)

    def set[
        mut_b: Bool, //, origin_b: Origin[mut=mut_b]
    ](
        mut self,
        rows: Slice,
        cols: Slice,
        src: MatrixView[Self.T, origin_b],
    ) raises:
        """Copies `src` into the region selected by `rows` and `cols`.

        Parameters:
            mut_b: Whether the source view is mutable.
            origin_b: The origin of the source view.

        Args:
            rows: The rows to write to.
            cols: The columns to write to.
            src: The source, which must match the target shape exactly.

        Raises:
            ValueError: If the shapes do not match.
        """
        var target = mutation.view_mut(self, rows, cols)
        mutation.assign(
            target, Slice(0, target.nrows()), Slice(0, target.ncols()), src
        )

    def set(mut self, row: Int, col: Int, value: Self.ElementType) raises:
        """Writes one element.

        `m[row, col] = value` does the same thing through the reference from
        `__getitem__`; this overload exists so that every write can be spelled
        the same way.

        Args:
            row: The row index.
            col: The column index.
            value: The value to write.

        Raises:
            IndexError: If the indices are out of bounds.
        """
        self[row, col] = value.copy()

    def view_mut(
        ref self, x: Slice, y: Slice
    ) raises -> MatrixView[Self.T, origin_of(self._data)]:
        """Gets a writable view of a region of this matrix.

        The counterpart of `m[x, y]`, which is always read-only. The view
        inherits the mutability of the receiver, so it is writable when bound
        to a `var` and read-only otherwise - including when the receiver is a
        temporary, where the result is read-only and passing it to `fill`,
        `store` or `assign` fails to compile.

        A mutable view is an exclusive borrow, so it cannot appear twice in one
        expression. Use `MatrixView.as_imm()` to demote it when it has to be
        combined with another view of the same matrix.

        Args:
            x: The rows to include.
            y: The columns to include.

        Returns:
            A view of the region, writable exactly when the receiver is.
        """
        return mutation.view_mut(self, x, y)

    # ===--------------------------------------------------------------------===#
    # Rearrangement
    # ===--------------------------------------------------------------------===#

    def transpose(self) -> Matrix[Self.T]:
        """Returns the transpose of this matrix.

        The result is a new C-contiguous matrix regardless of this matrix's
        layout. Transposing only moves elements, so this exists for every
        element type.

        There is no `.T` spelling to go with it: `T` is this struct's element
        type parameter, and a parameter and a method cannot share a name.

        Returns:
            A new matrix with the rows and columns exchanged.
        """
        return linamo.routines.linalg.transpose(self)

    # ===--------------------------------------------------------------------===#
    # Type conversion
    # ===--------------------------------------------------------------------===#

    def astype[
        d: DType, target: DType, //, Target: Copyable & Deinitable
    ](self) raises -> Matrix[Scalar[target]] where (
        Self.T == Scalar[d] and Target == Scalar[target]
    ):
        """Returns a C-contiguous copy of this matrix cast to `target`.

        Parameters:
            d: The dtype of this matrix's elements, deduced from `Self.T`.
            target: The dtype behind `Target`, deduced rather than
                written.
            Target: The type of the result elements.

        Returns:
            A new `Matrix[Scalar[target]]` with the same shape.
        """
        return linamo.routines.manipulation.astype[Scalar[target]](
            self._simd_view[d]()
        )

    # ===--------------------------------------------------------------------===#
    # String Representation and Writing
    # ===--------------------------------------------------------------------===#

    # [Mojo Miji]
    # Both of these render elements as text, so they exist only when the
    # element type can be written. The struct's `Writable` conformance carries
    # the same condition, which is what keeps `print(m)` available for
    # `Matrix[Float64]` and absent for an element type with no `write_to`.
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
                                        get_offset(
                                            i,
                                            j,
                                            self._row_stride,
                                            self._col_stride,
                                        )
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
        """Writes the matrix to a writer, header line first."""
        write_header(
            writer,
            "Matrix",
            element_type_name[Self.T](),
            self._nrows,
            self._ncols,
            self._row_stride,
            self._col_stride,
            0,
        )
        writer.write("\n")
        self._write_grid(writer)

    # ===------------------------------------------------------------------ ===#
    # Basic math dunders
    # ===------------------------------------------------------------------ ===#
    # The right-hand operand is a `MatrixView`, and that one overload also
    # serves `A + B` between two matrices: a `Matrix` argument converts through
    # the `@implicit` constructor in `types/matrix_view.mojo`. The conversion
    # is an O(1) metadata copy and yields a read-only view, so `A + A` is a
    # pair of shared borrows rather than an aliasing violation.

    def __add__[
        d: DType, origin: Origin, //
    ](self, other: MatrixView[Self.T, origin]) raises -> Matrix[
        Scalar[d]
    ] where (Self.T == Scalar[d]):
        """Performs element-wise addition."""
        return linamo.routines.math.add(
            self._simd_view[d](), other._as_simd[d]()
        )

    def __sub__[
        d: DType, origin: Origin, //
    ](self, other: MatrixView[Self.T, origin]) raises -> Matrix[
        Scalar[d]
    ] where (Self.T == Scalar[d]):
        """Performs element-wise subtraction."""
        return linamo.routines.math.sub(
            self._simd_view[d](), other._as_simd[d]()
        )

    def __mul__[
        d: DType, origin: Origin, //
    ](self, other: MatrixView[Self.T, origin]) raises -> Matrix[
        Scalar[d]
    ] where (Self.T == Scalar[d]):
        """Performs matrix multiplication, the same as `@`."""
        return linamo.routines.math.matmul(
            self._simd_view[d](), other._as_simd[d]()
        )

    def __matmul__[
        d: DType, origin: Origin, //
    ](self, other: MatrixView[Self.T, origin]) raises -> Matrix[
        Scalar[d]
    ] where (Self.T == Scalar[d]):
        """Performs matrix multiplication."""
        return linamo.routines.math.matmul(
            self._simd_view[d](), other._as_simd[d]()
        )

    def __neg__[
        d: DType, //
    ](self) -> Matrix[Scalar[d]] where Self.T == Scalar[d]:
        """Negates every element."""
        return linamo.routines.math.scalar_rsub(
            self._simd_view[d](), Scalar[d](0)
        )

    # ===------------------------------------------------------------------ ===#
    # Element-wise product, quotient and power
    # ===------------------------------------------------------------------ ===#
    # `*` multiplies matrices and `**` raises one to a power, so the
    # element-wise forms are spelled out. Each is the method form of the
    # like-named routine in `routines/math.mojo`, which takes the same
    # operands: `a.mul(b)` and `mul(a, b)` are one function reached two ways.

    def mul[
        d: DType, origin: Origin, //
    ](self, other: MatrixView[Self.T, origin]) raises -> Matrix[
        Scalar[d]
    ] where (Self.T == Scalar[d]):
        """Multiplies element by element (the Hadamard product)."""
        return linamo.routines.math.mul(
            self._simd_view[d](), other._as_simd[d]()
        )

    def mul[
        d: DType, //
    ](self, other: Self.ElementType) -> Matrix[Scalar[d]] where (
        Self.T == Scalar[d]
    ):
        """Multiplies every element by a scalar, the same as `*`."""
        return linamo.routines.math.scalar_mul(
            self._simd_view[d](), rebind[Scalar[d]](other)
        )

    def div[
        d: DType, origin: Origin, //
    ](self, other: MatrixView[Self.T, origin]) raises -> Matrix[
        Scalar[d]
    ] where (Self.T == Scalar[d]):
        """Divides element by element."""
        return linamo.routines.math.div(
            self._simd_view[d](), other._as_simd[d]()
        )

    def div[
        d: DType, //
    ](self, other: Self.ElementType) -> Matrix[Scalar[d]] where (
        Self.T == Scalar[d]
    ):
        """Divides every element by a scalar, the same as `/`."""
        return linamo.routines.math.scalar_div(
            self._simd_view[d](), rebind[Scalar[d]](other)
        )

    def pow[
        d: DType, origin: Origin, //
    ](self, other: MatrixView[Self.T, origin]) raises -> Matrix[
        Scalar[d]
    ] where (Self.T == Scalar[d]):
        """Raises each element to the matching element of `other`."""
        return linamo.routines.math.pow(
            self._simd_view[d](), other._as_simd[d]()
        )

    def pow[
        d: DType, //
    ](self, other: Self.ElementType) -> Matrix[Scalar[d]] where (
        Self.T == Scalar[d]
    ):
        """Raises every element to a scalar power."""
        return linamo.routines.math.scalar_pow(
            self._simd_view[d](), rebind[Scalar[d]](other)
        )

    # ===------------------------------------------------------------------ ===#
    # Scalar operands for the arithmetic dunders
    # ===------------------------------------------------------------------ ===#
    # `A + 2.0` and friends. A scalar is not a matrix and does not convert to
    # one, so it needs its own overload; the reflected forms below cover the
    # `2.0 + A` direction.

    def __add__[
        d: DType, //
    ](self, other: Self.ElementType) -> Matrix[Scalar[d]] where (
        Self.T == Scalar[d]
    ):
        """Adds a scalar to every element."""
        return linamo.routines.math.scalar_add(
            self._simd_view[d](), rebind[Scalar[d]](other)
        )

    def __sub__[
        d: DType, //
    ](self, other: Self.ElementType) -> Matrix[Scalar[d]] where (
        Self.T == Scalar[d]
    ):
        """Subtracts a scalar from every element."""
        return linamo.routines.math.scalar_sub(
            self._simd_view[d](), rebind[Scalar[d]](other)
        )

    def __mul__[
        d: DType, //
    ](self, other: Self.ElementType) -> Matrix[Scalar[d]] where (
        Self.T == Scalar[d]
    ):
        """Multiplies every element by a scalar."""
        return linamo.routines.math.scalar_mul(
            self._simd_view[d](), rebind[Scalar[d]](other)
        )

    def __truediv__[
        d: DType, //
    ](self, other: Self.ElementType) -> Matrix[Scalar[d]] where (
        Self.T == Scalar[d]
    ):
        """Divides every element by a scalar."""
        return linamo.routines.math.scalar_div(
            self._simd_view[d](), rebind[Scalar[d]](other)
        )

    def __floordiv__[
        d: DType, //
    ](self, other: Self.ElementType) -> Matrix[Scalar[d]] where (
        Self.T == Scalar[d]
    ):
        """Floor-divides every element by a scalar."""
        return linamo.routines.math.scalar_floordiv(
            self._simd_view[d](), rebind[Scalar[d]](other)
        )

    def __mod__[
        d: DType, //
    ](self, other: Self.ElementType) -> Matrix[Scalar[d]] where (
        Self.T == Scalar[d]
    ):
        """Takes every element modulo a scalar."""
        return linamo.routines.math.scalar_mod(
            self._simd_view[d](), rebind[Scalar[d]](other)
        )

    def __pow__[
        d: DType, //
    ](self, exponent: Int) raises -> Matrix[Scalar[d]] where (
        Self.T == Scalar[d]
    ):
        """Raises the matrix to an integer power: `A ** 3` is `A @ A @ A`.

        `A ** 0` is the identity and a negative exponent inverts first, so
        `A ** -1` is `inv(A)`. The matrix must be square.
        """
        return linamo.routines.linalg.matrix_power(
            self._simd_view[d](), exponent
        )

    # ===------------------------------------------------------------------ ===#
    # floordiv and mod
    # ===------------------------------------------------------------------ ===#
    # Element-wise, and unambiguously so: neither has a linear-algebra reading
    # that `*` and `**` could be confused with.

    def __floordiv__[
        d: DType, origin: Origin, //
    ](self, other: MatrixView[Self.T, origin]) raises -> Matrix[
        Scalar[d]
    ] where (Self.T == Scalar[d]):
        """Performs element-wise floor division."""
        return linamo.routines.math.floordiv(
            self._simd_view[d](), other._as_simd[d]()
        )

    def __mod__[
        d: DType, origin: Origin, //
    ](self, other: MatrixView[Self.T, origin]) raises -> Matrix[
        Scalar[d]
    ] where (Self.T == Scalar[d]):
        """Performs element-wise modulo."""
        return linamo.routines.math.mod(
            self._simd_view[d](), other._as_simd[d]()
        )

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

    def __radd__[
        d: DType, //
    ](self, other: Self.ElementType) -> Matrix[Scalar[d]] where (
        Self.T == Scalar[d]
    ):
        """Adds every element of the matrix to a scalar (`scalar + mat`)."""
        return linamo.routines.math.scalar_add(
            self._simd_view[d](), rebind[Scalar[d]](other)
        )

    def __rmul__[
        d: DType, //
    ](self, other: Self.ElementType) -> Matrix[Scalar[d]] where (
        Self.T == Scalar[d]
    ):
        """Multiplies a scalar by every element (`scalar * mat`)."""
        return linamo.routines.math.scalar_mul(
            self._simd_view[d](), rebind[Scalar[d]](other)
        )

    def __rsub__[
        d: DType, //
    ](self, other: Self.ElementType) -> Matrix[Scalar[d]] where (
        Self.T == Scalar[d]
    ):
        """Subtracts every element of the matrix from a scalar (`scalar - mat`).
        """
        return linamo.routines.math.scalar_rsub(
            self._simd_view[d](), rebind[Scalar[d]](other)
        )

    def __rtruediv__[
        d: DType, //
    ](self, other: Self.ElementType) -> Matrix[Scalar[d]] where (
        Self.T == Scalar[d]
    ):
        """Divides a scalar by every element of the matrix (`scalar / mat`)."""
        return linamo.routines.math.scalar_rdiv(
            self._simd_view[d](), rebind[Scalar[d]](other)
        )

    # ===------------------------------------------------------------------ ===#
    # Arbitrary-precision operands
    # ===------------------------------------------------------------------ ===#
    # The same operators once more, for an element type that carries its
    # arithmetic in `decimo.Numeric` instead of in a vector instruction. Two
    # `where` clauses, one method name: `Self.T == Scalar[d]` selects the SIMD
    # kernels above, `conforms_to(Self.T, Numeric)` selects the loops in
    # `routines/math.mojo`, and no element type satisfies both. A user writing
    # `A + B` never learns which of the two they got.
    #
    # `//` and `%` have no counterpart here. `Numeric` closes over `+ - * /`
    # and nothing else, and `/` on an integral element truncates toward zero
    # the way `Int` does, so `//` would be a second name for the same
    # operation on `BigInt` and a missing one on `BigDecimal`. `**` does carry
    # over, since a matrix power is repeated multiplication.

    def __add__[
        origin: Origin, //
    ](self, other: MatrixView[Self.T, origin]) raises -> Matrix[
        Self.T
    ] where conforms_to(Self.T, Numeric):
        """Performs element-wise addition."""
        return linamo.routines.math.add(self, other)

    def __sub__[
        origin: Origin, //
    ](self, other: MatrixView[Self.T, origin]) raises -> Matrix[
        Self.T
    ] where conforms_to(Self.T, Numeric):
        """Performs element-wise subtraction."""
        return linamo.routines.math.sub(self, other)

    def __mul__[
        origin: Origin, //
    ](self, other: MatrixView[Self.T, origin]) raises -> Matrix[
        Self.T
    ] where conforms_to(Self.T, Numeric):
        """Performs matrix multiplication, the same as `@`."""
        return linamo.routines.math.matmul(self, other)

    def __matmul__[
        origin: Origin, //
    ](self, other: MatrixView[Self.T, origin]) raises -> Matrix[
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
        """Raises the matrix to an integer power, as on a scalar element type.

        `Comparable` joins `Numeric` in the clause because a negative exponent
        goes through `inv`, whose pivoting ranks candidates by magnitude.
        """
        return linamo.routines.linalg.matrix_power(self, exponent)

    def mul[
        origin: Origin, //
    ](self, other: MatrixView[Self.T, origin]) raises -> Matrix[
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
        origin: Origin, //
    ](self, other: MatrixView[Self.T, origin]) raises -> Matrix[
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
        """Adds a value to every element."""
        return linamo.routines.math.scalar_add(self, other)

    def __sub__(
        self, other: Self.ElementType
    ) raises -> Matrix[Self.T] where conforms_to(Self.T, Numeric):
        """Subtracts a value from every element."""
        return linamo.routines.math.scalar_sub(self, other)

    def __mul__(
        self, other: Self.ElementType
    ) raises -> Matrix[Self.T] where conforms_to(Self.T, Numeric):
        """Multiplies every element by a value."""
        return linamo.routines.math.scalar_mul(self, other)

    def __truediv__(
        self, other: Self.ElementType
    ) raises -> Matrix[Self.T] where conforms_to(Self.T, Numeric):
        """Divides every element by a value."""
        return linamo.routines.math.scalar_div(self, other)

    def __radd__(
        self, other: Self.ElementType
    ) raises -> Matrix[Self.T] where conforms_to(Self.T, Numeric):
        """Adds every element to a value (`value + mat`)."""
        return linamo.routines.math.scalar_add(self, other)

    def __rmul__(
        self, other: Self.ElementType
    ) raises -> Matrix[Self.T] where conforms_to(Self.T, Numeric):
        """Multiplies a value by every element (`value * mat`)."""
        return linamo.routines.math.scalar_mul(self, other)

    def __rsub__(
        self, other: Self.ElementType
    ) raises -> Matrix[Self.T] where conforms_to(Self.T, Numeric):
        """Subtracts every element from a value (`value - mat`)."""
        return linamo.routines.math.scalar_rsub(self, other)

    def __rtruediv__(
        self, other: Self.ElementType
    ) raises -> Matrix[Self.T] where conforms_to(Self.T, Numeric):
        """Divides a value by every element (`value / mat`)."""
        return linamo.routines.math.scalar_rdiv(self, other)

    # ===------------------------------------------------------------------ ===#
    # In-place operators
    # ===------------------------------------------------------------------ ===#
    # These write back through the matrix's own strides rather than allocating
    # a result, so a transposed or otherwise non-contiguous matrix keeps its
    # layout. There is no `MatrixView` counterpart: a view is generic over
    # `origin` and Mojo checks the body against the read-only instantiation
    # too, so no method writing through `self._data` can be defined on it. Use
    # the free functions in `routines/mutation.mojo` for views.
    #
    # Aliasing (`a += a[:, :]`) is rejected by the borrow checker, which will
    # not hand out a mutable reference to `a` while a view of it is live.

    def __iadd__[
        d: DType, origin: Origin, //
    ](mut self, other: MatrixView[Self.T, origin]) raises where (
        Self.T == Scalar[d]
    ):
        """In-place element-wise addition with another matrix or view."""
        linamo.routines.math._elementwise_inplace[func=Scalar[d].__add__](
            rebind[Matrix[Scalar[d]]](self), other._as_simd[d]()
        )

    def __iadd__[
        d: DType, //
    ](mut self, other: Self.ElementType) where Self.T == Scalar[d]:
        """In-place element-wise addition with a scalar."""
        linamo.routines.math._scalar_elementwise_inplace[
            func=Scalar[d].__add__
        ](rebind[Matrix[Scalar[d]]](self), rebind[Scalar[d]](other))

    def __isub__[
        d: DType, origin: Origin, //
    ](mut self, other: MatrixView[Self.T, origin]) raises where (
        Self.T == Scalar[d]
    ):
        """In-place element-wise subtraction with another matrix or view."""
        linamo.routines.math._elementwise_inplace[func=Scalar[d].__sub__](
            rebind[Matrix[Scalar[d]]](self), other._as_simd[d]()
        )

    def __isub__[
        d: DType, //
    ](mut self, other: Self.ElementType) where Self.T == Scalar[d]:
        """In-place element-wise subtraction with a scalar."""
        linamo.routines.math._scalar_elementwise_inplace[
            func=Scalar[d].__sub__
        ](rebind[Matrix[Scalar[d]]](self), rebind[Scalar[d]](other))

    def __imul__[
        d: DType, //
    ](mut self, other: Self.ElementType) where Self.T == Scalar[d]:
        """In-place element-wise multiplication with a scalar."""
        linamo.routines.math._scalar_elementwise_inplace[
            func=Scalar[d].__mul__
        ](rebind[Matrix[Scalar[d]]](self), rebind[Scalar[d]](other))

    def __itruediv__[
        d: DType, //
    ](mut self, other: Self.ElementType) where Self.T == Scalar[d]:
        """In-place element-wise division with a scalar."""
        linamo.routines.math._scalar_elementwise_inplace[
            func=Scalar[d].__truediv__
        ](rebind[Matrix[Scalar[d]]](self), rebind[Scalar[d]](other))

    def __ifloordiv__[
        d: DType, origin: Origin, //
    ](mut self, other: MatrixView[Self.T, origin]) raises where (
        Self.T == Scalar[d]
    ):
        """In-place element-wise floor division with another matrix or view."""
        linamo.routines.math._elementwise_inplace[func=Scalar[d].__floordiv__](
            rebind[Matrix[Scalar[d]]](self), other._as_simd[d]()
        )

    def __ifloordiv__[
        d: DType, //
    ](mut self, other: Self.ElementType) where Self.T == Scalar[d]:
        """In-place element-wise floor division with a scalar."""
        linamo.routines.math._scalar_elementwise_inplace[
            func=Scalar[d].__floordiv__
        ](rebind[Matrix[Scalar[d]]](self), rebind[Scalar[d]](other))

    def __imod__[
        d: DType, origin: Origin, //
    ](mut self, other: MatrixView[Self.T, origin]) raises where (
        Self.T == Scalar[d]
    ):
        """In-place element-wise modulo with another matrix or view."""
        linamo.routines.math._elementwise_inplace[func=Scalar[d].__mod__](
            rebind[Matrix[Scalar[d]]](self), other._as_simd[d]()
        )

    def __imod__[
        d: DType, //
    ](mut self, other: Self.ElementType) where Self.T == Scalar[d]:
        """In-place element-wise modulo with a scalar."""
        linamo.routines.math._scalar_elementwise_inplace[
            func=Scalar[d].__mod__
        ](rebind[Matrix[Scalar[d]]](self), rebind[Scalar[d]](other))

    # ===------------------------------------------------------------------ ===#
    # Comparison operators
    # ===------------------------------------------------------------------ ===#
    # These return an element-wise `Matrix[DType.bool]` mask, as in NumPy —
    # not a single `Bool`. `Matrix` therefore deliberately does not conform to
    # `EqualityComparable`; use `utils/test_utils.assert_matrices_equal` to ask
    # whether two matrices are wholly identical.

    def __lt__[
        d: DType, origin: Origin, //
    ](self, other: MatrixView[Self.T, origin]) raises -> Matrix[
        Scalar[DType.bool]
    ] where (Self.T == Scalar[d]):
        """Element-wise less-than comparison with another matrix or view."""
        return linamo.routines.logic.less(
            self._simd_view[d](), other._as_simd[d]()
        )

    def __lt__[
        d: DType, //
    ](self, other: Self.ElementType) -> Matrix[Scalar[DType.bool]] where (
        Self.T == Scalar[d]
    ):
        """Element-wise less-than comparison against a scalar."""
        return linamo.routines.logic.scalar_less(
            self._simd_view[d](), rebind[Scalar[d]](other)
        )

    def __le__[
        d: DType, origin: Origin, //
    ](self, other: MatrixView[Self.T, origin]) raises -> Matrix[
        Scalar[DType.bool]
    ] where (Self.T == Scalar[d]):
        """Element-wise less-than-or-equal comparison with another matrix or view.
        """
        return linamo.routines.logic.less_equal(
            self._simd_view[d](), other._as_simd[d]()
        )

    def __le__[
        d: DType, //
    ](self, other: Self.ElementType) -> Matrix[Scalar[DType.bool]] where (
        Self.T == Scalar[d]
    ):
        """Element-wise less-than-or-equal comparison against a scalar."""
        return linamo.routines.logic.scalar_less_equal(
            self._simd_view[d](), rebind[Scalar[d]](other)
        )

    def __gt__[
        d: DType, origin: Origin, //
    ](self, other: MatrixView[Self.T, origin]) raises -> Matrix[
        Scalar[DType.bool]
    ] where (Self.T == Scalar[d]):
        """Element-wise greater-than comparison with another matrix or view."""
        return linamo.routines.logic.greater(
            self._simd_view[d](), other._as_simd[d]()
        )

    def __gt__[
        d: DType, //
    ](self, other: Self.ElementType) -> Matrix[Scalar[DType.bool]] where (
        Self.T == Scalar[d]
    ):
        """Element-wise greater-than comparison against a scalar."""
        return linamo.routines.logic.scalar_greater(
            self._simd_view[d](), rebind[Scalar[d]](other)
        )

    def __ge__[
        d: DType, origin: Origin, //
    ](self, other: MatrixView[Self.T, origin]) raises -> Matrix[
        Scalar[DType.bool]
    ] where (Self.T == Scalar[d]):
        """Element-wise greater-than-or-equal comparison with another matrix or view.
        """
        return linamo.routines.logic.greater_equal(
            self._simd_view[d](), other._as_simd[d]()
        )

    def __ge__[
        d: DType, //
    ](self, other: Self.ElementType) -> Matrix[Scalar[DType.bool]] where (
        Self.T == Scalar[d]
    ):
        """Element-wise greater-than-or-equal comparison against a scalar."""
        return linamo.routines.logic.scalar_greater_equal(
            self._simd_view[d](), rebind[Scalar[d]](other)
        )

    def __eq__[
        d: DType, origin: Origin, //
    ](self, other: MatrixView[Self.T, origin]) raises -> Matrix[
        Scalar[DType.bool]
    ] where (Self.T == Scalar[d]):
        """Element-wise equality comparison with another matrix or view."""
        return linamo.routines.logic.equal(
            self._simd_view[d](), other._as_simd[d]()
        )

    def __eq__[
        d: DType, //
    ](self, other: Self.ElementType) -> Matrix[Scalar[DType.bool]] where (
        Self.T == Scalar[d]
    ):
        """Element-wise equality comparison against a scalar."""
        return linamo.routines.logic.scalar_equal(
            self._simd_view[d](), rebind[Scalar[d]](other)
        )

    def __ne__[
        d: DType, origin: Origin, //
    ](self, other: MatrixView[Self.T, origin]) raises -> Matrix[
        Scalar[DType.bool]
    ] where (Self.T == Scalar[d]):
        """Element-wise inequality comparison with another matrix or view."""
        return linamo.routines.logic.not_equal(
            self._simd_view[d](), other._as_simd[d]()
        )

    def __ne__[
        d: DType, //
    ](self, other: Self.ElementType) -> Matrix[Scalar[DType.bool]] where (
        Self.T == Scalar[d]
    ):
        """Element-wise inequality comparison against a scalar."""
        return linamo.routines.logic.scalar_not_equal(
            self._simd_view[d](), rebind[Scalar[d]](other)
        )
