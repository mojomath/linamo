"""
This module defines `MatrixAxisIter`, the axis iterator shared by `Matrix` and
`MatrixView`.
"""

from linamo.types.matrix_view import MatrixView


struct MatrixAxisIter[
    mut: Bool,
    //,
    dtype: DType,
    origin: Origin[mut=mut],
    axis: Int = 0,
    forward: Bool = True,
](ImplicitlyCopyable, Movable, Sized):
    """Walks a matrix one row (`axis=0`) or one column (`axis=1`) at a time.

    Each step yields a `MatrixView` onto the parent buffer rather than a copy,
    so a full pass allocates nothing. The views inherit the parent's `origin`,
    which means iterating a mutable matrix yields writable rows and iterating
    an immutable one yields read-only rows -- the borrow checker decides, not a
    runtime flag.

    The iterator is parameterised on `axis` and `forward` rather than hardwired
    to forward row order, because that is the traversal `apply_along_axis` will
    need once the reduction routines land. `__iter__` and `__reversed__` are
    just the four corners of this one struct.

    Parameters:
        mut: Whether the underlying reference is mutable.
        dtype: The data type of the matrix elements.
        origin: The origin of the matrix being iterated.
        axis: 0 to walk rows, 1 to walk columns.
        forward: True to walk from the first lane to the last, False to reverse.
    """

    comptime Element = MatrixView[Self.dtype, Self.origin]
    """Each step yields a view: a `1 x ncols` row or an `nrows x 1` column."""

    var _data: Span[Scalar[Self.dtype], Self.origin]
    """The parent buffer that every yielded view points into."""
    var _nrows: Int
    """The number of rows in the matrix being iterated."""
    var _ncols: Int
    """The number of columns in the matrix being iterated."""
    var _row_stride: Int
    """The row stride inherited from the parent."""
    var _col_stride: Int
    """The column stride inherited from the parent."""
    var _offset: Int
    """The parent's own starting offset, added to every yielded view."""
    var _index: Int
    """How many lanes have been consumed so far."""

    def __init__(out self, src: MatrixView[Self.dtype, Self.origin]):
        """Builds an iterator over `src`.

        Args:
            src: The view to walk. `Matrix` passes its own full-extent view, so
                both types share this one constructor.
        """
        self._data = src._data
        self._nrows = src.nrows()
        self._ncols = src.ncols()
        self._row_stride = src.row_stride()
        self._col_stride = src.col_stride()
        self._offset = src.offset()
        self._index = 0

    def lane_count(self) -> Int:
        """Returns the total number of lanes along the iterated axis."""
        comptime if Self.axis == 0:
            return self._nrows
        else:
            return self._ncols

    def __len__(self) -> Int:
        """Returns the number of lanes still to be yielded."""
        return self.lane_count() - self._index

    def __has_next__(self) -> Bool:
        """Returns True while lanes remain."""
        return self.__len__() > 0

    def __iter__(self) -> Self:
        """Returns itself, so an iterator is usable directly in a `for` loop."""
        return self.copy()

    def __next__(mut self) -> Self.Element:
        """Yields the next lane as a view and advances the cursor."""
        # The cursor always counts up; `forward` only changes which lane an
        # index maps to, which keeps `__len__` direction-agnostic.
        var lane = self._index
        comptime if not Self.forward:
            lane = self.lane_count() - 1 - self._index
        self._index += 1

        comptime if Self.axis == 0:
            return Self.Element(
                buffer=self._data,
                nrows=1,
                ncols=self._ncols,
                row_stride=self._row_stride,
                col_stride=self._col_stride,
                offset=self._offset + lane * self._row_stride,
            )
        else:
            return Self.Element(
                buffer=self._data,
                nrows=self._nrows,
                ncols=1,
                row_stride=self._row_stride,
                col_stride=self._col_stride,
                offset=self._offset + lane * self._col_stride,
            )
