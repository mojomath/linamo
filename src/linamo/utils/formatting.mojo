"""
Layout of the printed form of a matrix.

`Matrix`, `MatrixView` and `StaticMatrix` all print through this module, so the
three renderings differ only in the header line each writes first. A grid is
laid out in three passes: every element that will be shown is rendered to text,
each column is measured, and the cells are padded so that the decimal points of
a column stand in one vertical line. That is the alignment NumPy uses, and the
only one that stays readable when a column holds `1648` next to `3.25`.

A grid never grows without bound. Rows are elided by element count, columns by
line width, and an element with a long fractional tail is trimmed rather than
rounded, so a shortened value carries a visible mark and its magnitude is never
misreported.

The comptime values below are the whole appearance contract. They are aliases
rather than fields because Mojo has no global variables yet; when it does, they
are what a configuration type would carry.
"""


# ===----------------------------------------------------------------------===#
# Appearance
# ===----------------------------------------------------------------------===#

comptime MAX_LINE_WIDTH = 88
"""The widest line a grid may occupy, brackets and padding included."""

comptime PRINT_THRESHOLD = 1000
"""The element count above which rows and columns are elided by count."""

comptime EDGE_ITEMS = 3
"""Rows, and columns, kept at each end of an elided dimension."""

comptime MIN_COLS_SHOWN = 3
"""Columns shown even when they do not fit `MAX_LINE_WIDTH`.

An arbitrary-precision element can be wider than a line on its own. Without a
floor the width rule would answer such a matrix with a single column, which
says less about it than an overlong line does.
"""

comptime MAX_FRAC_DIGITS = 8
"""Digits kept after the decimal point before `TRIM_MARK` takes over."""

comptime COLUMN_GAP = "  "
"""What separates two cells of a row."""

comptime EDGE_PAD = " "
"""What separates a row's brackets from its first and last cell."""

comptime ELISION = "..."
"""The stand-in for the rows and columns a grid does not show."""

comptime TRIM_MARK = "…"
"""The mark that ends a fractional part cut short by `MAX_FRAC_DIGITS`."""


# ===----------------------------------------------------------------------===#
# Cells
# ===----------------------------------------------------------------------===#


def display_width[o: Origin[mut=False], //](text: StringSpan[o]) -> Int:
    """Returns the number of columns a string occupies when printed.

    Args:
        text: The string to measure.

    Returns:
        The code point count, which is the printed width for the digits,
        signs and separators a number is made of. `TRIM_MARK` is one code
        point and three bytes, so a byte count would misalign every column
        holding a trimmed value.
    """
    return len(text.codepoints())


def _dot(cell: String) -> Int:
    """Returns the byte offset of the decimal point, or the length if none."""
    var found = cell.find(".")
    if found < 0:
        return cell.byte_length()
    return found


def trim_fraction(var cell: String) -> String:
    """Shortens a long fractional part and marks the cut.

    Only digits after the decimal point are dropped. The integer part is left
    alone, so the magnitude of the value printed is the magnitude of the value
    held, and a cell that ends in `TRIM_MARK` is visibly an abridged reading
    rather than a rounded one. A value written in exponent form is left whole,
    since cutting it would strip the exponent.

    Args:
        cell: The rendered element.

    Returns:
        The text to lay out in the grid.
    """
    var dot = cell.find(".")
    if dot < 0:
        return cell^
    var fraction = cell[byte = dot + 1 :]
    if not fraction.is_ascii_digit():
        return cell^
    if fraction.byte_length() <= MAX_FRAC_DIGITS:
        return cell^
    return String(cell[byte = 0 : dot + 1 + MAX_FRAC_DIGITS], TRIM_MARK)


def _pad(cell: String, int_width: Int, frac_width: Int) -> String:
    """Pads one cell so its decimal point lands on the column's."""
    var dot = _dot(cell)
    var head = String(cell[byte=0:dot])
    var tail = String(cell[byte=dot:])
    return String(
        String(" ") * (int_width - display_width(head)),
        head,
        tail,
        String(" ") * (frac_width - display_width(tail)),
    )


# ===----------------------------------------------------------------------===#
# Planning
# ===----------------------------------------------------------------------===#


def plan_indices(n: Int, elide: Bool) -> List[Int]:
    """Returns the row or column indices to render, in order.

    Args:
        n: The length of the dimension.
        elide: Whether the matrix is large enough for the count rule to apply.

    Returns:
        The indices to read, with `-1` marking the point where the ones left
        out are replaced by `ELISION`. All of them, in order, when nothing is
        left out.
    """
    var indices = List[Int]()
    if not elide or n <= 2 * EDGE_ITEMS:
        for i in range(n):
            indices.append(i)
        return indices^
    for i in range(EDGE_ITEMS):
        indices.append(i)
    indices.append(-1)
    for i in range(n - EDGE_ITEMS, n):
        indices.append(i)
    return indices^


def gap_position(indices: List[Int]) -> Int:
    """Returns where `plan_indices` put the elision, or `-1` if it put none."""
    for i in range(len(indices)):
        if indices[i] < 0:
            return i
    return -1


def elides(nrows: Int, ncols: Int) -> Bool:
    """Returns whether a matrix of this size is too large to print whole."""
    return nrows * ncols > PRINT_THRESHOLD


# ===----------------------------------------------------------------------===#
# The grid
# ===----------------------------------------------------------------------===#


def _row_width(widths: List[Int], columns: List[Int], gap_width: Int) -> Int:
    """Returns the printed width of a row laid out over these columns."""
    var total = 2 * display_width("[[") + 2 * display_width(EDGE_PAD)
    total += display_width(COLUMN_GAP) * (len(columns) - 1)
    for c in columns:
        total += gap_width if c < 0 else widths[c]
    return total


def _take_edges(values: List[Int], count: Int, gap: Bool) -> List[Int]:
    """Returns `count` columns taken from both ends, `-1` marking the middle.

    An odd count leans left, so the floor below can be honoured exactly rather
    than rounded up to the next even number.
    """
    if not gap:
        return values.copy()
    var chosen = List[Int]()
    for i in range((count + 1) // 2):
        chosen.append(values[i])
    chosen.append(-1)
    for i in range(len(values) - count // 2, len(values)):
        chosen.append(values[i])
    return chosen^


def write_grid[
    W: Writer, //
](mut writer: W, cells: List[List[String]], gap_col: Int):
    """Writes a laid-out grid of already-rendered cells.

    Args:
        writer: Where the grid goes.
        cells: One list per row to show, one entry per column to show. An
            empty list is a row left out and prints as `ELISION`; an entry
            whose column index was `-1` holds `ELISION` already.
        gap_col: The position in each row that `plan_indices` marked, or `-1`.

    The grid ends without a trailing newline, so `print` puts exactly one blank
    line's worth of separation after a matrix rather than two.
    """
    var ncols = 0
    for row in cells:
        if len(row) > 0:
            ncols = len(row)
            break
    if ncols == 0:
        writer.write("[]")
        return

    # Two widths per column, not one: the integer part is measured to the left
    # of the decimal point and the fraction to the right, and padding each
    # separately is what lines the points up.
    var int_widths = List[Int]()
    var frac_widths = List[Int]()
    for _ in range(ncols):
        int_widths.append(0)
        frac_widths.append(0)
    for row in cells:
        if len(row) == 0:
            continue
        for c in range(ncols):
            var dot = _dot(row[c])
            var head = display_width(row[c][byte=0:dot])
            var tail = display_width(row[c][byte=dot:])
            if head > int_widths[c]:
                int_widths[c] = head
            if tail > frac_widths[c]:
                frac_widths[c] = tail
    var widths = List[Int]()
    for c in range(ncols):
        widths.append(int_widths[c] + frac_widths[c])

    # Which columns survive. The count rule has already run; this is the
    # width rule, and `MIN_COLS_SHOWN` is a floor it may not go under even
    # where honouring it overruns the line.
    var values = List[Int]()
    for c in range(ncols):
        if c != gap_col:
            values.append(c)
    var kept = len(values)
    var floor = MIN_COLS_SHOWN if MIN_COLS_SHOWN < kept else kept

    var shown = List[Int]()
    var count = kept
    while count >= floor:
        shown = _take_edges(values, count, gap_col >= 0 or count < kept)
        if _row_width(widths, shown, display_width(ELISION)) <= MAX_LINE_WIDTH:
            break
        count -= 1
    if count < floor:
        shown = _take_edges(values, floor, gap_col >= 0 or floor < kept)

    for r in range(len(cells)):
        if r > 0:
            writer.write("\n")
        if len(cells[r]) == 0:
            writer.write(EDGE_PAD, ELISION)
            continue
        writer.write("[[" if r == 0 else " [", EDGE_PAD)
        for i in range(len(shown)):
            if i > 0:
                writer.write(COLUMN_GAP)
            var c = shown[i]
            if c < 0:
                writer.write(ELISION)
            else:
                writer.write(_pad(cells[r][c], int_widths[c], frac_widths[c]))
        writer.write(EDGE_PAD, "]")
    writer.write("]")


def write_header[
    W: Writer, //
](
    mut writer: W,
    name: String,
    element: String,
    nrows: Int,
    ncols: Int,
    row_stride: Int,
    col_stride: Int,
    offset: Int,
):
    """Writes the line that names a matrix above its grid.

    Strides and offset appear only when they are not the ones a freshly built
    matrix has. A plain matrix prints without them; a view, a transpose or an
    F-order matrix announces its layout, which is exactly when the layout is
    the thing worth knowing.

    Args:
        writer: Where the header goes.
        name: The name of the type.
        element: The name of the element type.
        nrows: The number of rows.
        ncols: The number of columns.
        row_stride: The step between two rows.
        col_stride: The step between two columns.
        offset: The first element's position in the underlying buffer.
    """
    writer.write(name, "[", element, "] ", nrows, "x", ncols)
    if row_stride != ncols or col_stride != 1:
        writer.write(", strides (", row_stride, ", ", col_stride, ")")
    if offset != 0:
        writer.write(", offset ", offset)
