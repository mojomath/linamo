"""
A tour of `Matrix`, the owning matrix type.

`Matrix` holds its own storage in a `List` and is the type you construct,
compute with and hand back. Everything that only *reads* a matrix - slicing,
iteration, the named routines - goes through `MatrixView` instead; see
`matrix_view.mojo` for that half of the API.

Run with:

```bash
pixi run mojo run -I src examples/matrix.mojo
```
"""

from linamo.prelude import *
from linamo.routines.math import add


def main() raises:
    creation()
    memory_layout()
    element_access()
    region_writes()
    arithmetic()
    comparison()
    iteration()
    shape_and_layout()
    linear_algebra()


# ===----------------------------------------------------------------------=== #
# Creation
# ===----------------------------------------------------------------------=== #


def creation() raises:
    print("=" * 80)
    print("CREATION")
    print("=" * 80)

    # From a nested list. The `order` argument decides how the elements are
    # laid out in memory, not how they are indexed: `m[i, j]` means the same
    # thing either way.
    var m1 = la.matrix[float64](
        [
            [1.1, 1.2, 1.3, 1.4],
            [2.1, 2.2, 2.3, 2.4],
            [3.1, 3.2, 3.3, 3.4],
        ],
        order="C",
    )
    print("From a nested list (row-major):\n", m1)

    # From a flat list plus a shape. This is the cheapest constructor: the
    # list is taken as-is and only the strides are computed.
    var m2 = la.matrix[int64](
        flat_list=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        nrows=3,
        ncols=4,
        order="C",
    )
    print("From a flat list, 3x4 row-major:\n", m2)

    var m3 = la.matrix[int64](
        flat_list=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        nrows=3,
        ncols=4,
        order="F",
    )
    print("The same flat list read column-major:\n", m3)

    # The filled constructors.
    print("zeros(2, 3):\n", la.zeros[float64](2, 3))
    print("ones(2, 3):\n", la.ones[float64](2, 3))
    print("full(2, 3, 7.5):\n", la.full[float64](2, 3, 7.5))
    print("eye(4) - the identity, built by name:\n", la.eye[float64](4))
    print("identity(3):\n", la.identity[float64](3))

    # `diag` goes both ways: a list becomes a diagonal matrix, and a matrix
    # gives back its diagonal.
    var d = la.diag[float64]([1.0, 2.0, 3.0])
    print("diag([1, 2, 3]):\n", d)
    var back = la.diag(d)
    print("diag(m) - the diagonal back out:", _join(back))

    # A copy is explicit. There is no implicit deep copy anywhere in Linamo.
    var m4 = m1.copy()
    m4[0, 0] = 99.0
    print("A copy is independent. Original m1[0, 0]:", m1[0, 0])
    print("                        Copy     m4[0, 0]:", m4[0, 0])


# ===----------------------------------------------------------------------=== #
# Memory layout
# ===----------------------------------------------------------------------=== #


def memory_layout() raises:
    print()
    print("=" * 80)
    print("MEMORY LAYOUT")
    print("=" * 80)

    var rows: List[List[Scalar[float64]]] = [
        [1.1, 1.2, 1.3],
        [2.1, 2.2, 2.3],
    ]

    var c = la.matrix[float64](rows.copy(), order="C")
    var f = la.matrix[float64](rows.copy(), order="F")

    # The two matrices are equal element by element but differ in the buffer.
    print("Row-major m[i, j]:\n", c)
    print("Column-major m[i, j] - identical:\n", f)
    print("Row-major buffer:   ", c.data)
    print("Column-major buffer:", f.data)

    # Strides say how far to step in the buffer for one row / one column.
    print(
        "Row-major strides:    row =",
        c.get_row_stride(),
        " col =",
        c.get_col_stride(),
    )
    print(
        "Column-major strides: row =",
        f.get_row_stride(),
        " col =",
        f.get_col_stride(),
    )

    # Contiguity is what the SIMD paths dispatch on.
    print("Row-major   is_c_contiguous:", c.is_c_contiguous())
    print("Row-major   is_f_contiguous:", c.is_f_contiguous())
    print("Column-major is_c_contiguous:", f.is_c_contiguous())
    print("Column-major is_f_contiguous:", f.is_f_contiguous())

    print("Shape:", c.get_nrows(), "x", c.get_ncols(), " size:", c.get_size())
    # `len()` is the row count, so that it agrees with what iteration yields.
    print("len(m) - the number of rows:", len(c))


# ===----------------------------------------------------------------------=== #
# Element access
# ===----------------------------------------------------------------------=== #


def element_access() raises:
    print()
    print("=" * 80)
    print("ELEMENT ACCESS")
    print("=" * 80)

    var m = la.matrix[int64](
        flat_list=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        nrows=3,
        ncols=4,
    )
    print("A 3x4 matrix:\n", m)

    # Reading.
    print("m[1, 2] =", m[1, 2])

    # Writing. There is no `__setitem__` on `Matrix`; `m[i, j]` returns a
    # reference, and assigning to it writes through. See the note in
    # `types/matrix.mojo` for why it has to be spelled this way.
    m[1, 2] = 999
    print("After m[1, 2] = 999:\n", m)

    # A `ref` binding names that same element, and inherits the mutability of
    # what it came from - so this needs `m` to be a `var`, as it is here.
    ref e = m[0, 0]
    e += 1000
    print("After `ref e = m[0, 0]` and `e += 1000`, m[0, 0] =", m[0, 0])

    # Bounds are checked, and the traceback names the file and line.
    try:
        _ = m[3, 0]
    except e:
        print("m[3, 0] raises:")
        print(e)

    # `get_unsafe` skips the check. Only the `-D ASSERT=all` build catches a
    # bad index here, so reach for it only in code that has already bounded
    # the indices itself.
    print("m.get_unsafe(2, 3) =", m.get_unsafe(2, 3))

    # SIMD access: a whole run of a row at once.
    print("m.load[4](0, 0) =", m.load[4](0, 0))
    m.store[4](2, 0, SIMD[int64, 4](70, 80, 90, 100))
    print("After m.store[4](2, 0, [70, 80, 90, 100]):\n", m)


# ===----------------------------------------------------------------------=== #
# Region writes
# ===----------------------------------------------------------------------=== #


def region_writes() raises:
    print()
    print("=" * 80)
    print("REGION WRITES")
    print("=" * 80)

    var m = la.zeros[int64](5, 5)
    print("A 5x5 matrix of zeros:\n", m)

    # `m[a:b, c:d] = src` cannot be spelled in Mojo 1.0 without breaking
    # ordinary slicing expressions, so the region write is a named method.
    m.fill(Slice(1, 4), Slice(1, 4), 7)
    print("After m.fill(1:4, 1:4, 7):\n", m)

    var block = la.matrix[int64]([[1, 2], [3, 4]])
    m.assign(Slice(0, 2), Slice(0, 2), block.view())
    print("After m.assign(0:2, 0:2, [[1, 2], [3, 4]]):\n", m)

    # A strided region works too - here every other row and column.
    m.fill(Slice(0, 5, 2), Slice(0, 5, 2), -1)
    print("After m.fill(0:5:2, 0:5:2, -1):\n", m)


# ===----------------------------------------------------------------------=== #
# Arithmetic
# ===----------------------------------------------------------------------=== #


def arithmetic() raises:
    print()
    print("=" * 80)
    print("ARITHMETIC")
    print("=" * 80)

    var a = la.matrix[float64]([[1.0, 2.0], [3.0, 4.0]])
    var b = la.matrix[float64]([[10.0, 20.0], [30.0, 40.0]])

    print("a:\n", a)
    print("b:\n", b)

    # Elementwise, matrix by matrix.
    print("a + b:\n", a + b)
    print("a - b:\n", a - b)
    print("a * b - elementwise, not matrix multiplication:\n", a * b)
    print("b / a:\n", b / a)

    # Matrix multiplication has its own operator.
    print("a @ b:\n", a @ b)

    # Scalar operands, on either side.
    print("a + 100:\n", a + 100.0)
    print("100 - a - the reflected form:\n", 100.0 - a)
    print("a * 2:\n", a * 2.0)
    print("a ** 2:\n", a**2.0)

    # Integer-flavoured operators.
    var i = la.matrix[int64]([[17, 23], [31, 47]])
    print("i // 5:\n", i // 5)
    print("i % 5:\n", i % 5)

    # In-place operators mutate the receiver rather than allocating.
    var c = a.copy()
    c += b
    print("After c += b:\n", c)
    c *= 2.0
    print("After c *= 2:\n", c)
    c -= b
    print("After c -= b:\n", c)

    # A shape mismatch is an error, not a silent broadcast.
    try:
        _ = a + la.ones[float64](3, 3)
    except e:
        print("Adding a 2x2 and a 3x3 raises:")
        print(e)

    # The named routines are the same operations under another spelling, and
    # are what the operators dispatch to.
    print("add(a, b), the routine behind `a + b`:\n", add(a, b))


# ===----------------------------------------------------------------------=== #
# Comparison
# ===----------------------------------------------------------------------=== #


def comparison() raises:
    print()
    print("=" * 80)
    print("COMPARISON")
    print("=" * 80)

    var a = la.matrix[int64]([[1, 5, 3], [7, 2, 9]])
    var b = la.matrix[int64]([[4, 4, 4], [4, 4, 4]])

    print("a:\n", a)
    print("b:\n", b)

    # Every comparison yields a boolean matrix of the same shape.
    print("a > b:\n", a > b)
    print("a <= b:\n", a <= b)
    print("a == b:\n", a == b)
    print("a != b:\n", a != b)

    # Against a scalar.
    print("a > 4:\n", a > 4)
    print("a == 5:\n", a == 5)


# ===----------------------------------------------------------------------=== #
# Iteration
# ===----------------------------------------------------------------------=== #


def iteration() raises:
    print()
    print("=" * 80)
    print("ITERATION")
    print("=" * 80)

    var m = la.matrix[int64](
        flat_list=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        nrows=3,
        ncols=4,
    )
    print("A 3x4 matrix:\n", m)

    # Iterating a matrix walks its rows, and each row arrives as a view - no
    # copying, so a row of a million columns costs the same as a row of four.
    print("for row in m:")
    for row in m:
        print("  ", row.__str__())

    print("m.cols():")
    for col in m.cols():
        print("  ", col.__str__())

    print("m.rows[False]() - last row first:")
    for row in m.rows[False]():
        print("  ", row.__str__())

    # Rows yielded by iteration are read-only whatever the receiver was; use
    # `rows_mut` from `linamo.routines.mutation` to walk a matrix writably.
    # `matrix_view.mojo` shows that.


# ===----------------------------------------------------------------------=== #
# Shape and layout
# ===----------------------------------------------------------------------=== #


def shape_and_layout() raises:
    print()
    print("=" * 80)
    print("SHAPE AND LAYOUT")
    print("=" * 80)

    var m = la.matrix[float64]([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    print("A 2x3 matrix:\n", m)

    # `order` is the *index* order, as in NumPy: "C" walks row by row, "F"
    # column by column. The result is a new matrix, always C-contiguous.
    print("reshape(m, 3, 2):\n", la.reshape(m, 3, 2))
    print('reshape(m, 3, 2, "F"):\n', la.reshape(m, 3, 2, "F"))
    print("flatten(m):\n", la.flatten(m))

    # Where the elements sit in memory is a separate question.
    print(
        'contiguous(m, "F") - same elements, F layout:\n', la.contiguous(m, "F")
    )

    # `resize` returns a new matrix. It cannot grow `m` in place: that would
    # reallocate the underlying `List` and dangle every live view of it.
    print("resize(m, 3, 3) - zero-padded:\n", la.resize(m, 3, 3))

    # These two allocate nothing. The result views `m`'s own buffer and
    # carries its origin, so `m` is kept alive as long as the view is.
    var v = la.reshape_view(m, 3, 2)
    print("reshape_view(m, 3, 2) - no copy:\n", v)
    m[0, 0] = 99.0
    print("after m[0, 0] = 99.0, the view sees it:\n", v)

    var row = la.matrix[float64]([[1.0, 2.0, 3.0]])
    # A stretched dimension gets a stride of 0, so every row is the same one.
    print("broadcast_to(row, 3, 3) - no copy:\n", la.broadcast_to(row, 3, 3))

    print("m.astype[int32]():\n", m.astype[int32]())


# ===----------------------------------------------------------------------=== #
# Linear algebra
# ===----------------------------------------------------------------------=== #


def linear_algebra() raises:
    print()
    print("=" * 80)
    print("LINEAR ALGEBRA")
    print("=" * 80)

    var a = la.matrix[float64](
        [
            [4.0, 3.0, 2.0],
            [3.0, 5.0, 1.0],
            [2.0, 1.0, 6.0],
        ]
    )
    print("A symmetric positive-definite matrix a:\n", a)

    print("transpose(a):\n", la.transpose(a))
    print("trace(a):", la.trace(a))
    print("det(a):", la.det(a))

    print("inv(a):\n", la.inv(a))
    print("a @ inv(a) - the identity, up to rounding:\n", a @ la.inv(a))

    # Ax = b.
    var b = la.matrix[float64]([[1.0], [2.0], [3.0]])
    var x = la.solve(a, b)
    print("solve(a, b) where b = [1, 2, 3]^T:\n", x)
    print("a @ x - should be b again:\n", a @ x)

    # Decompositions. `lu` returns L, U and the permutation vector.
    var lu_result = la.lu(a)
    print("lu(a) - L:\n", lu_result[0])
    print("lu(a) - U:\n", lu_result[1])
    print("lu(a) - permutation:", _join_int(lu_result[2]))

    var qr_result = la.qr(a)
    print("qr(a) - Q:\n", qr_result[0])
    print("qr(a) - R:\n", qr_result[1])

    print("cholesky(a) - the lower factor L with a = L L^T:\n", la.cholesky(a))

    # Least squares on an overdetermined system.
    var design = la.matrix[float64](
        [[1.0, 1.0], [1.0, 2.0], [1.0, 3.0], [1.0, 4.0]]
    )
    var obs = la.matrix[float64]([[2.1], [3.9], [6.2], [7.8]])
    print("lstsq(design, obs) - intercept and slope:\n", la.lstsq(design, obs))


# ===----------------------------------------------------------------------=== #
# Small helpers, so the printing above stays readable
# ===----------------------------------------------------------------------=== #


def _join[dtype: DType](values: List[Scalar[dtype]]) -> String:
    var out = String("[")
    for i in range(len(values)):
        if i > 0:
            out += ", "
        out += String(values[i])
    return out + "]"


def _join_int(values: List[Int]) -> String:
    var out = String("[")
    for i in range(len(values)):
        if i > 0:
            out += ", "
        out += String(values[i])
    return out + "]"
