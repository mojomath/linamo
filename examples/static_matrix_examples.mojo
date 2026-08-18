"""
A tour of `StaticMatrix`, the fixed-shape matrix.

`StaticMatrix[dtype, nrows, ncols]` carries its shape in the type, and holds
its elements in a single SIMD register rather than a heap `List`. Nothing is
allocated, every index is a compile-time-known stride away, and the whole
matrix can move through registers - which is what makes it worth having for
the small fixed sizes that show up in graphics, geometry and kernels.

The cost is that the shape is frozen at compile time and the buffer is padded
to powers of two, so a 6x5 matrix reserves 8x8 elements. That padding is what
lets indexing be a shift instead of a multiply.

Run with:

```bash
pixi run mojo run -I src examples/static_matrix.mojo
```
"""

import linamo as la


def main() raises:
    creation()
    padding()
    element_access()
    arithmetic()


# ===----------------------------------------------------------------------=== #
# Creation
# ===----------------------------------------------------------------------=== #


def creation() raises:
    print("=" * 80)
    print("CREATION")
    print("=" * 80)

    # The shape comes first, as compile-time parameters, then the dtype. Note
    # that the *type* spells them the other way round - `StaticMatrix[dtype,
    # nrows, ncols]` - so `smatrix[3, 4, la.float64]` builds a
    # `StaticMatrix[la.float64, 3, 4]`.
    var m1 = la.smatrix[3, 4, la.float64](
        [
            [1.1, 1.2, 1.3, 1.4],
            [2.1, 2.2, 2.3, 2.4],
            [3.1, 3.2, 3.3, 3.4],
        ],
    )
    print("From a nested list:\n", m1)

    # From a flat list, read row by row. The list length must match
    # `nrows * ncols` exactly; the padding is filled in for you.
    var m2 = la.smatrix[3, 4, la.int64](
        flat_list=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    )
    print("From a flat list:\n", m2)

    # A copy is explicit here too, and is a register move rather than an
    # allocation.
    var m3 = m2.copy()
    print("A copy:\n", m3)


# ===----------------------------------------------------------------------=== #
# Padding
# ===----------------------------------------------------------------------=== #


def padding() raises:
    print()
    print("=" * 80)
    print("PADDING AND LAYOUT")
    print("=" * 80)

    # A 3x5 is the interesting case: neither dimension is a power of two, so
    # both get rounded up - to a 4x8 buffer.
    var m = la.smatrix[3, 5, la.int64](
        flat_list=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    )

    print("Logical shape:", m.get_nrows(), "x", m.get_ncols())
    print(
        "Buffer shape: ",
        la.StaticMatrix[la.int64, 3, 5].BUFFER_ROW_LEN,
        "x",
        la.StaticMatrix[la.int64, 3, 5].BUFFER_COL_LEN,
    )
    print("Strides: row =", m.get_row_stride(), " col =", m.get_col_stride())

    # The raw register, padding and all: three zeros after each row of five,
    # then a whole unused fourth row. That is the price of a stride that is a
    # shift rather than a multiply.
    print("The underlying SIMD buffer:\n", m.data)

    # A shape that is already a power of two wastes nothing.
    var square = la.smatrix[4, 4, la.int64](
        flat_list=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    )
    print("A 4x4 needs no padding:\n", square.data)

    print("is_c_contiguous:", m.is_c_contiguous())
    print("is_row_contiguous:", m.is_row_contiguous())


# ===----------------------------------------------------------------------=== #
# Element access
# ===----------------------------------------------------------------------=== #


def element_access() raises:
    print()
    print("=" * 80)
    print("ELEMENT ACCESS")
    print("=" * 80)

    var m = la.smatrix[3, 4, la.int64](
        flat_list=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    )
    print("A 3x4 static matrix:\n", m)

    # Indexing reads out of the register. It returns a value, not a
    # reference - there is no memory to point at.
    print("m[0, 0] =", m[0, 0])
    print("m[1, 2] =", m[1, 2])
    print("m[2, 3] =", m[2, 3])

    print("Element count:", m.get_size())


# ===----------------------------------------------------------------------=== #
# Arithmetic
# ===----------------------------------------------------------------------=== #


def arithmetic() raises:
    print()
    print("=" * 80)
    print("ARITHMETIC")
    print("=" * 80)

    var a = la.smatrix[3, 3, la.float64](
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ],
    )
    var b = la.smatrix[3, 3, la.float64](
        [
            [10.0, 20.0, 30.0],
            [40.0, 50.0, 60.0],
            [70.0, 80.0, 90.0],
        ],
    )

    print("a:\n", a)
    print("b:\n", b)

    # Addition is one SIMD instruction on the whole matrix, padding included.
    print("a + b:\n", a + b)

    # Matrix multiplication. The shapes are checked at compile time, so a
    # mismatched `@` will not build rather than raising at run time.
    var c = la.smatrix[3, 2, la.float64](
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
    )
    print("a @ c - a 3x3 times a 3x2:\n", a @ c)
