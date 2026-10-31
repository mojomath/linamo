"""
Tests for matrix arithmetic operations (elementwise and matmul).
"""

import std.testing as testing
import linamo as la
from linamo.routines.math import add, sub, mul, div, matmul


# ===----------------------------------------------------------------------===#
# StaticMatrix arithmetic
# ===----------------------------------------------------------------------===#


def test_static_add() raises:
    """Test element-wise addition of two static matrices."""
    var a = la.smatrix[2, 3, Float64]([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    var b = la.smatrix[2, 3, Float64]([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]])
    var c = add(a, b)
    testing.assert_equal(c[0, 0], 11.0)
    testing.assert_equal(c[0, 1], 22.0)
    testing.assert_equal(c[0, 2], 33.0)
    testing.assert_equal(c[1, 0], 44.0)
    testing.assert_equal(c[1, 1], 55.0)
    testing.assert_equal(c[1, 2], 66.0)


def test_static_add_dunder() raises:
    """Test element-wise addition using __add__ operator."""
    var a = la.smatrix[2, 2, Float64]([[1.0, 2.0], [3.0, 4.0]])
    var b = la.smatrix[2, 2, Float64]([[5.0, 6.0], [7.0, 8.0]])
    var c = a + b
    testing.assert_equal(c[0, 0], 6.0)
    testing.assert_equal(c[0, 1], 8.0)
    testing.assert_equal(c[1, 0], 10.0)
    testing.assert_equal(c[1, 1], 12.0)


def test_static_sub() raises:
    """Test element-wise subtraction of two static matrices."""
    var a = la.smatrix[2, 2, Float64]([[10.0, 20.0], [30.0, 40.0]])
    var b = la.smatrix[2, 2, Float64]([[1.0, 2.0], [3.0, 4.0]])
    var c = sub(a, b)
    testing.assert_equal(c[0, 0], 9.0)
    testing.assert_equal(c[0, 1], 18.0)
    testing.assert_equal(c[1, 0], 27.0)
    testing.assert_equal(c[1, 1], 36.0)


def test_static_mul() raises:
    """Test element-wise multiplication of two static matrices."""
    var a = la.smatrix[2, 2, Float64]([[2.0, 3.0], [4.0, 5.0]])
    var b = la.smatrix[2, 2, Float64]([[10.0, 20.0], [30.0, 40.0]])
    var c = mul(a, b)
    testing.assert_equal(c[0, 0], 20.0)
    testing.assert_equal(c[0, 1], 60.0)
    testing.assert_equal(c[1, 0], 120.0)
    testing.assert_equal(c[1, 1], 200.0)


def test_static_div() raises:
    """Test element-wise division of two static matrices."""
    var a = la.smatrix[2, 2, Float64]([[10.0, 20.0], [30.0, 40.0]])
    var b = la.smatrix[2, 2, Float64]([[2.0, 4.0], [5.0, 8.0]])
    var c = div(a, b)
    testing.assert_equal(c[0, 0], 5.0)
    testing.assert_equal(c[0, 1], 5.0)
    testing.assert_equal(c[1, 0], 6.0)
    testing.assert_equal(c[1, 1], 5.0)


def test_static_matmul() raises:
    """Test static matrix multiplication."""
    var a = la.smatrix[2, 3, Float64]([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    var b = la.smatrix[3, 2, Float64]([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]])
    var c = matmul(a, b)
    # c[0,0] = 1*7 + 2*9 + 3*11 = 7+18+33 = 58
    # c[0,1] = 1*8 + 2*10 + 3*12 = 8+20+36 = 64
    # c[1,0] = 4*7 + 5*9 + 6*11 = 28+45+66 = 139
    # c[1,1] = 4*8 + 5*10 + 6*12 = 32+50+72 = 154
    testing.assert_equal(c[0, 0], 58.0)
    testing.assert_equal(c[0, 1], 64.0)
    testing.assert_equal(c[1, 0], 139.0)
    testing.assert_equal(c[1, 1], 154.0)


def test_static_matmul_dunder() raises:
    """Test static matrix multiplication using @ operator."""
    var a = la.smatrix[2, 2, Float64]([[1.0, 0.0], [0.0, 1.0]])
    var b = la.smatrix[2, 2, Float64]([[5.0, 6.0], [7.0, 8.0]])
    var c = a @ b
    # Identity @ B = B
    testing.assert_equal(c[0, 0], 5.0)
    testing.assert_equal(c[0, 1], 6.0)
    testing.assert_equal(c[1, 0], 7.0)
    testing.assert_equal(c[1, 1], 8.0)


# ===----------------------------------------------------------------------===#
# Dynamic Matrix arithmetic
# ===----------------------------------------------------------------------===#


def test_dynamic_matmul() raises:
    """Test dynamic matrix multiplication."""
    var a = la.matrix[Float64]([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    var b = la.matrix[Float64]([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]])
    var c = matmul(a, b)
    testing.assert_equal(c[0, 0], 58.0)
    testing.assert_equal(c[0, 1], 64.0)
    testing.assert_equal(c[1, 0], 139.0)
    testing.assert_equal(c[1, 1], 154.0)


def test_dynamic_matmul_dunder() raises:
    """Test dynamic matrix multiplication using @ operator."""
    var a = la.matrix[Float64]([[1.0, 0.0], [0.0, 1.0]])
    var b = la.matrix[Float64]([[5.0, 6.0], [7.0, 8.0]])
    var c = a @ b
    testing.assert_equal(c[0, 0], 5.0)
    testing.assert_equal(c[0, 1], 6.0)
    testing.assert_equal(c[1, 0], 7.0)
    testing.assert_equal(c[1, 1], 8.0)


def test_dynamic_matmul_dimension_mismatch_raises() raises:
    """Test that matmul with incompatible shapes raises ValueError."""
    var a = la.matrix[Float64]([[1.0, 2.0], [3.0, 4.0]])
    var b = la.matrix[Float64](
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
    )
    var raised = False
    try:
        var _c = matmul(a, b)
    except:
        raised = True
    testing.assert_true(
        raised, "Incompatible dimensions should raise ValueError"
    )


def test_dynamic_matmul_identity() raises:
    """Test multiplying by identity matrix."""
    var a = la.matrix[Float64](
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
    )
    var identity = la.matrix[Float64](
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    var c = a @ identity
    for i in range(3):
        for j in range(3):
            testing.assert_equal(c[i, j], a[i, j])


# ===----------------------------------------------------------------------===#
# Matmul over column-contiguous left operands (issue #1)
# ===----------------------------------------------------------------------===#
# These reach path 3 of `matmul`, which up to v0.2.0 accumulated into a freed
# buffer: the first product was already wrong and later ones grew from call to
# call. Each product is therefore taken more than once.


def test_matmul_f_order_repeated() raises:
    """F-order @ F-order gives the same exact product on every call."""
    var a = la.from_string[Float64]("[[1, 2], [3, 4]]", order="F")
    var b = la.from_string[Float64]("[[5, 6], [7, 8]]", order="F")
    for _ in range(3):
        var c = matmul(a, b)
        testing.assert_equal(c[0, 0], 19.0)
        testing.assert_equal(c[0, 1], 22.0)
        testing.assert_equal(c[1, 0], 43.0)
        testing.assert_equal(c[1, 1], 50.0)


def test_matmul_column_vector_times_strided_view_repeated() raises:
    """An `n x 1` C-order matrix is column-contiguous too."""
    var col = la.from_string[Float64]("[[1], [2], [3]]")
    var m = la.from_string[Float64]("[[10, 11, 20, 21, 30, 31]]")
    var row = m[Slice(0, 1), Slice(0, 6, 2)]
    for _ in range(3):
        var d = matmul(col, row)
        testing.assert_equal(d[0, 0], 10.0)
        testing.assert_equal(d[0, 1], 20.0)
        testing.assert_equal(d[0, 2], 30.0)
        testing.assert_equal(d[2, 2], 90.0)


def test_matmul_f_order_survives_later_allocations() raises:
    """A larger F-order product, interleaved with other allocations."""
    var a = la.contiguous(
        la.reshape(la.arange[Float64](0.0, 64.0), 16, 4), order="F"
    )
    var b = la.from_string[Float64]("[[1, 2], [3, 4], [5, 6], [7, 8]]")
    var want = matmul(la.contiguous(a), b)  # C-order: a different path
    for _ in range(10):
        var c = matmul(a, b)
        var _other = matmul(b, la.transpose(b))
        la.assert_matrices_equal(c, want)


def main() raises:
    testing.TestSuite.discover_tests[__functions_in_module()]().run()
