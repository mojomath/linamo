"""
Numpy-powered integration tests for matrix arithmetic operations.

Uses numpy.random to generate arbitrary-sized matrices and compares
linamo results against numpy as ground truth.
"""

import std.testing as testing
from linamo.routines.math import add, sub, mul, div, matmul
from linamo.routines.math import scalar_add, scalar_sub, scalar_mul, scalar_div
from linamo.routines.numpy_interop import matrix_from_numpy, to_numpy
from linamo.utils.test_utils import assert_matrices_close
from std.python import Python, PythonObject


# ===----------------------------------------------------------------------===#
# Helpers
# ===----------------------------------------------------------------------===#


def _test_elementwise_op(
    np: PythonObject,
    r: Int,
    c: Int,
    op_name: String,
) raises:
    """Helper: test a+b, a-b, a*b, a/b against numpy for shape (r, c)."""
    var a_np = np.random.rand(r, c)
    var b_np = np.random.rand(r, c) + 0.1  # avoid near-zero for div
    var a = matrix_from_numpy(a_np)
    var b = matrix_from_numpy(b_np)
    var label = op_name + " " + String(r) + "x" + String(c)
    if op_name == "add":
        assert_matrices_close(
            add(a, b),
            matrix_from_numpy(np.add(a_np, b_np)),
            msg=label,
            atol=1e-12,
        )
    elif op_name == "sub":
        assert_matrices_close(
            sub(a, b),
            matrix_from_numpy(np.subtract(a_np, b_np)),
            msg=label,
            atol=1e-12,
        )
    elif op_name == "mul":
        assert_matrices_close(
            mul(a, b),
            matrix_from_numpy(np.multiply(a_np, b_np)),
            msg=label,
            atol=1e-12,
        )
    elif op_name == "div":
        assert_matrices_close(
            div(a, b),
            matrix_from_numpy(np.divide(a_np, b_np)),
            msg=label,
            atol=1e-10,
        )


def _test_matmul(np: PythonObject, m: Int, k: Int, n: Int) raises:
    """Helper: test matmul (m,k) @ (k,n) against numpy."""
    var a_np = np.random.rand(m, k)
    var b_np = np.random.rand(k, n)
    var a = matrix_from_numpy(a_np)
    var b = matrix_from_numpy(b_np)
    var expected = matrix_from_numpy(np.matmul(a_np, b_np))
    assert_matrices_close(
        matmul(a, b),
        expected,
        msg="matmul "
        + String(m)
        + "x"
        + String(k)
        + " @ "
        + String(k)
        + "x"
        + String(n),
        atol=1e-9,
    )


# ===----------------------------------------------------------------------===#
# Elementwise ops vs numpy
# ===----------------------------------------------------------------------===#


def test_add_random() raises:
    """Tests add(A, B) == np.add(A, B) with random matrices of various sizes."""
    var np = Python.import_module("numpy")
    _test_elementwise_op(np, 3, 4, "add")
    _test_elementwise_op(np, 1, 1, "add")
    _test_elementwise_op(np, 7, 13, "add")
    _test_elementwise_op(np, 16, 16, "add")
    _test_elementwise_op(np, 1, 50, "add")
    _test_elementwise_op(np, 50, 1, "add")


def test_sub_random() raises:
    """Tests sub(A, B) == np.subtract(A, B) with random matrices."""
    var np = Python.import_module("numpy")
    _test_elementwise_op(np, 4, 5, "sub")
    _test_elementwise_op(np, 1, 1, "sub")
    _test_elementwise_op(np, 10, 3, "sub")
    _test_elementwise_op(np, 8, 8, "sub")


def test_mul_random() raises:
    """Tests mul(A, B) == np.multiply(A, B) (Hadamard) with random matrices."""
    var np = Python.import_module("numpy")
    _test_elementwise_op(np, 5, 6, "mul")
    _test_elementwise_op(np, 1, 1, "mul")
    _test_elementwise_op(np, 12, 9, "mul")
    _test_elementwise_op(np, 20, 20, "mul")


def test_div_random() raises:
    """Tests div(A, B) == np.divide(A, B) with random positive matrices."""
    var np = Python.import_module("numpy")
    _test_elementwise_op(np, 3, 3, "div")
    _test_elementwise_op(np, 1, 1, "div")
    _test_elementwise_op(np, 6, 10, "div")
    _test_elementwise_op(np, 15, 15, "div")


# ===----------------------------------------------------------------------===#
# Matmul vs numpy
# ===----------------------------------------------------------------------===#


def test_matmul_square_random() raises:
    """Tests matmul(A, B) == np.matmul(A, B) for random square matrices."""
    var np = Python.import_module("numpy")
    _test_matmul(np, 1, 1, 1)
    _test_matmul(np, 2, 2, 2)
    _test_matmul(np, 4, 4, 4)
    _test_matmul(np, 7, 7, 7)
    _test_matmul(np, 16, 16, 16)
    _test_matmul(np, 32, 32, 32)


def test_matmul_rect_random() raises:
    """Tests matmul(A, B) == np.matmul(A, B) for random rectangular matrices."""
    var np = Python.import_module("numpy")
    _test_matmul(np, 2, 5, 3)
    _test_matmul(np, 1, 10, 1)
    _test_matmul(np, 8, 4, 12)
    _test_matmul(np, 13, 7, 9)
    _test_matmul(np, 3, 1, 5)


def test_matmul_large_random() raises:
    """Tests matmul for larger matrices (64x48 @ 48x64)."""
    var np = Python.import_module("numpy")
    _test_matmul(np, 64, 48, 64)


# ===----------------------------------------------------------------------===#
# Scalar ops vs numpy
# ===----------------------------------------------------------------------===#


def test_scalar_add_random() raises:
    """Tests scalar_add(A, s) == A + s via numpy."""
    var np = Python.import_module("numpy")
    var a_np = np.random.rand(5, 7)
    var s = Float64(py=np.random.rand())
    var a = matrix_from_numpy(a_np)
    var expected = matrix_from_numpy(a_np + s)
    assert_matrices_close(
        scalar_add(a, s),
        expected,
        msg="scalar_add",
        atol=1e-12,
    )


def test_scalar_sub_random() raises:
    """Tests scalar_sub(A, s) == A - s via numpy."""
    var np = Python.import_module("numpy")
    var a_np = np.random.rand(6, 4)
    var s = Float64(py=np.random.rand())
    var a = matrix_from_numpy(a_np)
    var expected = matrix_from_numpy(a_np - s)
    assert_matrices_close(
        scalar_sub(a, s),
        expected,
        msg="scalar_sub",
        atol=1e-12,
    )


def test_scalar_mul_random() raises:
    """Tests scalar_mul(A, s) == A * s via numpy."""
    var np = Python.import_module("numpy")
    var a_np = np.random.rand(8, 3)
    var s = Float64(py=np.random.rand()) * 10.0
    var a = matrix_from_numpy(a_np)
    var expected = matrix_from_numpy(a_np * s)
    assert_matrices_close(
        scalar_mul(a, s),
        expected,
        msg="scalar_mul",
        atol=1e-12,
    )


def test_scalar_div_random() raises:
    """Tests scalar_div(A, s) == A / s via numpy."""
    var np = Python.import_module("numpy")
    var a_np = np.random.rand(4, 9)
    var s = Float64(py=np.random.rand()) + 0.5  # avoid near-zero
    var a = matrix_from_numpy(a_np)
    var expected = matrix_from_numpy(a_np / s)
    assert_matrices_close(
        scalar_div(a, s),
        expected,
        msg="scalar_div",
        atol=1e-12,
    )


# ===----------------------------------------------------------------------===#
# Test runner
# ===----------------------------------------------------------------------===#


def main() raises:
    testing.TestSuite.discover_tests[__functions_in_module()]().run()
