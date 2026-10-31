"""
Numpy-powered integration tests for matrix arithmetic operations.

Uses numpy.random to generate arbitrary-sized matrices and compares
linamo results against numpy as ground truth.
"""

import std.testing as testing
from linamo.routines.math import add, sub, mul, div, matmul
from linamo.routines.math import scalar_add, scalar_sub, scalar_mul, scalar_div
from linamo.routines.manipulation import contiguous
from linamo.types.matrix import Matrix
from linamo.types.matrix_view import MatrixView
from linamo.routines.numpy_interop import from_numpy, to_numpy
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
    var a = from_numpy(a_np)
    var b = from_numpy(b_np)
    var label = op_name + " " + String(r) + "x" + String(c)
    if op_name == "add":
        assert_matrices_close(
            add(a, b),
            from_numpy(np.add(a_np, b_np)),
            msg=label,
            atol=1e-12,
        )
    elif op_name == "sub":
        assert_matrices_close(
            sub(a, b),
            from_numpy(np.subtract(a_np, b_np)),
            msg=label,
            atol=1e-12,
        )
    elif op_name == "mul":
        assert_matrices_close(
            mul(a, b),
            from_numpy(np.multiply(a_np, b_np)),
            msg=label,
            atol=1e-12,
        )
    elif op_name == "div":
        assert_matrices_close(
            div(a, b),
            from_numpy(np.divide(a_np, b_np)),
            msg=label,
            atol=1e-10,
        )


def _test_matmul(np: PythonObject, m: Int, k: Int, n: Int) raises:
    """Helper: test matmul (m,k) @ (k,n) against numpy."""
    var a_np = np.random.rand(m, k)
    var b_np = np.random.rand(k, n)
    var a = from_numpy(a_np)
    var b = from_numpy(b_np)
    var expected = from_numpy(np.matmul(a_np, b_np))
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
# Matmul over every pair of layouts
# ===----------------------------------------------------------------------===#
# `matmul` picks one of four kernels from the operands' layouts, and the tests
# above only ever take the first (both C-order). Here each operand comes in
# C-order, F-order and as a strided view, and the nine pairs reach every
# kernel: B row-contiguous (any A with a C-order B), C-order A with F-order B,
# A column-contiguous (an F-order A, or an `n x 1` one) and the general
# fallback. Each product is taken twice, because a kernel that reuses a stale
# buffer can pass once and fail the next time (issue #1).


def _check_matmul[
    oa: Origin, ob: Origin, //
](
    a: MatrixView[Float64, oa],
    b: MatrixView[Float64, ob],
    want: Matrix[Float64],
    label: String,
) raises:
    for t in range(2):
        assert_matrices_close(
            matmul(a, b), want, msg=label + ", call " + String(t), atol=1e-9
        )


def _interleave_zeros(np: PythonObject, x: PythonObject) raises -> PythonObject:
    """Returns `x` with a zero column after each of its columns.

    Every second column of the result is `x` again, which is how the tests
    below build a strided view.
    """
    return np.stack(Python.list(x, np.zeros_like(x)), axis=2).reshape(
        x.shape[0], 2 * x.shape[1]
    )


def _test_matmul_layouts(np: PythonObject, m: Int, k: Int, n: Int) raises:
    """Helper: every layout pair of (m,k) @ (k,n) against numpy."""
    var a_np = np.random.rand(m, k)
    var b_np = np.random.rand(k, n)
    var want = from_numpy(np.matmul(a_np, b_np))

    var a_c = from_numpy(a_np)
    var a_f = contiguous(a_c, order="F")
    var a_big = from_numpy(_interleave_zeros(np, a_np))
    var a_s = a_big[0:m, 0 : 2 * k : 2]

    var b_c = from_numpy(b_np)
    var b_f = contiguous(b_c, order="F")
    var b_big = from_numpy(_interleave_zeros(np, b_np))
    var b_s = b_big[0:k, 0 : 2 * n : 2]

    var shape = (
        String(m) + "x" + String(k) + " @ " + String(k) + "x" + String(n)
    )
    _check_matmul(a_c.view(), b_c.view(), want, "C @ C " + shape)
    _check_matmul(a_c.view(), b_f.view(), want, "C @ F " + shape)
    _check_matmul(a_c.view(), b_s, want, "C @ strided " + shape)
    _check_matmul(a_f.view(), b_c.view(), want, "F @ C " + shape)
    _check_matmul(a_f.view(), b_f.view(), want, "F @ F " + shape)
    _check_matmul(a_f.view(), b_s, want, "F @ strided " + shape)
    _check_matmul(a_s, b_c.view(), want, "strided @ C " + shape)
    _check_matmul(a_s, b_f.view(), want, "strided @ F " + shape)
    _check_matmul(a_s, b_s, want, "strided @ strided " + shape)


def test_matmul_every_layout_pair() raises:
    """Tests matmul for all nine layout pairs against numpy."""
    var np = Python.import_module("numpy")
    _test_matmul_layouts(np, 2, 2, 2)
    _test_matmul_layouts(np, 5, 3, 4)
    _test_matmul_layouts(np, 17, 9, 13)  # not a multiple of the SIMD width
    _test_matmul_layouts(np, 3, 1, 5)  # A is n x 1, so column-contiguous
    _test_matmul_layouts(np, 6, 4, 64)  # enough columns to run in parallel
    _test_matmul_layouts(np, 64, 8, 1)


# ===----------------------------------------------------------------------===#
# Scalar ops vs numpy
# ===----------------------------------------------------------------------===#


def test_scalar_add_random() raises:
    """Tests scalar_add(A, s) == A + s via numpy."""
    var np = Python.import_module("numpy")
    var a_np = np.random.rand(5, 7)
    var s = Float64(py=np.random.rand())
    var a = from_numpy(a_np)
    var expected = from_numpy(a_np + s)
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
    var a = from_numpy(a_np)
    var expected = from_numpy(a_np - s)
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
    var a = from_numpy(a_np)
    var expected = from_numpy(a_np * s)
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
    var a = from_numpy(a_np)
    var expected = from_numpy(a_np / s)
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
