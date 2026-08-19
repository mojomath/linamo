"""
Test comparison utilities for Linamo.

Provides assertion functions for comparing matrices, including:

- `assert_matrices_equal()`: Exact element-wise comparison for C-contiguous matrices.
- `assert_matrices_close()`: Approximate comparison with configurable tolerances.
"""

from std.testing import assert_true

from linamo.types.matrix import Matrix


def assert_matrices_equal[
    dtype: DType = DType.float64
](a: Matrix[dtype], b: Matrix[dtype], msg: String = "") raises:
    """Assert two C-contiguous matrices are exactly equal element-by-element.

    Both matrices must be C-contiguous. This performs a flat buffer comparison
    which is essentially equivalent to memcmp for same-layout matrices.

    Parameters:
        dtype: The data type of the matrix elements.

    Args:
        a: First matrix.
        b: Second matrix.
        msg: Optional message prefix on failure.

    Raises:
        Error: If shapes don't match, matrices aren't C-contiguous, or
        any elements differ.
    """
    assert_true(
        a.nrows() == b.nrows() and a.ncols() == b.ncols(),
        msg=msg
        + " shape mismatch: ("
        + String(a.nrows())
        + "x"
        + String(a.ncols())
        + ") vs ("
        + String(b.nrows())
        + "x"
        + String(b.ncols())
        + ")",
    )

    # Both must be C-contiguous for flat buffer comparison
    assert_true(
        a.is_c_contiguous() and b.is_c_contiguous(),
        msg=msg + " both matrices must be C-contiguous for exact comparison",
    )

    var n = a.nrows() * a.ncols()
    var pa = a._data._data
    var pb = b._data._data
    for i in range(n):
        assert_true(
            pa[unsafe_offset=i] == pb[unsafe_offset=i],
            msg=msg
            + " element mismatch at flat index "
            + String(i)
            + ": "
            + String(pa[unsafe_offset=i])
            + " vs "
            + String(pb[unsafe_offset=i]),
        )


def assert_matrices_close[
    dtype: DType = DType.float64
](
    a: Matrix[dtype],
    b: Matrix[dtype],
    msg: String = "",
    rtol: Float64 = 1e-7,
    atol: Float64 = 1e-10,
) raises:
    """Assert two matrices have the same shape and approximately equal elements.

    Uses the formula: |a - b| <= atol + rtol * |b| for each element.
    Works with any memory layout (C-contiguous or F-contiguous).

    Parameters:
        dtype: The data type of the matrix elements.

    Args:
        a: First matrix.
        b: Second matrix.
        msg: Optional message prefix on failure.
        rtol: Relative tolerance (default 1e-7).
        atol: Absolute tolerance (default 1e-10).

    Raises:
        Error: If shapes don't match or any elements exceed tolerance.
    """
    assert_true(
        a.nrows() == b.nrows() and a.ncols() == b.ncols(),
        msg=msg
        + " shape mismatch: ("
        + String(a.nrows())
        + "x"
        + String(a.ncols())
        + ") vs ("
        + String(b.nrows())
        + "x"
        + String(b.ncols())
        + ")",
    )
    for i in range(a.nrows()):
        for j in range(a.ncols()):
            var va = a._data[i * a.row_stride() + j * a.col_stride()]
            var vb = b._data[i * b.row_stride() + j * b.col_stride()]
            var diff = Float64(va - vb)
            if diff < 0:
                diff = -diff
            var abs_vb = Float64(vb)
            if abs_vb < 0:
                abs_vb = -abs_vb
            assert_true(
                diff <= atol + rtol * abs_vb,
                msg=msg
                + " element mismatch at ("
                + String(i)
                + ","
                + String(j)
                + "): "
                + String(va)
                + " vs "
                + String(vb),
            )
