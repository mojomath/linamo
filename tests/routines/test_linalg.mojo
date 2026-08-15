"""
Tests for linear algebra routines: transpose, trace.
"""

import std.testing as testing
import linamo as mm
from linamo.routines.linalg import transpose, trace


# ===----------------------------------------------------------------------===#
# transpose
# ===----------------------------------------------------------------------===#


def test_transpose_basic() raises:
    """Test basic matrix transpose."""
    var mat = mm.matrix[DType.float64]([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    var t = transpose(mat)
    testing.assert_equal(t.nrows, 3)
    testing.assert_equal(t.ncols, 2)
    testing.assert_equal(t[0, 0], 1.0)
    testing.assert_equal(t[0, 1], 4.0)
    testing.assert_equal(t[1, 0], 2.0)
    testing.assert_equal(t[1, 1], 5.0)
    testing.assert_equal(t[2, 0], 3.0)
    testing.assert_equal(t[2, 1], 6.0)


def test_transpose_square() raises:
    """Test transpose of a square matrix."""
    var mat = mm.matrix[DType.float64]([[1.0, 2.0], [3.0, 4.0]])
    var t = transpose(mat)
    testing.assert_equal(t[0, 0], 1.0)
    testing.assert_equal(t[0, 1], 3.0)
    testing.assert_equal(t[1, 0], 2.0)
    testing.assert_equal(t[1, 1], 4.0)


def test_transpose_single_row() raises:
    """Test transpose of a single-row matrix becomes a column matrix."""
    var mat = mm.matrix[DType.float64]([[1.0, 2.0, 3.0]])
    var t = transpose(mat)
    testing.assert_equal(t.nrows, 3)
    testing.assert_equal(t.ncols, 1)
    testing.assert_equal(t[0, 0], 1.0)
    testing.assert_equal(t[1, 0], 2.0)
    testing.assert_equal(t[2, 0], 3.0)


def test_transpose_single_col() raises:
    """Test transpose of a single-column matrix becomes a row matrix."""
    var mat = mm.matrix[DType.float64]([[1.0], [2.0], [3.0]])
    var t = transpose(mat)
    testing.assert_equal(t.nrows, 1)
    testing.assert_equal(t.ncols, 3)
    testing.assert_equal(t[0, 0], 1.0)
    testing.assert_equal(t[0, 1], 2.0)
    testing.assert_equal(t[0, 2], 3.0)


def test_transpose_col_major() raises:
    """Test transpose of a column-major matrix."""
    var mat = mm.matrix[DType.float64]([[1.0, 2.0], [3.0, 4.0]], order="F")
    var t = transpose(mat)
    testing.assert_equal(t[0, 0], 1.0)
    testing.assert_equal(t[0, 1], 3.0)
    testing.assert_equal(t[1, 0], 2.0)
    testing.assert_equal(t[1, 1], 4.0)


def test_transpose_double() raises:
    """Test that transposing twice gives back the original values."""
    var mat = mm.matrix[DType.float64]([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    var tt = transpose(transpose(mat))
    testing.assert_equal(tt.nrows, 2)
    testing.assert_equal(tt.ncols, 3)
    for i in range(2):
        for j in range(3):
            testing.assert_equal(tt[i, j], mat[i, j])


def test_transpose_identity() raises:
    """Test that transpose of identity is identity."""
    var id = mm.eye[DType.float64](4)
    var t = transpose(id)
    for i in range(4):
        for j in range(4):
            testing.assert_equal(t[i, j], id[i, j])


def test_transpose_via_mm() raises:
    """Test transpose via the mm module shortcut."""
    var mat = mm.matrix[DType.float64]([[1.0, 2.0], [3.0, 4.0]])
    var t = mm.transpose(mat)
    testing.assert_equal(t[0, 1], 3.0)
    testing.assert_equal(t[1, 0], 2.0)


# ===----------------------------------------------------------------------===#
# trace
# ===----------------------------------------------------------------------===#


def test_trace_basic() raises:
    """Test trace of a square matrix."""
    var mat = mm.matrix[DType.float64](
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
    )
    var t = trace(mat)
    testing.assert_equal(t, 15.0)


def test_trace_identity() raises:
    """Test trace of identity matrix equals n."""
    var id = mm.eye[DType.float64](5)
    testing.assert_equal(trace(id), 5.0)


def test_trace_1x1() raises:
    """Test trace of a 1x1 matrix."""
    var mat = mm.matrix[DType.float64]([[42.0]])
    testing.assert_equal(trace(mat), 42.0)


def test_trace_col_major() raises:
    """Test trace of a column-major matrix."""
    var mat = mm.matrix[DType.float64]([[1.0, 0.0], [0.0, 2.0]], order="F")
    testing.assert_equal(trace(mat), 3.0)


def test_trace_nonsquare_raises() raises:
    """Test that trace on non-square matrix raises ValueError."""
    var mat = mm.matrix[DType.float64]([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    var raised = False
    try:
        var _t = trace(mat)
    except:
        raised = True
    testing.assert_true(raised, "Non-square matrix should raise ValueError")


def test_trace_via_mm() raises:
    """Test trace via the mm module shortcut."""
    var mat = mm.matrix[DType.float64]([[10.0, 0.0], [0.0, 20.0]])
    testing.assert_equal(mm.trace(mat), 30.0)


def main() raises:
    testing.TestSuite.discover_tests[__functions_in_module()]().run()
