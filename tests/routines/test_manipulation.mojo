"""
Tests for the Phase 5.4 shape and layout routines.

Four concerns run through the file, because each fails in its own way:

*Index order.* `order="F"` is checked against NumPy's answer for a non-square
matrix, since a C/F mix-up on a square one still produces a plausible grid.

*Strided operands.* Every routine takes a `MatrixView`, so each copying
routine is also run on `m[0:4:2, 0:4:2]`. One that works on `m` but not on a
stepped slice has hard-coded a stride.

*Aliasing.* `reshape_view` and `broadcast_to` return views over the input's
buffer. The tests write through the owner afterwards and read the change back
out of the view, which is the whole point of not copying.

*Layout preservation.* `contiguous(m, "F")` has to change the strides and
leave every `m[i, j]` where it was.
"""

import std.testing as testing
import linamo as la
from linamo.routines.manipulation import (
    astype,
    broadcast_to,
    contiguous,
    flatten,
    reorder_layout,
    reshape,
    reshape_view,
    resize,
)


def _m() raises -> la.Matrix[DType.float64]:
    """A 2x3 matrix: non-square, so a transposed axis cannot hide."""
    return la.matrix[DType.float64]([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])


def _m4() raises -> la.Matrix[DType.float64]:
    """A 4x4 matrix whose entries are their own flat C index."""
    var m = la.Matrix[DType.float64](
        nrows=4, ncols=4, row_stride=4, col_stride=1
    )
    for i in range(4):
        for j in range(4):
            m[i, j] = Float64(i * 4 + j)
    return m^


# ===----------------------------------------------------------------------===#
# reshape
# ===----------------------------------------------------------------------===#


def test_reshape_c_order() raises:
    var r = reshape(_m(), 3, 2)
    testing.assert_equal(r.nrows(), 3)
    testing.assert_equal(r.ncols(), 2)
    # np.reshape([[1,2,3],[4,5,6]], (3, 2)) -> [[1,2],[3,4],[5,6]]
    var expected: List[Float64] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    for k in range(6):
        testing.assert_equal(r[k // 2, k % 2], expected[k])


def test_reshape_f_order() raises:
    var r = reshape(_m(), 3, 2, "F")
    # np.reshape([[1,2,3],[4,5,6]], (3, 2), order="F") -> [[1,5],[4,3],[2,6]]
    testing.assert_equal(r[0, 0], 1.0)
    testing.assert_equal(r[0, 1], 5.0)
    testing.assert_equal(r[1, 0], 4.0)
    testing.assert_equal(r[1, 1], 3.0)
    testing.assert_equal(r[2, 0], 2.0)
    testing.assert_equal(r[2, 1], 6.0)


def test_reshape_result_is_c_contiguous() raises:
    """An owning result is C-contiguous whatever the input layout was."""
    var f = contiguous(_m(), "F")
    testing.assert_true(f.is_f_contiguous())
    var r = reshape(f, 3, 2)
    testing.assert_true(r.is_c_contiguous())
    testing.assert_equal(r[0, 0], 1.0)
    testing.assert_equal(r[0, 1], 2.0)


def test_reshape_of_a_strided_view() raises:
    var m = _m4()
    var s = m[0:4:2, 0:4:2]  # [[0, 2], [8, 10]]
    var r = reshape(s, 1, 4)
    testing.assert_equal(r[0, 0], 0.0)
    testing.assert_equal(r[0, 1], 2.0)
    testing.assert_equal(r[0, 2], 8.0)
    testing.assert_equal(r[0, 3], 10.0)


def test_reshape_size_mismatch_raises() raises:
    with testing.assert_raises():
        _ = reshape(_m(), 4, 2)


def test_reshape_bad_order_raises() raises:
    with testing.assert_raises():
        _ = reshape(_m(), 3, 2, "Z")


# ===----------------------------------------------------------------------===#
# reshape_view
# ===----------------------------------------------------------------------===#


def test_reshape_view_shares_the_buffer() raises:
    var m = _m()
    var v = reshape_view(m, 3, 2)
    testing.assert_equal(v.nrows(), 3)
    testing.assert_equal(v.ncols(), 2)
    testing.assert_equal(v[1, 0], 3.0)
    m[0, 0] = 99.0
    testing.assert_equal(v[0, 0], 99.0)


def test_reshape_view_keeps_the_input_layout() raises:
    var c = _m()
    testing.assert_true(reshape_view(c, 3, 2).is_c_contiguous())
    var f = contiguous(_m(), "F")
    testing.assert_true(reshape_view(f, 3, 2).is_f_contiguous())


def test_reshape_view_of_f_contiguous_reads_memory_order() raises:
    """The elements keep their positions in memory; only the shape changes."""
    var f = contiguous(_m(), "F")  # memory order: 1 4 2 5 3 6
    var v = reshape_view(f, 3, 2)  # F layout, so column-major fill
    testing.assert_equal(v[0, 0], 1.0)
    testing.assert_equal(v[1, 0], 4.0)
    testing.assert_equal(v[2, 0], 2.0)
    testing.assert_equal(v[0, 1], 5.0)


def test_reshape_view_of_a_strided_view_raises() raises:
    var m = _m4()
    with testing.assert_raises():
        _ = reshape_view(m[0:4:2, 0:4:2], 1, 4)


def test_reshape_view_size_mismatch_raises() raises:
    with testing.assert_raises():
        _ = reshape_view(_m(), 4, 2)


# ===----------------------------------------------------------------------===#
# flatten
# ===----------------------------------------------------------------------===#


def test_flatten_c_order() raises:
    var r = flatten(_m())
    testing.assert_equal(r.nrows(), 1)
    testing.assert_equal(r.ncols(), 6)
    for k in range(6):
        testing.assert_equal(r[0, k], Float64(k + 1))


def test_flatten_f_order() raises:
    var r = flatten(_m(), "F")
    var expected: List[Float64] = [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
    for k in range(6):
        testing.assert_equal(r[0, k], expected[k])


def test_flatten_of_a_strided_view() raises:
    var m = _m4()
    var r = flatten(m[1:3, 1:3])  # [[5, 6], [9, 10]]
    testing.assert_equal(r.ncols(), 4)
    testing.assert_equal(r[0, 0], 5.0)
    testing.assert_equal(r[0, 3], 10.0)


# ===----------------------------------------------------------------------===#
# resize
# ===----------------------------------------------------------------------===#


def test_resize_grows_and_zero_pads() raises:
    var r = resize(_m(), 3, 3)
    testing.assert_equal(r.nrows(), 3)
    testing.assert_equal(r.ncols(), 3)
    for k in range(6):
        testing.assert_equal(r[k // 3, k % 3], Float64(k + 1))
    for j in range(3):
        testing.assert_equal(r[2, j], 0.0)


def test_resize_shrinks_and_truncates() raises:
    var r = resize(_m(), 2, 2)
    testing.assert_equal(r[0, 0], 1.0)
    testing.assert_equal(r[0, 1], 2.0)
    testing.assert_equal(r[1, 0], 3.0)
    testing.assert_equal(r[1, 1], 4.0)


def test_resize_leaves_the_source_alone() raises:
    """The buffer of an existing matrix is fixed; `resize` returns a new one."""
    var m = _m()
    var r = resize(m, 4, 4)
    testing.assert_equal(m.nrows(), 2)
    testing.assert_equal(m.ncols(), 3)
    testing.assert_equal(m[0, 0], 1.0)
    testing.assert_equal(r.nrows(), 4)


def test_resize_negative_shape_raises() raises:
    with testing.assert_raises():
        _ = resize(_m(), -1, 3)


# ===----------------------------------------------------------------------===#
# contiguous / reorder_layout
# ===----------------------------------------------------------------------===#


def test_contiguous_f_changes_strides_not_elements() raises:
    var m = _m()
    var f = contiguous(m, "F")
    testing.assert_true(f.is_f_contiguous())
    testing.assert_equal(f.row_stride(), 1)
    testing.assert_equal(f.col_stride(), 2)
    for i in range(2):
        for j in range(3):
            testing.assert_equal(f[i, j], m[i, j])


def test_contiguous_densifies_a_strided_view() raises:
    var m = _m4()
    var d = contiguous(m[0:4:2, 0:4:2])
    testing.assert_true(d.is_c_contiguous())
    testing.assert_equal(d.nrows(), 2)
    testing.assert_equal(d.ncols(), 2)
    testing.assert_equal(d[0, 0], 0.0)
    testing.assert_equal(d[0, 1], 2.0)
    testing.assert_equal(d[1, 0], 8.0)
    testing.assert_equal(d[1, 1], 10.0)


def test_contiguous_copies_rather_than_aliases() raises:
    var m = _m()
    var c = contiguous(m)
    m[0, 0] = 99.0
    testing.assert_equal(c[0, 0], 1.0)


def test_contiguous_bad_order_raises() raises:
    with testing.assert_raises():
        _ = contiguous(_m(), "Z")


def test_reorder_layout_flips_both_ways() raises:
    var c = _m()
    var f = reorder_layout(c)
    testing.assert_true(f.is_f_contiguous())
    var back = reorder_layout(f)
    testing.assert_true(back.is_c_contiguous())
    for i in range(2):
        for j in range(3):
            testing.assert_equal(back[i, j], c[i, j])


def test_reorder_layout_of_a_strided_view_raises() raises:
    var m = _m4()
    with testing.assert_raises():
        _ = reorder_layout(m[0:4:2, 0:4:2])


# ===----------------------------------------------------------------------===#
# broadcast_to
# ===----------------------------------------------------------------------===#


def test_broadcast_row() raises:
    var r = la.matrix[DType.float64]([[1.0, 2.0, 3.0]])
    var b = broadcast_to(r, 3, 3)
    testing.assert_equal(b.nrows(), 3)
    testing.assert_equal(b.ncols(), 3)
    testing.assert_equal(b.row_stride(), 0)
    for i in range(3):
        for j in range(3):
            testing.assert_equal(b[i, j], Float64(j + 1))


def test_broadcast_column() raises:
    var c = la.matrix[DType.float64]([[1.0], [2.0], [3.0]])
    var b = broadcast_to(c, 3, 4)
    testing.assert_equal(b.col_stride(), 0)
    for i in range(3):
        for j in range(4):
            testing.assert_equal(b[i, j], Float64(i + 1))


def test_broadcast_scalar_matrix() raises:
    var s = la.matrix[DType.float64]([[7.0]])
    var b = broadcast_to(s, 2, 5)
    testing.assert_equal(b.size(), 10)
    testing.assert_equal(b[1, 4], 7.0)


def test_broadcast_shares_the_buffer() raises:
    var r = la.matrix[DType.float64]([[1.0, 2.0, 3.0]])
    var b = broadcast_to(r, 3, 3)
    r[0, 1] = 99.0
    testing.assert_equal(b[0, 1], 99.0)
    testing.assert_equal(b[2, 1], 99.0)


def test_broadcast_result_feeds_the_routine_layer() raises:
    """A stride-0 view is a normal operand; only the fast paths are skipped."""
    var r = la.matrix[DType.float64]([[1.0, 2.0, 3.0]])
    var b = broadcast_to(r, 2, 3)
    testing.assert_false(b.is_c_contiguous())
    var m = _m()
    var s = m + b
    testing.assert_equal(s[0, 0], 2.0)
    testing.assert_equal(s[1, 2], 9.0)
    testing.assert_equal(b.to_matrix()[1, 0], 1.0)


def test_broadcast_incompatible_raises() raises:
    with testing.assert_raises():
        _ = broadcast_to(_m(), 4, 3)
    with testing.assert_raises():
        _ = broadcast_to(_m(), 2, 6)


# ===----------------------------------------------------------------------===#
# astype
# ===----------------------------------------------------------------------===#


def test_astype_float_to_int_truncates() raises:
    var m = la.matrix[DType.float64]([[1.7, -2.9], [3.2, 4.5]])
    var i = astype[DType.int32](m)
    testing.assert_equal(i[0, 0], 1)
    testing.assert_equal(i[0, 1], -2)
    testing.assert_equal(i[1, 0], 3)
    testing.assert_equal(i[1, 1], 4)


def test_astype_method_on_matrix_and_view() raises:
    var m = _m()
    var a = m.astype[DType.float32]()
    testing.assert_equal(a[1, 2], Float32(6.0))
    var b = m[0:1, 0:2].astype[DType.float32]()
    testing.assert_equal(b.nrows(), 1)
    testing.assert_equal(b.ncols(), 2)
    testing.assert_equal(b[0, 1], Float32(2.0))


def test_astype_of_a_strided_view_is_dense() raises:
    var m = _m4()
    var i = astype[DType.int64](m[0:4:2, 0:4:2])
    testing.assert_true(i.is_c_contiguous())
    testing.assert_equal(i[1, 1], 10)


# ===----------------------------------------------------------------------===#
# fill
# ===----------------------------------------------------------------------===#


def test_fill_whole_matrix() raises:
    var m = _m()
    m.set(7.0)
    for i in range(2):
        for j in range(3):
            testing.assert_equal(m[i, j], 7.0)


def test_fill_whole_matrix_respects_strides() raises:
    var m = contiguous(_m(), "F")
    m.set(-1.0)
    testing.assert_true(m.is_f_contiguous())
    for i in range(2):
        for j in range(3):
            testing.assert_equal(m[i, j], -1.0)


def test_fill_region_still_works() raises:
    var m = _m()
    m.set(Slice(0, 1), Slice(0, 3), 0.0)
    testing.assert_equal(m[0, 0], 0.0)
    testing.assert_equal(m[0, 2], 0.0)
    testing.assert_equal(m[1, 0], 4.0)


def main() raises:
    var suite = testing.TestSuite.discover_tests[__functions_in_module()]()
    suite^.run()
