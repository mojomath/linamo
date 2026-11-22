"""
Tests for `norm`: each order, each element family, views, NaN and the edges.
"""

import std.testing as testing
from std.math import nan, sqrt

import linamo as la
from linamo.routines.linalg import norm
from linamo.types.complex import CFloat64


def _assert_raises_value_error(
    m: la.Matrix[Float64], ord: String, msg: String
) raises:
    """Asserts that `norm(m, ord)` raises."""
    var raised = False
    try:
        _ = norm(m, ord)
    except:
        raised = True
    testing.assert_true(raised, msg)


# ===----------------------------------------------------------------------===#
# Each order
# ===----------------------------------------------------------------------===#
# A 2 x 3 matrix, so that a mix-up between rows and columns changes the answer.
# Column sums of absolute values are 5, 7, 9; row sums are 6, 15.


def _a() raises -> la.Matrix[Float64]:
    return la.matrix[Float64]([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]])


def test_norm_frobenius_is_the_default() raises:
    """Test that the default is the Frobenius norm."""
    var a = _a()
    testing.assert_almost_equal(norm(a), sqrt(91.0))
    testing.assert_equal(norm(a), norm(a, "fro"))


def test_norm_one_is_the_largest_column_sum() raises:
    """Test "1" and "-1" against the column sums 5, 7, 9."""
    var a = _a()
    testing.assert_equal(norm(a, "1"), 9.0)
    testing.assert_equal(norm(a, "-1"), 5.0)


def test_norm_inf_is_the_largest_row_sum() raises:
    """Test "inf" and "-inf" against the row sums 6, 15."""
    var a = _a()
    testing.assert_equal(norm(a, "inf"), 15.0)
    testing.assert_equal(norm(a, "-inf"), 6.0)


def test_norm_column_vector_gives_the_vector_norms() raises:
    """Test that an n x 1 matrix gives the Euclidean, 1- and max norms."""
    var x = la.matrix[Float64]([[3.0], [-4.0], [0.0]])
    testing.assert_equal(norm(x), 5.0)
    testing.assert_equal(norm(x, "1"), 7.0)
    testing.assert_equal(norm(x, "inf"), 4.0)


def test_norm_of_transpose_swaps_one_and_inf() raises:
    """Test that the "1" norm of A^T is the "inf" norm of A."""
    var a = _a()
    var t = la.transpose(a)
    testing.assert_equal(norm(t, "1"), norm(a, "inf"))
    testing.assert_equal(norm(t, "-inf"), norm(a, "-1"))


def test_norm_reads_views_and_layouts() raises:
    """Test a strided sub-view and an F-order copy."""
    var big = la.matrix[Float64](
        [
            [1.0, 0.0, -2.0, 0.0, 3.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [-4.0, 0.0, 5.0, 0.0, -6.0],
        ]
    )
    var v = big[0:3:2, 0:5:2]
    var a = _a()
    for ord in ["fro", "1", "-1", "inf", "-inf"]:
        testing.assert_equal(norm(v, ord), norm(a, ord), ord)
        testing.assert_equal(
            norm(la.contiguous(a, order="F"), ord), norm(a, ord), ord
        )


def test_norm_via_package_alias() raises:
    """Test that `norm` resolves as `la.norm`."""
    testing.assert_equal(la.norm(_a(), "1"), 9.0)


# ===----------------------------------------------------------------------===#
# Element types
# ===----------------------------------------------------------------------===#


def test_norm_integer_dtype() raises:
    """Test that "1" and "inf" are exact over an integer dtype."""
    var a = la.matrix[Int32]([[1, -2, 3], [-4, 5, -6]])
    testing.assert_equal(norm(a, "1"), 9)
    testing.assert_equal(norm(a, "inf"), 15)


def test_norm_integer_dtype_frobenius_raises() raises:
    """Test that "fro" over an integer dtype raises instead of truncating."""
    var a = la.matrix[Int32]([[1, 2]])
    with testing.assert_raises(contains="floating-point"):
        _ = norm(a)


def test_norm_decimal_is_exact() raises:
    """Test "1" and "fro" over `Decimal`."""
    var a = la.from_string[la.Decimal]("[[0.1, -0.2], [0.3, 0.4]]")
    testing.assert_equal(String(norm(a, "1")), "0.6")
    testing.assert_equal(String(norm(a, "-inf")), "0.3")
    # sqrt(0.01 + 0.04 + 0.09 + 0.16) = sqrt(0.3)
    testing.assert_true(String(norm(a)).startswith("0.547722557505166113456"))


def test_norm_bint() raises:
    """Test "1" and "inf" over `BInt`, and that "fro" raises."""
    var a = la.from_string[la.BInt]("[[1, -2, 3], [-4, 5, -6]]")
    testing.assert_equal(String(norm(a, "1")), "9")
    testing.assert_equal(String(norm(a, "inf")), "15")
    with testing.assert_raises(contains="BInt"):
        _ = norm(a)


def test_norm_complex_is_real() raises:
    """Test that a complex matrix measures by magnitude and gives a real."""
    var c = la.from_string[CFloat64]("[[3+4i, 1], [0, -1i]]")
    var fro: Float64 = norm(c)
    testing.assert_almost_equal(fro, sqrt(27.0))
    testing.assert_equal(norm(c, "inf"), 6.0)
    testing.assert_equal(norm(c, "1"), 5.0)
    testing.assert_equal(norm(c, "-1"), 2.0)


# ===----------------------------------------------------------------------===#
# NaN, empty matrices and bad orders
# ===----------------------------------------------------------------------===#


def test_norm_nan_wins_wherever_it_is() raises:
    """Test that a NaN in any row or column makes every norm NaN."""
    comptime NAN = nan[DType.float64]()
    for r in range(2):
        for c in range(2):
            var a = la.matrix[Float64]([[1.0, 2.0], [3.0, 4.0]])
            a[r, c] = NAN
            for ord in ["fro", "1", "-1", "inf", "-inf"]:
                var got = norm(a, ord)
                testing.assert_true(
                    got != got,
                    String("NaN at (", r, ", ", c, "), ord ", ord),
                )


def test_norm_empty() raises:
    """Test empty matrices: the largest of no sums is 0, the smallest raises."""
    var rows0 = la.zeros[Float64](0, 3)
    testing.assert_equal(norm(rows0), 0.0)
    testing.assert_equal(norm(rows0, "1"), 0.0)
    testing.assert_equal(norm(rows0, "-1"), 0.0)
    testing.assert_equal(norm(rows0, "inf"), 0.0)
    _assert_raises_value_error(rows0, "-inf", "no rows for -inf")
    var cols0 = la.zeros[Float64](3, 0)
    testing.assert_equal(norm(cols0, "-inf"), 0.0)
    _assert_raises_value_error(cols0, "-1", "no columns for -1")


def test_norm_unknown_ord_raises() raises:
    """Test that an unknown order raises, naming the accepted ones."""
    with testing.assert_raises(contains="must be one of"):
        _ = norm(_a(), "max")


def test_norm_singular_value_orders_raise() raises:
    """Test that "2", "-2" and "nuc" raise until SVD exists."""
    for ord in ["2", "-2", "nuc"]:
        _assert_raises_value_error(_a(), ord, ord)


def main() raises:
    testing.TestSuite.discover_tests[__functions_in_module()]().run()
