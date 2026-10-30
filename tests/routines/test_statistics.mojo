"""
Tests for `mean`, `variance`, `stddev`, `cov` and `corrcoef`.

The values are small enough to work by hand, and each is noted beside its
assertion. The NumPy differential tests are in `test_statistics_numpy.mojo`.

As in the reduction tests, every axis case uses a non-square matrix, so an
implementation that swaps the two axes cannot pass by accident, and strided
views are checked because every routine reads through one.

`cov` and `corrcoef` take the columns as the variables unless `rowvar` is set,
which is the opposite of NumPy's default. Several tests below exist to pin that
down.
"""

import std.testing as testing
import linamo as la
from linamo import CFloat64, Decimal
from linamo.utils.test_utils import assert_matrices_close


def _m() raises -> la.Matrix[Float64]:
    """A 2x3 matrix: non-square, so a transposed axis cannot hide."""
    return la.matrix[Float64]([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])


def _close(got: Float64, want: Float64, msg: String = "") raises:
    testing.assert_almost_equal(got, want, msg=msg, atol=1e-12)


def _close_c(got: CFloat64, want: CFloat64) raises:
    _close(got.re(), want.re(), "real part")
    _close(got.im(), want.im(), "imaginary part")


# ===----------------------------------------------------------------------===#
# mean
# ===----------------------------------------------------------------------===#


def test_mean_full() raises:
    testing.assert_equal(la.mean(_m()), 3.5)


def test_mean_axis0_averages_down_columns() raises:
    var r = la.mean(_m(), 0)
    testing.assert_equal(r.nrows(), 1)
    testing.assert_equal(r.ncols(), 3)
    testing.assert_equal(r[0, 0], 2.5)
    testing.assert_equal(r[0, 1], 3.5)
    testing.assert_equal(r[0, 2], 4.5)


def test_mean_axis1_averages_across_rows() raises:
    var r = la.mean(_m(), 1)
    testing.assert_equal(r.nrows(), 2)
    testing.assert_equal(r.ncols(), 1)
    testing.assert_equal(r[0, 0], 2.0)
    testing.assert_equal(r[1, 0], 5.0)


def test_mean_of_a_strided_view() raises:
    var a = la.matrix[Float64](
        [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
        ]
    )
    # Rows 0 and 2, columns 0 and 2: 1, 3, 9, 11.
    var v = a[0:3:2, 0:4:2]
    testing.assert_equal(la.mean(v), 6.0)
    var r = la.mean(v, 0)
    testing.assert_equal(r[0, 0], 5.0)
    testing.assert_equal(r[0, 1], 7.0)


def test_mean_float32() raises:
    var a = la.matrix[Float32]([[1.0, 2.0], [3.0, 6.0]])
    testing.assert_equal(la.mean(a), Float32(3.0))


def test_mean_of_empty_raises() raises:
    var e = la.zeros[Float64](0, 3)
    with testing.assert_raises():
        _ = la.mean(e)


def test_mean_bad_axis_raises() raises:
    with testing.assert_raises():
        _ = la.mean(_m(), 2)


# ===----------------------------------------------------------------------===#
# variance / stddev
# ===----------------------------------------------------------------------===#


def test_variance_full_population() raises:
    # 1..6 about 3.5: squared deviations sum to 17.5, over 6.
    _close(la.variance(_m()), 17.5 / 6.0)


def test_variance_full_sample() raises:
    # The same 17.5 over 5.
    _close(la.variance(_m(), ddof=1), 3.5)


def test_variance_axis0() raises:
    # Each column is {x, x + 3}: deviations of 1.5, so 2.25.
    var r = la.variance(_m(), 0)
    testing.assert_equal(r.nrows(), 1)
    testing.assert_equal(r.ncols(), 3)
    for j in range(3):
        _close(r[0, j], 2.25)


def test_variance_axis1_with_ddof() raises:
    # Each row is {x, x + 1, x + 2}: squared deviations sum to 2, over 2.
    var r = la.variance(_m(), 1, ddof=1)
    testing.assert_equal(r.nrows(), 2)
    testing.assert_equal(r.ncols(), 1)
    _close(r[0, 0], 1.0)
    _close(r[1, 0], 1.0)


def test_variance_without_degrees_of_freedom_raises() raises:
    var one = la.matrix[Float64]([[7.0]])
    _close(la.variance(one), 0.0)
    with testing.assert_raises():
        _ = la.variance(one, ddof=1)
    with testing.assert_raises():
        _ = la.variance(_m(), 0, ddof=2)


def test_stddev_is_the_root_of_variance() raises:
    _close(la.stddev(_m(), ddof=1), 3.5**0.5)
    var r = la.stddev(_m(), 0)
    for j in range(3):
        _close(r[0, j], 1.5)


def test_variance_bad_axis_raises() raises:
    with testing.assert_raises():
        _ = la.variance(_m(), 3)
    with testing.assert_raises():
        _ = la.stddev(_m(), -1)


# ===----------------------------------------------------------------------===#
# cov / corrcoef
# ===----------------------------------------------------------------------===#


def _xy() raises -> la.Matrix[Float64]:
    """Three observations of two variables, `y = 4 - x`."""
    return la.matrix[Float64]([[1.0, 3.0], [2.0, 2.0], [3.0, 1.0]])


def test_cov_takes_columns_as_variables() raises:
    var c = la.cov(_xy())
    testing.assert_equal(c.nrows(), 2)
    testing.assert_equal(c.ncols(), 2)
    var want = la.matrix[Float64]([[1.0, -1.0], [-1.0, 1.0]])
    assert_matrices_close(c, want)


def test_cov_rowvar_takes_rows_as_variables() raises:
    var c = la.cov(la.transpose(_xy()), rowvar=True)
    var want = la.matrix[Float64]([[1.0, -1.0], [-1.0, 1.0]])
    assert_matrices_close(c, want)


def test_cov_shape_follows_the_variables() raises:
    # 2x3: three variables of two observations, or two of three.
    testing.assert_equal(la.cov(_m()).nrows(), 3)
    testing.assert_equal(la.cov(_m(), rowvar=True).nrows(), 2)


def test_cov_population_ddof() raises:
    var c = la.cov(_xy(), ddof=0)
    _close(c[0, 0], 2.0 / 3.0)
    _close(c[0, 1], -2.0 / 3.0)


def test_cov_diagonal_is_the_sample_variance() raises:
    var c = la.cov(_m())
    var v = la.variance(_m(), 0, ddof=1)
    for j in range(3):
        _close(c[j, j], v[0, j])


def test_cov_without_degrees_of_freedom_raises() raises:
    var one_obs = la.matrix[Float64]([[1.0, 2.0]])
    with testing.assert_raises():
        _ = la.cov(one_obs)


def test_corrcoef_of_perfectly_anticorrelated_variables() raises:
    var r = la.corrcoef(_xy())
    var want = la.matrix[Float64]([[1.0, -1.0], [-1.0, 1.0]])
    assert_matrices_close(r, want)


def test_corrcoef_partial_correlation() raises:
    # x = 1, 2, 3, 4 and y = 1, 3, 2, 4, both with mean 2.5: the cross
    # deviations sum to 4 and each variable's squared ones to 5, so r = 0.8.
    var a = la.matrix[Float64]([[1.0, 1.0], [2.0, 3.0], [3.0, 2.0], [4.0, 4.0]])
    var r = la.corrcoef(a)
    _close(r[0, 0], 1.0)
    _close(r[0, 1], 0.8)
    _close(r[1, 0], 0.8)


def test_corrcoef_constant_variable_is_nan() raises:
    var a = la.matrix[Float64]([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]])
    var r = la.corrcoef(a)
    _close(r[0, 0], 1.0)
    testing.assert_true(r[1, 1] != r[1, 1], "expected NaN")


# ===----------------------------------------------------------------------===#
# Decimal elements
# ===----------------------------------------------------------------------===#


def _dec() raises -> la.Matrix[Decimal]:
    return la.from_string[Decimal]("[[0.1, 0.2], [0.3, 0.4]]")


def test_decimal_mean_is_exact() raises:
    testing.assert_true(la.mean(_dec()) == Decimal("0.25"))
    var r = la.mean(_dec(), 0)
    testing.assert_true(r[0, 0] == Decimal("0.2"))
    testing.assert_true(r[0, 1] == Decimal("0.3"))


def test_decimal_variance_is_exact() raises:
    # Deviations 0.15, 0.05, 0.05, 0.15: squares sum to 0.05, over 4.
    testing.assert_true(la.variance(_dec()) == Decimal("0.0125"))
    var r = la.variance(_dec(), 1)
    testing.assert_true(r[0, 0] == Decimal("0.0025"))


def test_decimal_stddev() raises:
    var a = la.from_string[Decimal]("[[1], [3]]")
    testing.assert_true(la.stddev(a) == Decimal("1"))


def test_decimal_cov_and_corrcoef() raises:
    var a = la.from_string[Decimal]("[[1, 3], [2, 2], [3, 1]]")
    var c = la.cov(a)
    testing.assert_true(c[0, 0] == Decimal("1"))
    testing.assert_true(c[0, 1] == Decimal("-1"))
    var r = la.corrcoef(a)
    testing.assert_true(r[1, 1] == Decimal("1"))
    testing.assert_true(r[1, 0] == Decimal("-1"))


def test_decimal_corrcoef_constant_variable_raises() raises:
    # Where a float gives NaN, a decimal division by zero raises.
    var a = la.from_string[Decimal]("[[1, 5], [2, 5], [3, 5]]")
    with testing.assert_raises():
        _ = la.corrcoef(a)


# ===----------------------------------------------------------------------===#
# Complex elements
# ===----------------------------------------------------------------------===#


def test_complex_mean() raises:
    var a = la.matrix[CFloat64]([[CFloat64(1, 1), CFloat64(3, -1)]])
    testing.assert_equal(la.mean(a), CFloat64(2, 0))


def test_complex_variance_uses_the_modulus() raises:
    # Deviations i and -i about a mean of 1. Squaring them would give
    # -1 + -1 = -2; the modulus gives 1 + 1 = 2, over 2.
    var a = la.matrix[CFloat64]([[CFloat64(1, 1)], [CFloat64(1, -1)]])
    testing.assert_equal(la.variance(a), CFloat64(1, 0))
    testing.assert_equal(la.stddev(a), CFloat64(1, 0))
    var r = la.variance(a, 0)
    testing.assert_equal(r[0, 0], CFloat64(1, 0))


def test_complex_cov_is_hermitian() raises:
    # Columns a = [i, -i] and b = [1, -1], both with mean 0:
    # sum a conj(a) = 2, sum a conj(b) = 2i, sum b conj(a) = -2i.
    var x = la.matrix[CFloat64](
        [[CFloat64(0, 1), CFloat64(1, 0)], [CFloat64(0, -1), CFloat64(-1, 0)]]
    )
    var c = la.cov(x)
    testing.assert_equal(c[0, 0], CFloat64(2, 0))
    testing.assert_equal(c[0, 1], CFloat64(0, 2))
    testing.assert_equal(c[1, 0], CFloat64(0, -2))
    testing.assert_equal(c[1, 1], CFloat64(2, 0))
    # Each entry is divided by sqrt(2) twice, so the ones are only close.
    var r = la.corrcoef(x)
    _close_c(r[0, 0], CFloat64(1, 0))
    _close_c(r[0, 1], CFloat64(0, 1))
    _close_c(r[1, 0], CFloat64(0, -1))


def main() raises:
    testing.TestSuite.discover_tests[__functions_in_module()]().run()
