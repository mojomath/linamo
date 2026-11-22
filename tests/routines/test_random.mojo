"""
Tests for random matrix generation: rand, randn and seed.
"""

import std.testing as testing
import linamo as la
from linamo.routines.random import rand, randn, seed


def test_rand_shape_and_layout() raises:
    """Test that rand produces the requested shape, C-contiguous."""
    var mat = rand[Float64](3, 4)
    testing.assert_equal(mat.nrows(), 3)
    testing.assert_equal(mat.ncols(), 4)
    testing.assert_true(mat.is_c_contiguous())


def test_rand_default_range() raises:
    """Test that the default range is [0, 1]."""
    var mat = rand[Float64](4, 4)
    for i in range(4):
        for j in range(4):
            testing.assert_true(mat[i, j] >= 0.0 and mat[i, j] <= 1.0)


def test_rand_explicit_range() raises:
    """Test that every element lands inside an explicit range."""
    var mat = rand[Float64](5, 5, -2.0, 2.0)
    for i in range(5):
        for j in range(5):
            testing.assert_true(mat[i, j] >= -2.0 and mat[i, j] <= 2.0)


def test_rand_is_reproducible_under_seed() raises:
    """Test that the same seed produces the same matrix."""
    seed(42)
    var a = rand[Float64](2, 3)
    seed(42)
    var b = rand[Float64](2, 3)
    for i in range(2):
        for j in range(3):
            testing.assert_equal(a[i, j], b[i, j])


def test_rand_differs_across_seeds() raises:
    """Test that different seeds produce different matrices."""
    seed(1)
    var a = rand[Float64](3, 3)
    seed(2)
    var b = rand[Float64](3, 3)
    var identical = True
    for i in range(3):
        for j in range(3):
            if a[i, j] != b[i, j]:
                identical = False
    testing.assert_false(identical, "Two seeds should not agree element-wise")


def test_rand_is_not_constant() raises:
    """Test that the elements are not all the same value."""
    seed(7)
    var mat = rand[Float64](4, 4)
    var all_same = True
    for i in range(4):
        for j in range(4):
            if mat[i, j] != mat[0, 0]:
                all_same = False
    testing.assert_false(all_same, "Random elements should not all be equal")


def test_rand_integer_dtype() raises:
    """Test that an integer dtype draws integers inside the closed range."""
    seed(3)
    var mat = rand[Int64](4, 4, 1, 6)
    for i in range(4):
        for j in range(4):
            testing.assert_true(mat[i, j] >= 1 and mat[i, j] <= 6)


def test_rand_degenerate_range() raises:
    """Test that low == high fills the matrix with that value."""
    var mat = rand[Int64](2, 2, 5, 5)
    for i in range(2):
        for j in range(2):
            testing.assert_equal(mat[i, j], Int64(5))


def test_rand_low_above_high_raises() raises:
    """Test that low greater than high raises."""
    var raised = False
    try:
        var _m = rand[Float64](2, 2, 1.0, 0.0)
    except:
        raised = True
    testing.assert_true(raised, "low > high should raise")


def test_rand_bounds_of_any_scalar_type() raises:
    """Test that `low` and `high` need not match the element type."""
    seed(4)
    var f = rand[Float32](3, 3, -2.0, 2.0)  # float64 literals
    var lo: Int32 = 1
    var hi: Float64 = 6.0
    var k = rand[Int64](3, 3, lo, hi)  # two other dtypes
    for i in range(3):
        for j in range(3):
            testing.assert_true(f[i, j] >= -2.0 and f[i, j] <= 2.0)
            testing.assert_true(k[i, j] >= 1 and k[i, j] <= 6)


def test_rand_integer_dtype_rejects_fractional_bounds() raises:
    """Test that an integer element type refuses a bound it would truncate."""
    with testing.assert_raises(contains="whole numbers"):
        _ = rand[Int64](2, 2, 1.5, 6.0)


def test_rand_integer_dtype_rejects_out_of_range_bounds() raises:
    """Test that an integer element type refuses a bound it cannot hold."""
    with testing.assert_raises(contains="whole numbers"):
        _ = rand[Int8](2, 2, 0, 1000)
    with testing.assert_raises(contains="whole numbers"):
        _ = rand[UInt8](2, 2, -1, 5)


def test_rand_exported_from_package() raises:
    """Test that rand and seed are reachable through the package alias."""
    la.seed(11)
    var mat = la.rand[Float64](2, 2)
    testing.assert_equal(mat.nrows(), 2)


# ===----------------------------------------------------------------------===#
# randn
# ===----------------------------------------------------------------------===#
# Every sample below is drawn under a fixed seed, so the statistics are the
# same on every run and the tolerances cannot flake. With 40000 draws the
# standard error of the mean is stddev / 200, far inside them.


def test_randn_shape_and_layout() raises:
    """Test that randn produces the requested shape, C-contiguous."""
    var mat = randn[Float64](3, 4)
    testing.assert_equal(mat.nrows(), 3)
    testing.assert_equal(mat.ncols(), 4)
    testing.assert_true(mat.is_c_contiguous())


def test_randn_is_standard_normal_by_default() raises:
    """Test that the default sample has mean near 0 and stddev near 1."""
    seed(5)
    var mat = randn[Float64](200, 200)
    testing.assert_almost_equal(la.mean(mat), 0.0, atol=0.05)
    testing.assert_almost_equal(la.stddev(mat), 1.0, atol=0.05)


def test_randn_mean_and_stddev() raises:
    """Test that `mean` shifts and `stddev` scales the distribution."""
    seed(6)
    var mat = randn[Float64](200, 200, 2.0, 3.0)
    testing.assert_almost_equal(la.mean(mat), 2.0, atol=0.1)
    testing.assert_almost_equal(la.stddev(mat), 3.0, atol=0.1)


def test_randn_has_negative_values() raises:
    """Test that the draws fall on both sides of the mean, unlike `rand`."""
    seed(7)
    var mat = randn[Float64](10, 10)
    var negatives = 0
    for i in range(10):
        for j in range(10):
            if mat[i, j] < 0:
                negatives += 1
    testing.assert_true(negatives > 20 and negatives < 80)


def test_randn_is_reproducible_under_seed() raises:
    """Test that the same seed gives the same matrix."""
    seed(42)
    var a = randn[Float64](3, 3)
    seed(42)
    var b = randn[Float64](3, 3)
    for i in range(3):
        for j in range(3):
            testing.assert_equal(a[i, j], b[i, j])


def test_randn_differs_across_seeds() raises:
    """Test that different seeds give different matrices."""
    seed(1)
    var a = randn[Float64](3, 3)
    seed(2)
    var b = randn[Float64](3, 3)
    var differ = False
    for i in range(3):
        for j in range(3):
            if a[i, j] != b[i, j]:
                differ = True
    testing.assert_true(differ)


def test_randn_zero_stddev_gives_the_mean() raises:
    """Test that `stddev=0` fills the matrix with the mean."""
    var mat = randn[Float64](2, 3, 1.5, 0.0)
    for i in range(2):
        for j in range(3):
            testing.assert_equal(mat[i, j], 1.5)


def test_randn_float32() raises:
    """Test that a `Float32` matrix comes out with a standard normal sample."""
    seed(8)
    var mat = randn[Float32](200, 200)
    testing.assert_almost_equal(Float64(la.mean(mat)), 0.0, atol=0.05)


def test_randn_mean_and_stddev_of_any_scalar_type() raises:
    """Test that `mean` and `stddev` need not match the element type."""
    seed(9)
    var a = randn[Float32](200, 200, 2.0, 3.0)  # float64 literals
    testing.assert_almost_equal(Float64(la.mean(a)), 2.0, atol=0.1)
    testing.assert_almost_equal(Float64(la.stddev(a)), 3.0, atol=0.1)
    var mu: Float32 = 1.0
    var sigma: Int32 = 0
    var b = randn[Float64](2, 2, mu, sigma)  # two other dtypes
    testing.assert_equal(b[1, 1], 1.0)


def test_randn_negative_stddev_raises() raises:
    """Test that a negative `stddev` raises."""
    with testing.assert_raises(contains="must not be negative"):
        _ = randn[Float64](2, 2, 0.0, -1.0)


def test_randn_exported_from_package() raises:
    """Test that randn is reachable through the package alias."""
    la.seed(11)
    var mat = la.randn[Float64](2, 2)
    testing.assert_equal(mat.nrows(), 2)


def main() raises:
    testing.TestSuite.discover_tests[__functions_in_module()]().run()
