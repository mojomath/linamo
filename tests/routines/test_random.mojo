"""
Tests for random matrix generation: rand and seed.
"""

import std.testing as testing
import linamo as la
from linamo.routines.random import rand, seed


def test_rand_shape_and_layout() raises:
    """Test that rand produces the requested shape, C-contiguous."""
    var mat = rand[DType.float64](3, 4)
    testing.assert_equal(mat.nrows(), 3)
    testing.assert_equal(mat.ncols(), 4)
    testing.assert_true(mat.is_c_contiguous())


def test_rand_default_range() raises:
    """Test that the default range is [0, 1]."""
    var mat = rand[DType.float64](4, 4)
    for i in range(4):
        for j in range(4):
            testing.assert_true(mat[i, j] >= 0.0 and mat[i, j] <= 1.0)


def test_rand_explicit_range() raises:
    """Test that every element lands inside an explicit range."""
    var mat = rand[DType.float64](5, 5, -2.0, 2.0)
    for i in range(5):
        for j in range(5):
            testing.assert_true(mat[i, j] >= -2.0 and mat[i, j] <= 2.0)


def test_rand_is_reproducible_under_seed() raises:
    """Test that the same seed produces the same matrix."""
    seed(42)
    var a = rand[DType.float64](2, 3)
    seed(42)
    var b = rand[DType.float64](2, 3)
    for i in range(2):
        for j in range(3):
            testing.assert_equal(a[i, j], b[i, j])


def test_rand_differs_across_seeds() raises:
    """Test that different seeds produce different matrices."""
    seed(1)
    var a = rand[DType.float64](3, 3)
    seed(2)
    var b = rand[DType.float64](3, 3)
    var identical = True
    for i in range(3):
        for j in range(3):
            if a[i, j] != b[i, j]:
                identical = False
    testing.assert_false(identical, "Two seeds should not agree element-wise")


def test_rand_is_not_constant() raises:
    """Test that the elements are not all the same value."""
    seed(7)
    var mat = rand[DType.float64](4, 4)
    var all_same = True
    for i in range(4):
        for j in range(4):
            if mat[i, j] != mat[0, 0]:
                all_same = False
    testing.assert_false(all_same, "Random elements should not all be equal")


def test_rand_integer_dtype() raises:
    """Test that an integer dtype draws integers inside the closed range."""
    seed(3)
    var mat = rand[DType.int64](4, 4, 1, 6)
    for i in range(4):
        for j in range(4):
            testing.assert_true(mat[i, j] >= 1 and mat[i, j] <= 6)


def test_rand_degenerate_range() raises:
    """Test that low == high fills the matrix with that value."""
    var mat = rand[DType.int64](2, 2, 5, 5)
    for i in range(2):
        for j in range(2):
            testing.assert_equal(mat[i, j], Int64(5))


def test_rand_low_above_high_raises() raises:
    """Test that low greater than high raises."""
    var raised = False
    try:
        var _m = rand[DType.float64](2, 2, 1.0, 0.0)
    except:
        raised = True
    testing.assert_true(raised, "low > high should raise")


def test_rand_exported_from_package() raises:
    """Test that rand and seed are reachable through the package alias."""
    la.seed(11)
    var mat = la.rand[DType.float64](2, 2)
    testing.assert_equal(mat.nrows(), 2)


def main() raises:
    testing.TestSuite.discover_tests[__functions_in_module()]().run()
