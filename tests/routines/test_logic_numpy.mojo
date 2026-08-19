"""
Numpy-powered ground-truth tests for the closeness and logical routines.

These two families are copied from NumPy's semantics rather than invented, and
the places they can quietly diverge are the ones no hand-written expectation
catches: the exact tolerance formula, its asymmetry in the operands, and what
`isclose` does at the infinities. Every test here asks numpy for the answer
and compares element for element.
"""

import std.testing as testing
import linamo as la
from linamo.routines.logic import (
    allclose,
    isclose,
    logical_and,
    logical_not,
    logical_or,
    logical_xor,
)
from linamo.routines.numpy_interop import from_numpy, to_numpy
from linamo.types.matrix import Matrix
from std.python import Python, PythonObject


def _assert_same_mask(
    mask: Matrix[DType.bool], np_mask: PythonObject, msg: String
) raises:
    """Assert a Linamo mask matches a numpy boolean array element for element.

    The numpy mask crosses over as floats rather than through element-by-
    element `PythonObject` access: `from_numpy` is the bridge the rest of the
    suite uses, and one `astype` is cheaper than r*c round trips into Python.
    """
    var np = Python.import_module("numpy")
    var truth = from_numpy(np_mask.astype(np.float64))
    testing.assert_equal(mask.nrows(), truth.nrows(), msg=msg)
    testing.assert_equal(mask.ncols(), truth.ncols(), msg=msg)
    for i in range(mask.nrows()):
        for j in range(mask.ncols()):
            testing.assert_equal(
                Bool(mask[i, j]),
                truth[i, j] != 0.0,
                msg=msg + " at (" + String(i) + ", " + String(j) + ")",
            )


def test_isclose_matches_numpy_on_random_data() raises:
    """Test the mask against numpy on data straddling the tolerance."""
    var np = Python.import_module("numpy")
    var a_np = np.random.rand(5, 7)
    # A perturbation of the same order as the default rtol, so that roughly
    # half the entries land on each side of the boundary.
    var b_np = a_np * (1.0 + (np.random.rand(5, 7) * 4e-5 - 2e-5))
    var a = from_numpy(a_np)
    var b = from_numpy(b_np)
    _assert_same_mask(isclose(a, b), np.isclose(a_np, b_np), "isclose")


def test_isclose_matches_numpy_with_custom_tolerances() raises:
    """Test that rtol and atol are read the same way numpy reads them."""
    var np = Python.import_module("numpy")
    var a_np = np.array([[0.0, 1e-9, 1.0, 100.0], [1e-3, 2.0, -1.0, 0.5]])
    var b_np = np.array([[1e-9, 0.0, 1.001, 100.01], [0.0, 2.0, 1.0, 0.5]])
    var a = from_numpy(a_np)
    var b = from_numpy(b_np)
    _assert_same_mask(
        isclose(a, b, rtol=1e-3, atol=1e-8),
        np.isclose(a_np, b_np, rtol=1e-3, atol=1e-8),
        "isclose rtol=1e-3",
    )
    _assert_same_mask(
        isclose(a, b, rtol=0.0, atol=0.0),
        np.isclose(a_np, b_np, rtol=0.0, atol=0.0),
        "isclose zero tolerance",
    )


def test_isclose_matches_numpy_on_non_finite_values() raises:
    """Test the infinity and NaN rules, both with and without equal_nan."""
    var np = Python.import_module("numpy")
    var inf = Float64("inf")
    var nan = Float64("nan")
    var a = la.matrix[DType.float64](
        [[inf, inf, inf, nan], [nan, -inf, 1.0, -inf]]
    )
    var b = la.matrix[DType.float64](
        [[inf, -inf, 1e308, nan], [1.0, -inf, inf, inf]]
    )
    var a_np = np.array([[inf, inf, inf, nan], [nan, -inf, 1.0, -inf]])
    var b_np = np.array([[inf, -inf, 1e308, nan], [1.0, -inf, inf, inf]])
    _assert_same_mask(
        isclose(a, b), np.isclose(a_np, b_np), "isclose non-finite"
    )
    _assert_same_mask(
        isclose(a, b, equal_nan=True),
        np.isclose(a_np, b_np, equal_nan=True),
        "isclose equal_nan",
    )


def test_isclose_asymmetry_matches_numpy() raises:
    """Test that swapping the operands changes the answer the same way.

    `|a - b| <= atol + rtol * |b|` reads `b` as the reference, so the mask is
    not symmetric. Comparing both orders against numpy pins the convention.
    """
    var np = Python.import_module("numpy")
    var a_np = np.array([[1.0, 1e-8, 100.0], [0.0, 3.0, 1e6]])
    var b_np = np.array([[1.001, 0.0, 100.1], [1e-8, 3.0, 1e6 + 5.0]])
    var a = from_numpy(a_np)
    var b = from_numpy(b_np)
    _assert_same_mask(
        isclose(a, b, rtol=1e-4, atol=0.0),
        np.isclose(a_np, b_np, rtol=1e-4, atol=0.0),
        "isclose(a, b)",
    )
    _assert_same_mask(
        isclose(b, a, rtol=1e-4, atol=0.0),
        np.isclose(b_np, a_np, rtol=1e-4, atol=0.0),
        "isclose(b, a)",
    )


def test_allclose_matches_numpy() raises:
    """Test the reduction against numpy, on data that agrees and data that does
    not."""
    var np = Python.import_module("numpy")
    var a_np = np.random.rand(4, 6)
    var same_np = a_np * (1.0 + 1e-9)
    var a = from_numpy(a_np)
    testing.assert_equal(
        allclose(a, from_numpy(same_np)),
        Bool(np.allclose(a_np, same_np)),
        msg="allclose on matching data",
    )

    # One element moved well outside the tolerance, written on the Linamo side
    # and handed back to numpy, so both libraries reduce the same array.
    var off = from_numpy(a_np)
    off.set(2, 3, off[2, 3] + 0.5)
    var off_np = to_numpy(off)
    testing.assert_equal(
        allclose(a, off),
        Bool(np.allclose(a_np, off_np)),
        msg="allclose on one differing element",
    )


def test_logical_connectives_match_numpy() raises:
    """Test the four connectives against numpy on mixed non-zero data."""
    var np = Python.import_module("numpy")
    var a_np = np.array([[0.0, 1.0, 0.0, 2.0], [-1.0, 0.0, 3.0, 0.0]])
    var b_np = np.array([[0.0, 0.0, 1.0, 1.0], [2.0, 2.0, 0.0, 0.0]])
    var a = from_numpy(a_np)
    var b = from_numpy(b_np)
    _assert_same_mask(
        logical_and(a, b), np.logical_and(a_np, b_np), "logical_and"
    )
    _assert_same_mask(logical_or(a, b), np.logical_or(a_np, b_np), "logical_or")
    _assert_same_mask(
        logical_xor(a, b), np.logical_xor(a_np, b_np), "logical_xor"
    )
    _assert_same_mask(logical_not(a), np.logical_not(a_np), "logical_not")


def test_logical_connectives_match_numpy_on_masks() raises:
    """Test the connectives on boolean operands, the common case."""
    var np = Python.import_module("numpy")
    var m_np = np.random.rand(3, 5)
    var m = from_numpy(m_np)
    var lo_np = np.greater(m_np, 0.25)
    var hi_np = np.less(m_np, 0.75)
    _assert_same_mask(
        logical_and(m > 0.25, m < 0.75),
        np.logical_and(lo_np, hi_np),
        "logical_and on masks",
    )
    _assert_same_mask(
        logical_not(m > 0.25), np.logical_not(lo_np), "logical_not on a mask"
    )


def main() raises:
    testing.TestSuite.discover_tests[__functions_in_module()]().run()
