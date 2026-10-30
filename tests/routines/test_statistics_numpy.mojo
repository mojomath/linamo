"""
NumPy differential tests for `mean`, `variance`, `stddev`, `cov` and
`corrcoef`.

Every result is checked against NumPy on random non-square data. NumPy's
`cov` and `corrcoef` take the *rows* as the variables by default and Linamo's
take the columns, so each comparison spells out `rowvar` on the NumPy side.
"""

import std.testing as testing
import linamo as la
from linamo.routines.numpy_interop import from_numpy
from linamo.utils.test_utils import assert_matrices_close
from std.python import Python, PythonObject


def _data(np: PythonObject, r: Int, c: Int) raises -> PythonObject:
    """Random data with a spread of scales, so the columns differ."""
    return np.random.rand(r, c) * np.arange(1, c + 1) + np.arange(c)


def test_mean_variance_stddev_full() raises:
    var np = Python.import_module("numpy")
    np.random.seed(11)
    var a_np = _data(np, 7, 4)
    var a = from_numpy(a_np)
    testing.assert_almost_equal(
        la.mean(a), Float64(py=np.mean(a_np)), atol=1e-12
    )
    for ddof in range(2):
        testing.assert_almost_equal(
            la.variance(a, ddof=ddof),
            Float64(py=np.var(a_np, ddof=ddof)),
            atol=1e-12,
        )
        testing.assert_almost_equal(
            la.stddev(a, ddof=ddof),
            Float64(py=np.std(a_np, ddof=ddof)),
            atol=1e-12,
        )


def test_mean_variance_stddev_by_axis() raises:
    var np = Python.import_module("numpy")
    np.random.seed(12)
    var a_np = _data(np, 6, 5)
    var a = from_numpy(a_np)
    for axis in range(2):
        var label = "axis=" + String(axis)
        assert_matrices_close(
            la.mean(a, axis),
            from_numpy(np.mean(a_np, axis=axis, keepdims=True)),
            msg="mean " + label,
        )
        for ddof in range(2):
            assert_matrices_close(
                la.variance(a, axis, ddof=ddof),
                from_numpy(np.var(a_np, axis=axis, ddof=ddof, keepdims=True)),
                msg="variance " + label,
            )
            assert_matrices_close(
                la.stddev(a, axis, ddof=ddof),
                from_numpy(np.std(a_np, axis=axis, ddof=ddof, keepdims=True)),
                msg="stddev " + label,
            )


def test_cov_matches_numpy_in_both_layouts() raises:
    var np = Python.import_module("numpy")
    np.random.seed(13)
    var a_np = _data(np, 9, 4)
    var a = from_numpy(a_np)
    for ddof in range(3):
        assert_matrices_close(
            la.cov(a, ddof=ddof),
            from_numpy(np.cov(a_np, rowvar=False, ddof=ddof)),
            msg="cov, columns as variables",
        )
        assert_matrices_close(
            la.cov(a, rowvar=True, ddof=ddof),
            from_numpy(np.cov(a_np, rowvar=True, ddof=ddof)),
            msg="cov, rows as variables",
        )


def test_corrcoef_matches_numpy_in_both_layouts() raises:
    var np = Python.import_module("numpy")
    np.random.seed(14)
    var a_np = _data(np, 8, 5)
    var a = from_numpy(a_np)
    assert_matrices_close(
        la.corrcoef(a),
        from_numpy(np.corrcoef(a_np, rowvar=False)),
        msg="corrcoef, columns as variables",
    )
    assert_matrices_close(
        la.corrcoef(a, rowvar=True),
        from_numpy(np.corrcoef(a_np, rowvar=True)),
        msg="corrcoef, rows as variables",
    )


def test_statistics_of_a_strided_view() raises:
    var np = Python.import_module("numpy")
    np.random.seed(15)
    var a_np = _data(np, 10, 8)
    var a = from_numpy(a_np)
    var v = a[1:10:2, 0:8:3]
    var v_np = a_np[1:10:2, 0:8:3]
    assert_matrices_close(
        la.variance(v, 0, ddof=1),
        from_numpy(np.var(v_np, axis=0, ddof=1, keepdims=True)),
        msg="variance of a view",
    )
    assert_matrices_close(
        la.cov(v),
        from_numpy(np.cov(v_np, rowvar=False)),
        msg="cov of a view",
    )
    assert_matrices_close(
        la.corrcoef(v),
        from_numpy(np.corrcoef(v_np, rowvar=False)),
        msg="corrcoef of a view",
    )


def main() raises:
    testing.TestSuite.discover_tests[__functions_in_module()]().run()
