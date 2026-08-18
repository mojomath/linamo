from linamo.types.matrix import Matrix
from linamo.types.matrix_view import MatrixView
from linamo.types.static_matrix import StaticMatrix

from linamo.routines.creation import (
    matrix,
    smatrix,
    zeros,
    ones,
    full,
    eye,
    identity,
    diag,
)
from linamo.routines.linalg import (
    transpose,
    trace,
    lu,
    cholesky,
    qr,
    det,
    solve,
    inv,
    lstsq,
)
from linamo.routines.functional import apply_along_axis, fold
from linamo.routines.logic import all, any
from linamo.routines.math import max, min, prod, cumprod
from linamo.routines.searching import argmax, argmin
from linamo.routines.sorting import argsort, sort, sort_inplace
from linamo.routines.statistics import cumsum, sum
from linamo.routines.numpy_interop import matrix_from_numpy, to_numpy
from linamo.utils.test_utils import (
    assert_matrices_equal,
    assert_matrices_close,
)
