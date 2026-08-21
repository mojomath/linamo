"""
Defines linear algebra routines for matrices.
"""

from std.math import sqrt

from decimo import Numeric

import linamo.routines.math

from linamo.types.errors import ValueError
from linamo.types.matrix import Matrix
from linamo.types.matrix_view import MatrixView
from linamo.utils.indexing import get_offset

# ===---------------------------------------------------------------------- ===#
# Transpose
# ===---------------------------------------------------------------------- ===#


def transpose[
    T: Copyable & Deinitable, origin: Origin, //
](view: MatrixView[T, origin]) -> Matrix[T]:
    """Returns the transpose of a matrix view.

    The result is always stored in row-major (C) order regardless of the
    input layout.

    Transposing only moves elements, so this is generic over the element type
    and works for a `Matrix[BigInt]` as readily as a `Matrix[Float64]`. The
    loop walks the *result* in buffer order rather than scattering into
    uninitialised storage, which is what makes it safe for an element that
    owns a heap allocation.

    Parameters:
        T: The type of the matrix elements.
        origin: The origin of the input view.

    Args:
        view: The matrix or view to transpose.

    Returns:
        A new C-contiguous matrix with the rows and columns exchanged.
    """
    var nrows = view.ncols()  # transposed
    var ncols = view.nrows()  # transposed
    var data = List[T](capacity=nrows * ncols)
    for i in range(nrows):
        for j in range(ncols):
            data.append(view[j, i].copy())
    return Matrix[T](
        buffer=data^,
        nrows=nrows,
        ncols=ncols,
        row_stride=ncols,
        col_stride=1,
    )


def trace[
    dtype: DType, origin: Origin, //
](view: MatrixView[Scalar[dtype], origin]) raises -> Scalar[dtype]:
    """Computes the trace (sum of diagonal elements) of a square matrix view."""
    if view.nrows() != view.ncols():
        raise ValueError(
            function="trace()",
            message="Matrix must be square to compute trace.",
        )
    var result: Scalar[dtype] = 0
    for i in range(view.nrows()):
        result += view[i, i]
    return result


def trace[
    T: Numeric, origin: Origin, //
](view: MatrixView[T, origin]) raises -> T:
    """Computes the trace (sum of diagonal elements) of a square matrix view.

    Parameters:
        T: The type of the matrix elements.
        origin: The origin of the operand.

    Args:
        view: The matrix or view to reduce. Must be square.

    Raises:
        ValueError: If the matrix is not square.

    Returns:
        The sum of the diagonal elements.
    """
    if view.nrows() != view.ncols():
        raise ValueError(
            function="trace()",
            message="Matrix must be square to compute trace.",
        )
    var acc = T.zero()
    for i in range(view.nrows()):
        acc = acc + view[i, i]
    return acc^


def lu[
    dtype: DType, origin: Origin, //
](
    view: MatrixView[Scalar[dtype], origin],
) raises -> Tuple[
    Matrix[Scalar[dtype]], Matrix[Scalar[dtype]], List[Int]
]:
    """Computes the LU decomposition with partial pivoting: PA = LU.

    The input matrix view is decomposed into a unit lower-triangular matrix L,
    an upper-triangular matrix U, and a permutation vector P such that
    the rows of A permuted by P equal L @ U.
    """
    if view.nrows() != view.ncols():
        raise ValueError(
            function="lu()",
            message="Matrix must be square for LU decomposition.",
        )
    var n = view.nrows()

    # Work on a row-major copy for uniform indexing.
    var u_data = List[Scalar[dtype]](unsafe_uninit_length=n * n)
    for i in range(n):
        for j in range(n):
            u_data[i * n + j] = view[i, j]

    var l_data = List[Scalar[dtype]](length=n * n, fill=0)

    # Initialise piv as identity permutation.
    var piv = List[Int](unsafe_uninit_length=n)
    for i in range(n):
        piv[i] = i

    for k in range(n):
        # --- partial pivoting: find row with largest |u[i,k]| for i >= k ---
        var max_val: Scalar[dtype] = 0
        var max_row = k
        for i in range(k, n):
            var val = u_data[i * n + k]
            if val < 0:
                val = -val
            if val > max_val:
                max_val = val
                max_row = i

        # Swap rows in u_data
        if max_row != k:
            for j in range(n):
                var tmp = u_data[k * n + j]
                u_data[k * n + j] = u_data[max_row * n + j]
                u_data[max_row * n + j] = tmp
            # Swap already-computed L columns
            for j in range(k):
                var tmp = l_data[k * n + j]
                l_data[k * n + j] = l_data[max_row * n + j]
                l_data[max_row * n + j] = tmp
            # Swap piv
            var tmp_piv = piv[k]
            piv[k] = piv[max_row]
            piv[max_row] = tmp_piv

        # --- elimination ---
        var pivot = u_data[k * n + k]
        for i in range(k + 1, n):
            var factor = u_data[i * n + k] / pivot
            l_data[i * n + k] = factor
            for j in range(k, n):
                u_data[i * n + j] = (
                    u_data[i * n + j] - factor * u_data[k * n + j]
                )

    # Set L diagonal to 1.
    for i in range(n):
        l_data[i * n + i] = 1

    # Zero out below-diagonal in U.
    for i in range(n):
        for j in range(i):
            u_data[i * n + j] = 0

    var L = Matrix[Scalar[dtype]](
        buffer=l_data^, nrows=n, ncols=n, row_stride=n, col_stride=1
    )
    var U = Matrix[Scalar[dtype]](
        buffer=u_data^, nrows=n, ncols=n, row_stride=n, col_stride=1
    )
    return (L^, U^, piv^)


# ===---------------------------------------------------------------------- ===#
# Elimination over arbitrary-precision elements
# ===---------------------------------------------------------------------- ===#
# `lu`, `det`, `solve` and `inv` each carry a second implementation below the
# scalar one, reached when the element type is `decimo.Numeric`. The algorithm
# is the same in both; what differs is that there is no vector instruction to
# reach for, the constants are `T.zero()` and `T.one()` rather than literals,
# and ordering comes from `Comparable`.
#
# `Comparable` is in the bound because partial pivoting has to rank candidate
# pivots by magnitude. An `abs` requirement would be redundant on top of it:
# `-x if x < T.zero() else x` is built from what `Numeric` and `Comparable`
# already provide.
#
# These four routines divide, so they mean whatever `/` means on the element
# type. `BigDecimal` and `Decimal128` give a quotient rounded to the type's
# precision, and the answers are the usual approximate ones carrying more
# digits than `Float64` would. `BigInt` truncates toward zero, and an integer
# matrix has no integer inverse in general, so a solve or an inverse over
# `BigInt` returns whole numbers that answer nothing: use a decimal element
# type for these.


def lu[
    T: Numeric & Comparable, origin: Origin, //
](view: MatrixView[T, origin]) raises -> Tuple[Matrix[T], Matrix[T], List[Int]]:
    """Computes the LU decomposition with partial pivoting: PA = LU.

    The input matrix view is decomposed into a unit lower-triangular matrix L,
    an upper-triangular matrix U, and a permutation vector P such that the rows
    of A permuted by P equal L @ U.

    A singular matrix decomposes without complaint and leaves a zero on the
    diagonal of U, which is what lets `det` report zero for it. The routines
    that go on to divide by that diagonal are the ones that raise.

    Parameters:
        T: The type of the matrix elements.
        origin: The origin of the operand.

    Args:
        view: The square matrix or view to decompose.

    Returns:
        The triple `(L, U, P)`.

    Raises:
        ValueError: If the matrix is not square.
    """
    if view.nrows() != view.ncols():
        raise ValueError(
            function="lu()",
            message="Matrix must be square for LU decomposition.",
        )
    var n = view.nrows()
    var zero = T.zero()

    # Work on a row-major copy for uniform indexing.
    var u_data = List[T](length=n * n, fill=zero)
    for i in range(n):
        for j in range(n):
            u_data[i * n + j] = view[i, j].copy()

    var l_data = List[T](length=n * n, fill=zero)

    # Initialise piv as identity permutation.
    var piv = List[Int](unsafe_uninit_length=n)
    for i in range(n):
        piv[i] = i

    for k in range(n):
        # --- partial pivoting: find row with largest |u[i,k]| for i >= k ---
        var max_val = zero.copy()
        var max_row = k
        for i in range(k, n):
            var val = u_data[i * n + k].copy()
            if val < zero:
                val = -val
            if val > max_val:
                max_val = val^
                max_row = i

        # Swap rows in u_data
        if max_row != k:
            for j in range(n):
                var tmp = u_data[k * n + j].copy()
                u_data[k * n + j] = u_data[max_row * n + j].copy()
                u_data[max_row * n + j] = tmp^
            # Swap already-computed L columns
            for j in range(k):
                var tmp = l_data[k * n + j].copy()
                l_data[k * n + j] = l_data[max_row * n + j].copy()
                l_data[max_row * n + j] = tmp^
            # Swap piv
            var tmp_piv = piv[k]
            piv[k] = piv[max_row]
            piv[max_row] = tmp_piv

        # --- elimination ---
        var pivot = u_data[k * n + k].copy()
        if pivot == zero:
            # Nothing in this column to eliminate with. Dividing would raise,
            # and the zero left on U's diagonal is the honest result.
            continue
        for i in range(k + 1, n):
            var factor = u_data[i * n + k] / pivot
            for j in range(k, n):
                u_data[i * n + j] = (
                    u_data[i * n + j] - factor * u_data[k * n + j]
                )
            l_data[i * n + k] = factor^

    # Set L diagonal to 1.
    var one = T.one()
    for i in range(n):
        l_data[i * n + i] = one.copy()

    # Zero out below-diagonal in U.
    for i in range(n):
        for j in range(i):
            u_data[i * n + j] = zero.copy()

    var L = Matrix[T](
        buffer=l_data^, nrows=n, ncols=n, row_stride=n, col_stride=1
    )
    var U = Matrix[T](
        buffer=u_data^, nrows=n, ncols=n, row_stride=n, col_stride=1
    )
    return (L^, U^, piv^)


def cholesky[
    dtype: DType, origin: Origin, //
](view: MatrixView[Scalar[dtype], origin]) raises -> Matrix[Scalar[dtype]]:
    """Computes the Cholesky decomposition: A = L L^T.

    The input must be a symmetric positive-definite matrix view. The result is
    a lower-triangular matrix L such that A = L @ L^T.
    """
    if view.nrows() != view.ncols():
        raise ValueError(
            function="cholesky()",
            message="Matrix must be square for Cholesky decomposition.",
        )
    var n = view.nrows()
    var l_data = List[Scalar[dtype]](length=n * n, fill=0)

    for i in range(n):
        for j in range(i + 1):
            var s: Scalar[dtype] = 0
            for k in range(j):
                s += l_data[i * n + k] * l_data[j * n + k]

            if i == j:
                var diag_val = view[i, i] - s
                if diag_val <= 0:
                    raise ValueError(
                        function="cholesky()",
                        message=(
                            "Matrix is not positive-definite (non-positive"
                            " diagonal encountered)."
                        ),
                    )
                l_data[i * n + j] = sqrt(diag_val)
            else:
                l_data[i * n + j] = (view[i, j] - s) / l_data[j * n + j]

    return Matrix[Scalar[dtype]](
        buffer=l_data^, nrows=n, ncols=n, row_stride=n, col_stride=1
    )


def qr[
    dtype: DType, origin: Origin, //
](view: MatrixView[Scalar[dtype], origin]) raises -> Tuple[
    Matrix[Scalar[dtype]], Matrix[Scalar[dtype]]
]:
    """Computes the QR decomposition: A = Q R (Householder reflections).

    Works for any m x n matrix view with m >= n.
    """
    var m = view.nrows()
    var n = view.ncols()
    if m < n:
        raise ValueError(
            function="qr()",
            message="QR decomposition requires nrows >= ncols.",
        )

    # Copy A into row-major R workspace (m x n).
    var r_data = List[Scalar[dtype]](unsafe_uninit_length=m * n)
    for i in range(m):
        for j in range(n):
            r_data[i * n + j] = view[i, j]

    # Q starts as identity (m x m).
    var q_data = List[Scalar[dtype]](length=m * m, fill=0)
    for i in range(m):
        q_data[i * m + i] = 1

    for k in range(n):
        # --- Build Householder vector v for column k below row k ---
        # Compute norm of r_data[k:m, k].
        var sigma: Scalar[dtype] = 0
        for i in range(k, m):
            sigma += r_data[i * n + k] * r_data[i * n + k]
        var norm_x = sqrt(sigma)

        if norm_x == 0:
            continue  # Column is already zero; skip.

        # Choose sign to avoid cancellation.
        var x_k = r_data[k * n + k]
        var sign: Scalar[dtype] = 1
        if x_k < 0:
            sign = -1
        var v_k0 = x_k + sign * norm_x

        # Store v in a temporary list (length m - k). v[0] = v_k0, rest = r[i,k].
        var v_len = m - k
        var v = List[Scalar[dtype]](unsafe_uninit_length=v_len)
        v[0] = v_k0
        for i in range(1, v_len):
            v[i] = r_data[(k + i) * n + k]

        # Compute tau = 2 / (v^T v).
        var vtv: Scalar[dtype] = 0
        for i in range(v_len):
            vtv += v[i] * v[i]
        var tau = Scalar[dtype](2) / vtv

        # --- Apply Householder to R: R[k:m, k:n] -= tau * v * (v^T * R[k:m, k:n]) ---
        for j in range(k, n):
            var dot: Scalar[dtype] = 0
            for i in range(v_len):
                dot += v[i] * r_data[(k + i) * n + j]
            for i in range(v_len):
                r_data[(k + i) * n + j] = (
                    r_data[(k + i) * n + j] - tau * v[i] * dot
                )

        # --- Accumulate Q: Q[:, k:m] -= tau * (Q[:, k:m] * v) * v^T ---
        for i in range(m):
            var dot: Scalar[dtype] = 0
            for j2 in range(v_len):
                dot += q_data[i * m + (k + j2)] * v[j2]
            for j2 in range(v_len):
                q_data[i * m + (k + j2)] = (
                    q_data[i * m + (k + j2)] - tau * dot * v[j2]
                )

    var Q = Matrix[Scalar[dtype]](
        buffer=q_data^, nrows=m, ncols=m, row_stride=m, col_stride=1
    )
    var R = Matrix[Scalar[dtype]](
        buffer=r_data^, nrows=m, ncols=n, row_stride=n, col_stride=1
    )
    return (Q^, R^)


def det[
    dtype: DType, origin: Origin, //
](view: MatrixView[Scalar[dtype], origin]) raises -> Scalar[dtype]:
    """Computes the determinant of a square matrix view via LU decomposition."""
    if view.nrows() != view.ncols():
        raise ValueError(
            function="det()",
            message="Matrix must be square to compute determinant.",
        )
    var n = view.nrows()
    var lu_result = lu(view)
    ref U = lu_result[1]
    ref piv = lu_result[2]

    var d: Scalar[dtype] = 1
    for i in range(n):
        d *= U._data[i * n + i]

    var piv_copy = List[Int](unsafe_uninit_length=n)
    for i in range(n):
        piv_copy[i] = piv[i]

    var swaps = 0
    for i in range(n):
        while piv_copy[i] != i:
            var target = piv_copy[i]
            piv_copy[i] = piv_copy[target]
            piv_copy[target] = target
            swaps += 1

    if swaps % 2 == 1:
        d = -d

    return d


def det[
    T: Numeric & Comparable, origin: Origin, //
](view: MatrixView[T, origin]) raises -> T:
    """Computes the determinant of a square matrix view via LU decomposition.

    Parameters:
        T: The type of the matrix elements.
        origin: The origin of the operand.

    Args:
        view: The square matrix or view to reduce.

    Returns:
        The determinant.

    Raises:
        ValueError: If the matrix is not square.
    """
    if view.nrows() != view.ncols():
        raise ValueError(
            function="det()",
            message="Matrix must be square to compute determinant.",
        )
    var n = view.nrows()
    var lu_result = lu(view)
    ref U = lu_result[1]
    ref piv = lu_result[2]

    var d = T.one()
    for i in range(n):
        d = d * U._data[i * n + i]

    var piv_copy = List[Int](unsafe_uninit_length=n)
    for i in range(n):
        piv_copy[i] = piv[i]

    var swaps = 0
    for i in range(n):
        while piv_copy[i] != i:
            var target = piv_copy[i]
            piv_copy[i] = piv_copy[target]
            piv_copy[target] = target
            swaps += 1

    if swaps % 2 == 1:
        d = -d

    return d^


def solve[
    dtype: DType, origin_a: Origin, origin_b: Origin, //
](
    A: MatrixView[Scalar[dtype], origin_a],
    b: MatrixView[Scalar[dtype], origin_b],
) raises -> Matrix[Scalar[dtype]]:
    """Solves the linear system Ax = b for x, using LU decomposition.

    Both A and b can be matrix views. The right-hand side b can be a
    column vector (n x 1) or a matrix (n x k) for multiple right-hand sides.
    """
    if A.nrows() != A.ncols():
        raise ValueError(
            function="solve()",
            message="Coefficient matrix A must be square.",
        )
    var n = A.nrows()
    if b.nrows() != n:
        raise ValueError(
            function="solve()",
            message="Dimensions of A and b do not match: A is "
            + String(n)
            + "x"
            + String(n)
            + " but b has "
            + String(b.nrows())
            + " rows.",
        )

    var k = b.ncols()  # number of right-hand sides

    # LU decompose: PA = LU
    var lu_result = lu(A)
    ref L = lu_result[0]
    ref U = lu_result[1]
    ref piv = lu_result[2]

    # Permute rows of b according to piv: Pb
    var pb_data = List[Scalar[dtype]](unsafe_uninit_length=n * k)
    for i in range(n):
        var src_row = piv[i]
        for j in range(k):
            pb_data[i * k + j] = b[src_row, j]

    # Allocate solution workspace x (n x k), row-major.
    var x_data = List[Scalar[dtype]](length=n * k, fill=0)

    # For each right-hand side column:
    for col in range(k):
        # Forward substitution: Ly = Pb  (L is unit lower-triangular)
        var y = List[Scalar[dtype]](unsafe_uninit_length=n)
        for i in range(n):
            var s: Scalar[dtype] = pb_data[i * k + col]
            for j2 in range(i):
                s -= L._data[i * n + j2] * y[j2]
            y[i] = s

        # Back substitution: Ux = y
        for i in range(n - 1, -1, -1):
            var s: Scalar[dtype] = y[i]
            for j2 in range(i + 1, n):
                s -= U._data[i * n + j2] * x_data[j2 * k + col]
            x_data[i * k + col] = s / U._data[i * n + i]

    return Matrix[Scalar[dtype]](
        buffer=x_data^,
        nrows=n,
        ncols=k,
        row_stride=k,
        col_stride=1,
    )


def solve[
    T: Numeric & Comparable, origin_a: Origin, origin_b: Origin, //
](A: MatrixView[T, origin_a], b: MatrixView[T, origin_b]) raises -> Matrix[T]:
    """Solves the linear system Ax = b for x, using LU decomposition.

    Both A and b can be matrix views. The right-hand side b can be a column
    vector (n x 1) or a matrix (n x k) for multiple right-hand sides.

    Parameters:
        T: The type of the matrix elements.
        origin_a: The origin of the coefficient matrix.
        origin_b: The origin of the right-hand side.

    Args:
        A: The square coefficient matrix or view.
        b: The right-hand side, with as many rows as A.

    Returns:
        A new matrix holding the solution, with the shape of b.

    Raises:
        ValueError: If A is not square, if the shapes do not match, or if A is
            singular.
    """
    if A.nrows() != A.ncols():
        raise ValueError(
            function="solve()",
            message="Coefficient matrix A must be square.",
        )
    var n = A.nrows()
    if b.nrows() != n:
        raise ValueError(
            function="solve()",
            message="Dimensions of A and b do not match: A is "
            + String(n)
            + "x"
            + String(n)
            + " but b has "
            + String(b.nrows())
            + " rows.",
        )

    var k = b.ncols()  # number of right-hand sides
    var zero = T.zero()

    # LU decompose: PA = LU
    var lu_result = lu(A)
    ref L = lu_result[0]
    ref U = lu_result[1]
    ref piv = lu_result[2]

    for i in range(n):
        if U._data[i * n + i] == zero:
            raise ValueError(
                function="solve()",
                message="Coefficient matrix A is singular.",
            )

    # Permute rows of b according to piv: Pb
    var pb_data = List[T](length=n * k, fill=zero)
    for i in range(n):
        var src_row = piv[i]
        for j in range(k):
            pb_data[i * k + j] = b[src_row, j].copy()

    # Allocate solution workspace x (n x k), row-major.
    var x_data = List[T](length=n * k, fill=zero)

    # For each right-hand side column:
    for col in range(k):
        # Forward substitution: Ly = Pb  (L is unit lower-triangular)
        var y = List[T](length=n, fill=zero)
        for i in range(n):
            var s = pb_data[i * k + col].copy()
            for j2 in range(i):
                s = s - L._data[i * n + j2] * y[j2]
            y[i] = s^

        # Back substitution: Ux = y
        for i in range(n - 1, -1, -1):
            var s = y[i].copy()
            for j2 in range(i + 1, n):
                s = s - U._data[i * n + j2] * x_data[j2 * k + col]
            x_data[i * k + col] = s / U._data[i * n + i]

    return Matrix[T](
        buffer=x_data^,
        nrows=n,
        ncols=k,
        row_stride=k,
        col_stride=1,
    )


def inv[
    dtype: DType, origin: Origin, //
](view: MatrixView[Scalar[dtype], origin]) raises -> Matrix[Scalar[dtype]]:
    """Computes the inverse of a square matrix view using LU decomposition.

    Solves A @ X = I for X.
    """
    if view.nrows() != view.ncols():
        raise ValueError(
            function="inv()",
            message="Matrix must be square to compute inverse.",
        )
    var n = view.nrows()

    # Build identity matrix as RHS
    var eye_data = List[Scalar[dtype]](length=n * n, fill=0)
    for i in range(n):
        eye_data[i * n + i] = 1
    var I = Matrix[Scalar[dtype]](
        buffer=eye_data^,
        nrows=n,
        ncols=n,
        row_stride=n,
        col_stride=1,
    )

    return solve(view, I.view())


def inv[
    T: Numeric & Comparable, origin: Origin, //
](view: MatrixView[T, origin]) raises -> Matrix[T]:
    """Computes the inverse of a square matrix view using LU decomposition.

    Solves A @ X = I for X.

    Parameters:
        T: The type of the matrix elements.
        origin: The origin of the operand.

    Args:
        view: The square matrix or view to invert.

    Returns:
        A new matrix holding the inverse.

    Raises:
        ValueError: If the matrix is not square or is singular.
    """
    if view.nrows() != view.ncols():
        raise ValueError(
            function="inv()",
            message="Matrix must be square to compute inverse.",
        )
    var n = view.nrows()

    # Build identity matrix as RHS
    var eye_data = List[T](length=n * n, fill=T.zero())
    var one = T.one()
    for i in range(n):
        eye_data[i * n + i] = one.copy()
    var I = Matrix[T](
        buffer=eye_data^,
        nrows=n,
        ncols=n,
        row_stride=n,
        col_stride=1,
    )

    return solve(view, I.view())


# ===---------------------------------------------------------------------- ===#
# Matrix power
# ===---------------------------------------------------------------------- ===#
# Repeated multiplication by squaring: `A ** 13` costs five products rather
# than twelve. The exponent is an `Int` because only a whole number of
# multiplications is defined; a fractional matrix power needs an
# eigendecomposition and is a different routine.


def matrix_power[
    dtype: DType, origin: Origin, //
](view: MatrixView[Scalar[dtype], origin], exponent: Int) raises -> Matrix[
    Scalar[dtype]
]:
    """Raises a square matrix to an integer power.

    `matrix_power(A, 3)` is `A @ A @ A`, `matrix_power(A, 0)` is the identity,
    and a negative exponent inverts first: `matrix_power(A, -2)` is
    `inv(A) @ inv(A)`. This is what `A ** n` calls.

    Parameters:
        dtype: The dtype behind the element type, deduced rather than written.
        origin: The origin of the input view.

    Args:
        view: The square matrix or view to raise.
        exponent: The power to raise it to. May be negative.

    Returns:
        A new matrix holding the product.

    Raises:
        ValueError: If the matrix is not square, or if a negative exponent is
            asked of a singular matrix.
    """
    if view.nrows() != view.ncols():
        raise ValueError(
            function="matrix_power()",
            message="Matrix must be square to be raised to a power.",
        )
    var n = view.nrows()

    var base: Matrix[Scalar[dtype]]
    var remaining: Int
    if exponent < 0:
        base = inv(view)
        remaining = -exponent
    else:
        base = view.to_matrix()
        remaining = exponent

    var eye_data = List[Scalar[dtype]](length=n * n, fill=0)
    for i in range(n):
        eye_data[i * n + i] = 1
    var result = Matrix[Scalar[dtype]](
        buffer=eye_data^,
        nrows=n,
        ncols=n,
        row_stride=n,
        col_stride=1,
    )

    while remaining > 0:
        if remaining % 2 == 1:
            result = linamo.routines.math.matmul(result.view(), base.view())
        remaining //= 2
        if remaining > 0:
            base = linamo.routines.math.matmul(base.view(), base.view())
    return result^


def matrix_power[
    T: Numeric & Comparable, origin: Origin, //
](view: MatrixView[T, origin], exponent: Int) raises -> Matrix[T]:
    """Raises a square arbitrary-precision matrix to an integer power.

    `matrix_power(A, 3)` is `A @ A @ A`, `matrix_power(A, 0)` is the identity,
    and a negative exponent inverts first: `matrix_power(A, -2)` is
    `inv(A) @ inv(A)`. This is what `A ** n` calls.

    Parameters:
        T: The type of the matrix elements.
        origin: The origin of the input view.

    Args:
        view: The square matrix or view to raise.
        exponent: The power to raise it to. May be negative.

    Returns:
        A new matrix holding the product.

    Raises:
        ValueError: If the matrix is not square, or if a negative exponent is
            asked of a singular matrix.
    """
    if view.nrows() != view.ncols():
        raise ValueError(
            function="matrix_power()",
            message="Matrix must be square to be raised to a power.",
        )
    var n = view.nrows()

    var data = List[T](length=n * n, fill=T.zero())
    for i in range(n):
        data[i * n + i] = T.one()
    var result = Matrix[T](
        buffer=data^,
        nrows=n,
        ncols=n,
        row_stride=n,
        col_stride=1,
    )

    var base: Matrix[T]
    var remaining: Int
    if exponent < 0:
        base = inv(view)
        remaining = -exponent
    else:
        base = view.to_matrix()
        remaining = exponent

    while remaining > 0:
        if remaining % 2 == 1:
            result = linamo.routines.math.matmul(result.view(), base.view())
        remaining //= 2
        if remaining > 0:
            base = linamo.routines.math.matmul(base.view(), base.view())
    return result^


def lstsq[
    dtype: DType, origin_a: Origin, origin_b: Origin, //
](
    A: MatrixView[Scalar[dtype], origin_a],
    b: MatrixView[Scalar[dtype], origin_b],
) raises -> Matrix[Scalar[dtype]]:
    """Solves the least squares problem min ||Ax - b||₂ via QR decomposition.

    Works for overdetermined systems (m >= n). For multiple right-hand
    sides, b should have shape (m x k).
    """
    var m = A.nrows()
    var n = A.ncols()
    if m < n:
        raise ValueError(
            function="lstsq()",
            message="Least squares requires nrows >= ncols (overdetermined).",
        )
    if b.nrows() != m:
        raise ValueError(
            function="lstsq()",
            message="Dimensions of A and b do not match: A has "
            + String(m)
            + " rows but b has "
            + String(b.nrows())
            + " rows.",
        )

    var k = b.ncols()  # number of right-hand sides

    # QR decomposition: A = Q R  (Q: m×m, R: m×n)
    var qr_result = qr(A)
    ref Q = qr_result[0]
    ref R = qr_result[1]

    # Compute Q^T b (m×m transposed @ m×k = m×k).
    # We only need the first n rows of Q^T b.
    var qtb_data = List[Scalar[dtype]](unsafe_uninit_length=n * k)
    for i in range(n):
        for j in range(k):
            var s: Scalar[dtype] = 0
            for p in range(m):
                # Q^T[i, p] = Q[p, i]  (Q is stored row-major, m×m)
                s += Q._data[p * m + i] * b[p, j]
            qtb_data[i * k + j] = s

    # Back substitution: R1 x = (Q^T b)[:n]
    var x_data = List[Scalar[dtype]](length=n * k, fill=0)
    for col in range(k):
        for i in range(n - 1, -1, -1):
            var s: Scalar[dtype] = qtb_data[i * k + col]
            for j2 in range(i + 1, n):
                s -= R._data[i * n + j2] * x_data[j2 * k + col]
            x_data[i * k + col] = s / R._data[i * n + i]

    return Matrix[Scalar[dtype]](
        buffer=x_data^,
        nrows=n,
        ncols=k,
        row_stride=k,
        col_stride=1,
    )
