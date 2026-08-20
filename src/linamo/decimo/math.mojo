"""
Arithmetic on matrices of arbitrary-precision elements.
"""

from decimo import Numeric

from linamo.types.errors import ValueError
from linamo.types.matrix import Matrix
from linamo.types.matrix_view import MatrixView


# ===----------------------------------------------------------------------===#
# Element-wise binary operations
# ===----------------------------------------------------------------------===#
# These take `MatrixView` operands and nothing else, exactly as the scalar
# routines do, so a `Matrix` argument converts through the `@implicit`
# constructor and `add(a, b)` compiles whichever of the two types each operand
# happens to be.
#
# There is no SIMD path and no `parallelize` here. A `BigInt` addition
# allocates, so the loop is memory-bound rather than issue-bound, and the
# element-by-element walk below is what the operation costs.


def _elementwise[
    T: Numeric,
    func: def(T, T) raises thin -> T,
    origin_a: Origin,
    origin_b: Origin,
](a: MatrixView[T, origin_a], b: MatrixView[T, origin_b]) raises -> Matrix[T]:
    """Applies `func` element-wise to two views of the same shape.

    Parameters:
        T: The type of the matrix elements.
        func: The binary operation to apply.
        origin_a: The origin of the first view.
        origin_b: The origin of the second view.

    Args:
        a: The first input.
        b: The second input.

    Raises:
        ValueError: If the shapes do not match.

    Returns:
        A new C-contiguous matrix holding the results.
    """
    if a.nrows() != b.nrows() or a.ncols() != b.ncols():
        raise ValueError(
            function="_elementwise()",
            message="Input matrices must have the same shape.",
        )
    var buffer = List[T](capacity=a.nrows() * a.ncols())
    for i in range(a.nrows()):
        for j in range(a.ncols()):
            buffer.append(func(a[i, j], b[i, j]))
    return Matrix[T](buffer^, a.nrows(), a.ncols(), a.ncols(), 1)


def add[
    T: Numeric, origin_a: Origin, origin_b: Origin
](a: MatrixView[T, origin_a], b: MatrixView[T, origin_b]) raises -> Matrix[T]:
    """Performs element-wise addition.

    Parameters:
        T: The type of the matrix elements.
        origin_a: The origin of the first view.
        origin_b: The origin of the second view.

    Args:
        a: The first input.
        b: The second input.

    Raises:
        ValueError: If the shapes do not match.

    Returns:
        A new matrix holding the element-wise sum.
    """
    return _elementwise[func=T.__add__](a, b)


def sub[
    T: Numeric, origin_a: Origin, origin_b: Origin
](a: MatrixView[T, origin_a], b: MatrixView[T, origin_b]) raises -> Matrix[T]:
    """Performs element-wise subtraction.

    Parameters:
        T: The type of the matrix elements.
        origin_a: The origin of the first view.
        origin_b: The origin of the second view.

    Args:
        a: The first input.
        b: The second input.

    Raises:
        ValueError: If the shapes do not match.

    Returns:
        A new matrix holding the element-wise difference.
    """
    return _elementwise[func=T.__sub__](a, b)


def mul[
    T: Numeric, origin_a: Origin, origin_b: Origin
](a: MatrixView[T, origin_a], b: MatrixView[T, origin_b]) raises -> Matrix[T]:
    """Performs element-wise multiplication.

    This is the Hadamard product, not matrix multiplication; see `matmul`.

    Parameters:
        T: The type of the matrix elements.
        origin_a: The origin of the first view.
        origin_b: The origin of the second view.

    Args:
        a: The first input.
        b: The second input.

    Raises:
        ValueError: If the shapes do not match.

    Returns:
        A new matrix holding the element-wise product.
    """
    return _elementwise[func=T.__mul__](a, b)


# ===----------------------------------------------------------------------===#
# Unary and scalar operations
# ===----------------------------------------------------------------------===#


def neg[
    T: Numeric, origin: Origin
](a: MatrixView[T, origin]) raises -> Matrix[T]:
    """Negates every element.

    Parameters:
        T: The type of the matrix elements.
        origin: The origin of the input view.

    Args:
        a: The input.

    Returns:
        A new matrix holding the negated elements.
    """
    var buffer = List[T](capacity=a.nrows() * a.ncols())
    for i in range(a.nrows()):
        for j in range(a.ncols()):
            buffer.append(-a[i, j])
    return Matrix[T](buffer^, a.nrows(), a.ncols(), a.ncols(), 1)


def scalar_add[
    T: Numeric, origin: Origin
](a: MatrixView[T, origin], value: T) raises -> Matrix[T]:
    """Adds one value to every element.

    Parameters:
        T: The type of the matrix elements.
        origin: The origin of the input view.

    Args:
        a: The input.
        value: The value added to every element.

    Returns:
        A new matrix holding the results.
    """
    var buffer = List[T](capacity=a.nrows() * a.ncols())
    for i in range(a.nrows()):
        for j in range(a.ncols()):
            buffer.append(a[i, j] + value)
    return Matrix[T](buffer^, a.nrows(), a.ncols(), a.ncols(), 1)


def scalar_mul[
    T: Numeric, origin: Origin
](a: MatrixView[T, origin], value: T) raises -> Matrix[T]:
    """Multiplies every element by one value.

    Parameters:
        T: The type of the matrix elements.
        origin: The origin of the input view.

    Args:
        a: The input.
        value: The value every element is multiplied by.

    Returns:
        A new matrix holding the results.
    """
    var buffer = List[T](capacity=a.nrows() * a.ncols())
    for i in range(a.nrows()):
        for j in range(a.ncols()):
            buffer.append(a[i, j] * value)
    return Matrix[T](buffer^, a.nrows(), a.ncols(), a.ncols(), 1)


# ===----------------------------------------------------------------------===#
# Matrix multiplication and reductions
# ===----------------------------------------------------------------------===#


def matmul[
    T: Numeric, origin_a: Origin, origin_b: Origin
](a: MatrixView[T, origin_a], b: MatrixView[T, origin_b]) raises -> Matrix[T]:
    """Performs matrix multiplication.

    A plain triple loop. The scalar kernel's four layout-specialised SIMD paths
    exist to keep a vector unit fed; here every `+` and `*` allocates, so the
    layout of the operands is not what the time is spent on, and one readable
    loop is the honest implementation.

    Parameters:
        T: The type of the matrix elements.
        origin_a: The origin of the first view.
        origin_b: The origin of the second view.

    Args:
        a: The left operand.
        b: The right operand.

    Raises:
        ValueError: If the inner dimensions do not match.

    Returns:
        A new C-contiguous `a.nrows() x b.ncols()` matrix.
    """
    if a.ncols() != b.nrows():
        raise ValueError(
            function="matmul()",
            message=(
                "Inner dimensions of a and b must match for matrix"
                " multiplication."
            ),
        )
    var M = a.nrows()
    var N = b.ncols()
    var K = a.ncols()
    var buffer = List[T](capacity=M * N)
    for i in range(M):
        for j in range(N):
            var acc = T.zero()
            for k in range(K):
                acc = acc + a[i, k] * b[k, j]
            buffer.append(acc^)
    return Matrix[T](buffer^, M, N, N, 1)


def total[T: Numeric, origin: Origin](a: MatrixView[T, origin]) raises -> T:
    """Returns the sum of every element.

    Named `total` rather than `sum` because `linamo.sum` reduces along an axis
    and returns a matrix; this is the whole-matrix scalar.

    Parameters:
        T: The type of the matrix elements.
        origin: The origin of the input view.

    Args:
        a: The input.

    Returns:
        The sum of every element, or zero for an empty input.
    """
    var acc = T.zero()
    for i in range(a.nrows()):
        for j in range(a.ncols()):
            acc = acc + a[i, j]
    return acc^


def trace[T: Numeric, origin: Origin](a: MatrixView[T, origin]) raises -> T:
    """Returns the sum of the diagonal elements.

    Parameters:
        T: The type of the matrix elements.
        origin: The origin of the input view.

    Args:
        a: The input, which must be square.

    Raises:
        ValueError: If the input is not square.

    Returns:
        The sum of the diagonal elements.
    """
    if a.nrows() != a.ncols():
        raise ValueError(
            function="trace()",
            message="Trace is defined for square matrices only.",
        )
    var acc = T.zero()
    for i in range(a.nrows()):
        acc = acc + a[i, i]
    return acc^
