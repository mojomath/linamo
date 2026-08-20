"""
Defines mathematical routines for matrices.
"""

from std.algorithm import vectorize
from max.algorithm import parallelize
from std.sys import (
    CompilationTarget,
    num_physical_cores,
    simd_width_of,
)

from decimo import Numeric

from linamo.types.errors import ValueError
from linamo.types.static_matrix import StaticMatrix
from linamo.types.matrix import Matrix
from linamo.types.matrix_view import MatrixView
from linamo.routines.functional import apply_along_axis, fold
from linamo.utils.indexing import get_offset

# [Mojo Miji]
# The routines below take `MatrixView` operands and nothing else: one
# signature per operation, not one per combination of `Matrix` and
# `MatrixView`. A `Matrix` argument converts implicitly (see the `@implicit`
# constructor in `types/matrix_view.mojo`), so `add(a, b)` compiles whichever
# of the two types each operand happens to be, and the conversion is an O(1)
# metadata copy.
#
# A trait bound cannot do this job. `def add[M: MatrixLike](a: M, b: M)` fails
# for a reason deeper than the missing parameterised traits: converting `M` to
# a view has to yield a type whose `origin` parameter depends on the *borrow of
# the argument*, and no trait method can name that. An implicit constructor
# can, because `out self` may be written in terms of the argument. Operand-type
# genericity therefore goes through conversion, not through a trait.

# ===---------------------------------------------------------------------- ===#
# StaticMatrix element-wise operations
# ===---------------------------------------------------------------------- ===#
# `StaticMatrix` keeps its own overloads: it holds a single SIMD register
# rather than a strided buffer, so it shares no kernel with the code below.


def add[
    T: Copyable & Deinitable, nrows: Int, ncols: Int
](
    a: StaticMatrix[T, nrows, ncols], b: StaticMatrix[T, nrows, ncols]
) -> StaticMatrix[T, nrows, ncols]:
    """Performs element-wise addition of two matrices.

    Args:
        a: The first input matrix.
        b: The second input matrix.

    Returns:
        A new matrix containing the element-wise sum of a and b.
    """
    return StaticMatrix[T, nrows, ncols](a._data + b._data)


def sub[
    T: Copyable & Deinitable, nrows: Int, ncols: Int
](
    a: StaticMatrix[T, nrows, ncols], b: StaticMatrix[T, nrows, ncols]
) -> StaticMatrix[T, nrows, ncols]:
    """Performs element-wise subtraction of two matrices.

    Args:
        a: The first input matrix.
        b: The second input matrix.

    Returns:
        A new matrix containing the element-wise difference of a and b.
    """
    return StaticMatrix[T, nrows, ncols](a._data - b._data)


def mul[
    T: Copyable & Deinitable, nrows: Int, ncols: Int
](
    a: StaticMatrix[T, nrows, ncols], b: StaticMatrix[T, nrows, ncols]
) -> StaticMatrix[T, nrows, ncols]:
    """Performs element-wise multiplication of two matrices.

    Args:
        a: The first input matrix.
        b: The second input matrix.

    Returns:
        A new matrix containing the element-wise product of a and b.
    """
    return StaticMatrix[T, nrows, ncols](a._data * b._data)


def div[
    T: Copyable & Deinitable, nrows: Int, ncols: Int
](
    a: StaticMatrix[T, nrows, ncols], b: StaticMatrix[T, nrows, ncols]
) -> StaticMatrix[T, nrows, ncols]:
    """Performs element-wise division of two matrices.

    Args:
        a: The first input matrix.
        b: The second input matrix.

    Returns:
        A new matrix containing the element-wise quotient of a and b.
    """
    return StaticMatrix[T, nrows, ncols](a._data / b._data)


def matmul[
    T: Copyable & Deinitable, nrows: Int, ncols: Int, inner_dim: Int
](
    a: StaticMatrix[T, nrows, inner_dim],
    b: StaticMatrix[T, inner_dim, ncols],
) -> StaticMatrix[T, nrows, ncols]:
    """Performs matrix multiplication of two matrices.

    Args:
        a: The first input matrix.
        b: The second input matrix.

    Returns:
        A new matrix containing the product of a and b.
    """
    comptime d = StaticMatrix[T, nrows, ncols].dtype
    var result = StaticMatrix[T, nrows, ncols]()
    for i in range(nrows):
        for j in range(ncols):
            var sum = Scalar[d](0)
            for k in range(inner_dim):
                sum += rebind[Scalar[d]](a[i, k]) * rebind[Scalar[d]](b[k, j])
            result._set_flat(i * result.BUFFER_COL_LEN + j, rebind[T](sum))
    return result^


# --------------------------------------------------------------------------- #
# Core view-based matmul implementation
# --------------------------------------------------------------------------- #
# The canonical SIMD-optimised matmul operates on MatrixView.
# Converting Matrix → MatrixView via `.view()` is free (metadata copy only;
# the underlying data lives in a Span that borrows from the Matrix's List).
# This avoids duplicating the implementation for every Matrix/View combination.


# --------------------------------------------------------------------------- #
# Matmul tuning constants
# --------------------------------------------------------------------------- #
# `vectorize[w]` hands the kernel `w` elements per call. Once `w` is wider than
# a physical vector register the compiler lowers the wide `SIMD` value into
# several registers, so `w` is an unroll factor counted in registers, not a
# hardware width. The optimum is therefore a fixed register count rather than
# a fixed element count, which is why it cannot be one literal shared by every
# dtype: on a 128-bit vector unit throughput peaks at 32 elements for float64,
# 64 for float32 and 128 for float16. `_MATMUL_UNROLL_REGISTERS` registers
# in all three cases. Below that count the pipeline is underfed; above it the
# kernel spills to the stack, which costs more than the unrolling saves.
comptime _VECTOR_REGISTERS = 32 if (
    CompilationTarget.has_neon() or CompilationTarget.has_avx512f()
) else 16
"""Architectural vector registers: 32 under NEON and AVX-512, 16 under SSE/AVX2.
"""

comptime _MATMUL_UNROLL_REGISTERS = _VECTOR_REGISTERS // 2
"""Registers the unrolled body may fill.

Each FMA keeps two vectors live, the accumulator and the `B` operand, so the
unroll saturates the register file at half its size.
"""

# Rows (or columns) a worker must own before spawning it pays for itself.
comptime _MATMUL_ITEMS_PER_WORKER = 24


def _matmul_workers(items: Int) -> Int:
    """Worker count for splitting `items` rows (or columns) across cores.

    Returns 0 or 1 when the work is too small to be worth dispatching, which
    the caller reads as "run this loop serially".

    Args:
        items: The number of rows or columns the parallel loop ranges over.

    Returns:
        The number of workers to hand to `parallelize`.
    """
    # One task per row asks the scheduler to dispatch work items far smaller
    # than the dispatch itself; chunking amortises that. The cap sits below the
    # core count because a barrier-synchronised loop finishes no sooner than
    # its slowest worker, and the slowest core on an asymmetric machine is much
    # slower than the rest.
    var cap = num_physical_cores() * 3 // 4
    if cap < 1:
        cap = 1
    var workers = items // _MATMUL_ITEMS_PER_WORKER
    if workers > cap:
        workers = cap
    return workers


def _matmul_view_simd[
    dtype: DType,
    origin_a: Origin,
    origin_b: Origin,
](
    a: MatrixView[Scalar[dtype], origin_a],
    b: MatrixView[Scalar[dtype], origin_b],
) raises -> Matrix[Scalar[dtype]]:
    """Core matrix multiplication on MatrixView operands.

    Dispatches to one of four SIMD-optimised paths based on the memory layout
    of the two operands, falling back to a general stride-aware loop when
    neither operand has contiguous rows or columns.

    **Path 1 - B row-contiguous** (covers R×R, F×R, any×R):
        Vectorize across B's row (SIMD load), parallelize over A's rows.
        A is accessed element-wise with the generic offset formula.

    **Path 2 - C×F** (A row-contiguous, B column-contiguous):
        Each result element is a dot-product of a contiguous A-row and a
        contiguous B-column.  SIMD vectorize the K loop with `reduce_add`.

    **Path 3 - A column-contiguous** (covers F×F, F×weird):
        Vectorize across A's column (SIMD load), accumulate in a temporary
        contiguous column buffer, then scatter to the C-contiguous result.
        Parallelize over B's columns.

    **Path 4 - General fallback**:
        Stride-aware scalar triple loop, parallelize over rows.

    The result is always a freshly allocated, C-contiguous Matrix.
    """
    comptime simd_w = _MATMUL_UNROLL_REGISTERS * simd_width_of[dtype]()

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

    # Result is always C-contiguous (row-major).
    var result = Matrix[Scalar[dtype]](M, N, N, 1)

    # Shared pointer setup – used by all SIMD paths.
    var a_ptr = a._data.unsafe_ptr()
    var b_ptr = b._data.unsafe_ptr()
    var a_off = a.offset()
    var b_off = b.offset()
    var a_rs = a.row_stride()
    var a_cs = a.col_stride()
    var b_rs = b.row_stride()
    var b_cs = b.col_stride()

    # ---------------------------------------------------------------------- #
    # Path 1: B row-contiguous (col_stride == 1)
    # Covers R×R, F×R, and any layout where B's rows are contiguous.
    # Vectorize over B's columns + parallelize over A's rows.
    # ---------------------------------------------------------------------- #
    if b.is_row_contiguous():

        @parameter
        def process_row_broadcast(i: Int):
            for k in range(K):
                # [Mojo Miji]
                # Broadcast A[i,k] (scalar) and SIMD-multiply with row k of B,
                # accumulating into row i of result.
                def vec_col_broadcast[
                    w: Int
                ](j: Int) {
                    mut result,
                    imm a_ptr,
                    imm b_ptr,
                    imm a_off,
                    imm b_off,
                    imm a_rs,
                    imm a_cs,
                    imm b_rs,
                    imm N,
                    imm i,
                    imm k,
                }:
                    var r_idx = i * N + j
                    result._data._data.unsafe_store[width=w](
                        r_idx,
                        result._data._data.unsafe_load[width=w](r_idx)
                        + a_ptr.unsafe_load[width=1](
                            a_off + i * a_rs + k * a_cs
                        )
                        * b_ptr.unsafe_load[width=w](b_off + k * b_rs + j),
                    )

                vectorize[simd_w](N, vec_col_broadcast)

        var workers = _matmul_workers(M)
        if workers > 1:
            parallelize[process_row_broadcast](M, workers)
        else:
            for i in range(M):
                process_row_broadcast(i)

    # ---------------------------------------------------------------------- #
    # Path 2: A row-contiguous, B column-contiguous  (C × F)
    # Both A's row and B's column are contiguous over the K dimension,
    # so each result element is a SIMD dot-product with reduce_add.
    # ---------------------------------------------------------------------- #
    elif a.is_row_contiguous() and b.is_col_contiguous():

        @parameter
        def process_row_dot(i: Int):
            for j in range(N):
                var dot_sum: Scalar[dtype] = 0

                def vec_k_dot[
                    w: Int
                ](k: Int) {
                    mut dot_sum,
                    imm a_ptr,
                    imm b_ptr,
                    imm a_off,
                    imm b_off,
                    imm a_rs,
                    imm b_cs,
                    imm i,
                    imm j,
                }:
                    dot_sum += (
                        a_ptr.unsafe_load[width=w](a_off + i * a_rs + k)
                        * b_ptr.unsafe_load[width=w](b_off + j * b_cs + k)
                    ).reduce_add()

                vectorize[simd_w](K, vec_k_dot)
                result._data._data.unsafe_store[width=1](i * N + j, dot_sum)

        var workers = _matmul_workers(M)
        if workers > 1:
            parallelize[process_row_dot](M, workers)
        else:
            for i in range(M):
                process_row_dot(i)

    # ---------------------------------------------------------------------- #
    # Path 3: A column-contiguous  (covers F × F, F × weird)
    # A's columns are contiguous → vectorize over rows of A.
    # Accumulate in a temporary contiguous column buffer, then scatter to
    # the C-contiguous result.  Parallelize over B's columns.
    # ---------------------------------------------------------------------- #
    elif a.is_col_contiguous():

        @parameter
        def process_col_accumulate(j: Int):
            # Temporary column buffer for SIMD accumulation.
            var temp = List[Scalar[dtype]](length=M, fill=0)
            var temp_ptr = temp._data

            for k in range(K):
                var b_kj = b_ptr.unsafe_load[width=1](
                    b_off + k * b_rs + j * b_cs
                )

                def vec_row_accumulate[
                    w: Int
                ](i: Int) {
                    imm temp_ptr,
                    imm a_ptr,
                    imm a_off,
                    imm a_cs,
                    imm b_kj,
                    imm k,
                }:
                    temp_ptr.unsafe_store[width=w](
                        i,
                        temp_ptr.unsafe_load[width=w](i)
                        + a_ptr.unsafe_load[width=w](a_off + k * a_cs + i)
                        * b_kj,
                    )

                vectorize[simd_w](M, vec_row_accumulate)

            # Scatter temp column into result's column j.
            for i in range(M):
                result._data._data.unsafe_store[width=1](
                    i * N + j, temp_ptr.unsafe_load[width=1](i)
                )

        var workers = _matmul_workers(N)
        if workers > 1:
            parallelize[process_col_accumulate](N, workers)
        else:
            for j in range(N):
                process_col_accumulate(j)

    # ---------------------------------------------------------------------- #
    # Path 4: General fallback – any memory layout
    # ---------------------------------------------------------------------- #
    else:

        @parameter
        def process_row_general(i: Int):
            for j in range(N):
                var sum: Scalar[dtype] = 0
                for k in range(K):
                    sum += (
                        a._data[a_off + i * a_rs + k * a_cs]
                        * b._data[b_off + k * b_rs + j * b_cs]
                    )
                result._data._data.unsafe_store[width=1](i * N + j, sum)

        var workers = _matmul_workers(M)
        if workers > 1:
            parallelize[process_row_general](M, workers)
        else:
            for i in range(M):
                process_row_general(i)

    return result^


# --------------------------------------------------------------------------- #
# Public matmul entry point
# --------------------------------------------------------------------------- #


def matmul[
    dtype: DType,
    origin_a: Origin,
    origin_b: Origin,
](
    a: MatrixView[Scalar[dtype], origin_a],
    b: MatrixView[Scalar[dtype], origin_b],
) raises -> Matrix[Scalar[dtype]]:
    """Performs matrix multiplication.

    Either operand may be a `Matrix` or a `MatrixView`; a `Matrix` is converted
    to a read-only view implicitly.

    Args:
        a: The first operand.
        b: The second operand.

    Returns:
        A new C-contiguous matrix containing the product of a and b.
    """
    return _matmul_view_simd(a, b)


# ===---------------------------------------------------------------------- ===#
# Element-wise operation primitives
# ===---------------------------------------------------------------------- ===#
# [Mojo Miji]
# We define a single, generic `_elementwise_view` function that takes a
# `func: def(Scalar, Scalar) thin -> Scalar` compile-time parameter. This avoids
# duplicating near-identical code for add, sub, mul, div. Each public function
# is a thin wrapper that plugs in the right SIMD dunder method directly
# (e.g. `Scalar[dtype].__add__`).
#
# For C-contiguous operands, a SIMD-vectorised fast path is used (SIMD
# loads/stores with per-lane func application).  For non-contiguous views the
# fallback is a stride-aware double loop.
#
# Every public binary operation takes two `MatrixView` operands and nothing
# else. A `Matrix` argument still works, because `MatrixView` has an
# `@implicit` constructor from `Matrix` (see `types/matrix_view.mojo`), so the
# compiler inserts the conversion at the call site. That is what lets one
# signature stand in for the four combinations of Matrix / MatrixView that used
# to be written out by hand.
#
# The conversion always produces a *read-only* view, which is what makes
# `add(a, a)` legal: two immutable borrows of one matrix are fine, two mutable
# ones are not.
#
# The same pattern is applied to scalar–matrix operations via
# `_scalar_elementwise_view`.

# --------------------------------------------------------------------------- #
# Core view-based element-wise implementation
# --------------------------------------------------------------------------- #


def _elementwise_view[
    dtype: DType,
    func: def(Scalar[dtype], Scalar[dtype]) thin -> Scalar[dtype],
    origin_a: Origin,
    origin_b: Origin,
](
    a: MatrixView[Scalar[dtype], origin_a],
    b: MatrixView[Scalar[dtype], origin_b],
) raises -> Matrix[Scalar[dtype]]:
    """Core element-wise binary operation on two MatrixView operands.

    When both operands are C-contiguous, a SIMD-vectorised fast path is
    taken (linear memory traversal with vectorize).  Otherwise, a stride-aware
    double loop is used.

    The result is always a freshly allocated, C-contiguous Matrix.
    """
    if a.nrows() != b.nrows() or a.ncols() != b.ncols():
        raise ValueError(
            function="_elementwise_view()",
            message="Input matrices must have the same shape.",
        )
    var M = a.nrows()
    var N = a.ncols()
    var total = M * N
    var result = Matrix[Scalar[dtype]](M, N, N, 1)

    if a.is_c_contiguous() and b.is_c_contiguous():
        # Fast SIMD path: both operands are dense row-major, so we can do a
        # single linear pass with vectorized loads/stores.
        comptime simd_w = simd_width_of[dtype]()
        var a_ptr = a._data.unsafe_ptr()
        var b_ptr = b._data.unsafe_ptr()
        var a_off = a.offset()
        var b_off = b.offset()

        def vec_op[
            w: Int
        ](idx: Int) {mut result, imm a_ptr, imm b_ptr, imm a_off, imm b_off,}:
            var a_chunk = a_ptr.unsafe_load[width=w](a_off + idx)
            var b_chunk = b_ptr.unsafe_load[width=w](b_off + idx)
            var res = SIMD[dtype, w](0)

            comptime for lane in range(w):
                res[lane] = func(a_chunk[lane], b_chunk[lane])
            result._data._data.unsafe_store[width=w](idx, res)

        vectorize[simd_w](total, vec_op)
    else:
        # General stride-aware fallback.
        for i in range(M):
            for j in range(N):
                result._data[i * N + j] = func(a[i, j], b[i, j])

    return result^


# --------------------------------------------------------------------------- #
# Core view-based scalar element-wise implementation
# --------------------------------------------------------------------------- #


def _scalar_elementwise_view[
    dtype: DType,
    func: def(Scalar[dtype], Scalar[dtype]) thin -> Scalar[dtype],
    origin: Origin,
](
    mat: MatrixView[Scalar[dtype], origin],
    scalar: Scalar[dtype],
) -> Matrix[
    Scalar[dtype]
]:
    """Core scalar–matrix element-wise operation on a MatrixView operand.

    When the operand is C-contiguous, a SIMD-vectorised fast path is taken.
    Otherwise, a stride-aware double loop is used.

    The result is always a freshly allocated, C-contiguous Matrix.
    """
    var M = mat.nrows()
    var N = mat.ncols()
    var total = M * N
    var result = Matrix[Scalar[dtype]](M, N, N, 1)

    if mat.is_c_contiguous():
        comptime simd_w = simd_width_of[dtype]()
        var m_ptr = mat._data.unsafe_ptr()
        var m_off = mat.offset()

        def vec_scalar[
            w: Int
        ](idx: Int) {mut result, imm m_ptr, imm m_off, imm scalar,}:
            var m_chunk = m_ptr.unsafe_load[width=w](m_off + idx)
            var s_chunk = SIMD[dtype, w](scalar)
            var res = SIMD[dtype, w](0)

            comptime for lane in range(w):
                res[lane] = func(m_chunk[lane], s_chunk[lane])
            result._data._data.unsafe_store[width=w](idx, res)

        vectorize[simd_w](total, vec_scalar)
    else:
        for i in range(M):
            for j in range(N):
                result._data[i * N + j] = func(mat[i, j], scalar)

    return result^


# ===---------------------------------------------------------------------- ===#
# Dynamic element-wise operations: add, sub, mul, div
# ===---------------------------------------------------------------------- ===#
# One signature each. `Matrix` operands convert implicitly, so all four
# combinations of Matrix / MatrixView call straight through to
# `_elementwise_view` with the appropriate SIMD dunder as the kernel.


# --------------------------------------------------------------------------- #
# add
# --------------------------------------------------------------------------- #


def add[
    dtype: DType, origin_a: Origin, origin_b: Origin
](
    a: MatrixView[Scalar[dtype], origin_a],
    b: MatrixView[Scalar[dtype], origin_b],
) raises -> Matrix[Scalar[dtype]]:
    """Element-wise addition of two matrices or views."""
    return _elementwise_view[func=Scalar[dtype].__add__](a, b)


# --------------------------------------------------------------------------- #
# sub
# --------------------------------------------------------------------------- #


def sub[
    dtype: DType, origin_a: Origin, origin_b: Origin
](
    a: MatrixView[Scalar[dtype], origin_a],
    b: MatrixView[Scalar[dtype], origin_b],
) raises -> Matrix[Scalar[dtype]]:
    """Element-wise subtraction of two matrices or views."""
    return _elementwise_view[func=Scalar[dtype].__sub__](a, b)


# --------------------------------------------------------------------------- #
# mul
# --------------------------------------------------------------------------- #


def mul[
    dtype: DType, origin_a: Origin, origin_b: Origin
](
    a: MatrixView[Scalar[dtype], origin_a],
    b: MatrixView[Scalar[dtype], origin_b],
) raises -> Matrix[Scalar[dtype]]:
    """Element-wise multiplication of two matrices or views."""
    return _elementwise_view[func=Scalar[dtype].__mul__](a, b)


# --------------------------------------------------------------------------- #
# div
# --------------------------------------------------------------------------- #


def div[
    dtype: DType, origin_a: Origin, origin_b: Origin
](
    a: MatrixView[Scalar[dtype], origin_a],
    b: MatrixView[Scalar[dtype], origin_b],
) raises -> Matrix[Scalar[dtype]]:
    """Element-wise division of two matrices or views."""
    return _elementwise_view[func=Scalar[dtype].__truediv__](a, b)


# ===---------------------------------------------------------------------- ===#
# Scalar–Matrix operations
# ===---------------------------------------------------------------------- ===#
# One signature each; a `Matrix` operand converts implicitly.


def scalar_add[
    dtype: DType, origin: Origin
](mat: MatrixView[Scalar[dtype], origin], scalar: Scalar[dtype]) -> Matrix[
    Scalar[dtype]
]:
    """Adds a scalar to every element of a matrix or view."""
    return _scalar_elementwise_view[func=Scalar[dtype].__add__](mat, scalar)


def scalar_sub[
    dtype: DType, origin: Origin
](mat: MatrixView[Scalar[dtype], origin], scalar: Scalar[dtype]) -> Matrix[
    Scalar[dtype]
]:
    """Subtracts a scalar from every element of a matrix or view."""
    return _scalar_elementwise_view[func=Scalar[dtype].__sub__](mat, scalar)


def scalar_mul[
    dtype: DType, origin: Origin
](mat: MatrixView[Scalar[dtype], origin], scalar: Scalar[dtype]) -> Matrix[
    Scalar[dtype]
]:
    """Multiplies every element of a matrix view by a scalar."""
    return _scalar_elementwise_view[func=Scalar[dtype].__mul__](mat, scalar)


def scalar_div[
    dtype: DType, origin: Origin
](mat: MatrixView[Scalar[dtype], origin], scalar: Scalar[dtype]) -> Matrix[
    Scalar[dtype]
]:
    """Divides every element of a matrix view by a scalar."""
    return _scalar_elementwise_view[func=Scalar[dtype].__truediv__](mat, scalar)


# ===---------------------------------------------------------------------- ===#
# Core in-place element-wise implementations
# ===---------------------------------------------------------------------- ===#
# The out-of-place cores above always allocate a fresh C-contiguous result.
# The in-place operators (`+=`, `-=`, ...) must not: they write back through
# the left operand's own strides, so a matrix that is a transpose or a
# non-contiguous buffer stays exactly where it is.
#
# The target is a `Matrix` rather than a `MatrixView`. A view is generic over
# `origin`, and Mojo checks a body against every instantiation including the
# read-only one, so nothing writing through `self._data` can be defined on it —
# the same constraint that pushed bulk writes into `routines/mutation.mojo`.


def _elementwise_inplace[
    dtype: DType,
    func: def(Scalar[dtype], Scalar[dtype]) thin -> Scalar[dtype],
    origin_b: Origin,
](mut a: Matrix[Scalar[dtype]], b: MatrixView[Scalar[dtype], origin_b]) raises:
    """Core in-place element-wise binary operation, writing into `a`.

    When both operands are C-contiguous, a SIMD-vectorised fast path is taken.
    Otherwise a stride-aware double loop writes through `a`'s own strides.
    """
    if a.nrows() != b.nrows() or a.ncols() != b.ncols():
        raise ValueError(
            function="_elementwise_inplace()",
            message="Input matrices must have the same shape.",
        )
    var M = a.nrows()
    var N = a.ncols()

    if a.is_c_contiguous() and b.is_c_contiguous():
        comptime simd_w = simd_width_of[dtype]()
        var a_ptr = a._data._data
        var b_ptr = b._data.unsafe_ptr()
        var b_off = b.offset()

        def vec_op[
            w: Int
        ](idx: Int) {imm a_ptr, imm b_ptr, imm b_off,}:
            var a_chunk = a_ptr.unsafe_load[width=w](idx)
            var b_chunk = b_ptr.unsafe_load[width=w](b_off + idx)
            var res = SIMD[dtype, w](0)

            comptime for lane in range(w):
                res[lane] = func(a_chunk[lane], b_chunk[lane])
            a_ptr.unsafe_store[width=w](idx, res)

        vectorize[simd_w](M * N, vec_op)
    else:
        for i in range(M):
            for j in range(N):
                var idx = get_offset(i, j, a.row_stride(), a.col_stride())
                a._data[idx] = func(a._data[idx], b[i, j])


def _scalar_elementwise_inplace[
    dtype: DType,
    func: def(Scalar[dtype], Scalar[dtype]) thin -> Scalar[dtype],
](mut mat: Matrix[Scalar[dtype]], scalar: Scalar[dtype]):
    """Core in-place scalar element-wise operation, writing into `mat`."""
    var M = mat.nrows()
    var N = mat.ncols()

    if mat.is_c_contiguous():
        comptime simd_w = simd_width_of[dtype]()
        var m_ptr = mat._data._data

        def vec_scalar[
            w: Int
        ](idx: Int) {imm m_ptr, imm scalar,}:
            var m_chunk = m_ptr.unsafe_load[width=w](idx)
            var s_chunk = SIMD[dtype, w](scalar)
            var res = SIMD[dtype, w](0)

            comptime for lane in range(w):
                res[lane] = func(m_chunk[lane], s_chunk[lane])
            m_ptr.unsafe_store[width=w](idx, res)

        vectorize[simd_w](M * N, vec_scalar)
    else:
        for i in range(M):
            for j in range(N):
                var idx = get_offset(i, j, mat.row_stride(), mat.col_stride())
                mat._data[idx] = func(mat._data[idx], scalar)


# ===---------------------------------------------------------------------- ===#
# Reflected scalar operand order
# ===---------------------------------------------------------------------- ===#
# `_scalar_elementwise_view` always calls `func(element, scalar)`. For the
# non-commutative reflected operators (`2.0 - A`, `2.0 / A`) the operands have
# to arrive the other way round, so the order is flipped in the kernel rather
# than by threading a `reverse` flag through the core.


def _rsub_op[dtype: DType](a: Scalar[dtype], b: Scalar[dtype]) -> Scalar[dtype]:
    """Returns `b - a`, the operand order `__rsub__` needs."""
    return b - a


def _rdiv_op[dtype: DType](a: Scalar[dtype], b: Scalar[dtype]) -> Scalar[dtype]:
    """Returns `b / a`, the operand order `__rtruediv__` needs."""
    return b / a


# ===---------------------------------------------------------------------- ===#
# Element-wise floordiv, mod, pow
# ===---------------------------------------------------------------------- ===#
# Same 4-overload shape as add/sub/mul/div above: view×view, mat×mat,
# mat×view, view×mat, all delegating to `_elementwise_view`.


# --------------------------------------------------------------------------- #
# floordiv
# --------------------------------------------------------------------------- #


def floordiv[
    dtype: DType, origin_a: Origin, origin_b: Origin
](
    a: MatrixView[Scalar[dtype], origin_a],
    b: MatrixView[Scalar[dtype], origin_b],
) raises -> Matrix[Scalar[dtype]]:
    """Element-wise floor division of two matrices or views."""
    return _elementwise_view[func=Scalar[dtype].__floordiv__](a, b)


# --------------------------------------------------------------------------- #
# mod
# --------------------------------------------------------------------------- #


def mod[
    dtype: DType, origin_a: Origin, origin_b: Origin
](
    a: MatrixView[Scalar[dtype], origin_a],
    b: MatrixView[Scalar[dtype], origin_b],
) raises -> Matrix[Scalar[dtype]]:
    """Element-wise modulo of two matrices or views."""
    return _elementwise_view[func=Scalar[dtype].__mod__](a, b)


# --------------------------------------------------------------------------- #
# pow
# --------------------------------------------------------------------------- #


def pow[
    dtype: DType, origin_a: Origin, origin_b: Origin
](
    a: MatrixView[Scalar[dtype], origin_a],
    b: MatrixView[Scalar[dtype], origin_b],
) raises -> Matrix[Scalar[dtype]]:
    """Element-wise exponentiation of two matrices or views."""
    return _elementwise_view[func=Scalar[dtype].__pow__](a, b)


# ===---------------------------------------------------------------------- ===#
# Scalar–Matrix operations: floordiv, mod, pow, and reflected sub / div
# ===---------------------------------------------------------------------- ===#
# One signature each; a `Matrix` operand converts implicitly.


def scalar_floordiv[
    dtype: DType, origin: Origin
](mat: MatrixView[Scalar[dtype], origin], scalar: Scalar[dtype]) -> Matrix[
    Scalar[dtype]
]:
    """Floor-divides every element of a matrix view by a scalar."""
    return _scalar_elementwise_view[func=Scalar[dtype].__floordiv__](
        mat, scalar
    )


def scalar_mod[
    dtype: DType, origin: Origin
](mat: MatrixView[Scalar[dtype], origin], scalar: Scalar[dtype]) -> Matrix[
    Scalar[dtype]
]:
    """Takes every element of a matrix view modulo a scalar."""
    return _scalar_elementwise_view[func=Scalar[dtype].__mod__](mat, scalar)


def scalar_pow[
    dtype: DType, origin: Origin
](mat: MatrixView[Scalar[dtype], origin], scalar: Scalar[dtype]) -> Matrix[
    Scalar[dtype]
]:
    """Raises every element of a matrix view to a scalar power."""
    return _scalar_elementwise_view[func=Scalar[dtype].__pow__](mat, scalar)


def scalar_rsub[
    dtype: DType, origin: Origin
](mat: MatrixView[Scalar[dtype], origin], scalar: Scalar[dtype]) -> Matrix[
    Scalar[dtype]
]:
    """Subtracts every element of a matrix view from a scalar (`scalar - mat`).
    """
    return _scalar_elementwise_view[func=_rsub_op[dtype]](mat, scalar)


def scalar_rdiv[
    dtype: DType, origin: Origin
](mat: MatrixView[Scalar[dtype], origin], scalar: Scalar[dtype]) -> Matrix[
    Scalar[dtype]
]:
    """Divides a scalar by every element of a matrix view (`scalar / mat`)."""
    return _scalar_elementwise_view[func=_rdiv_op[dtype]](mat, scalar)


# ===---------------------------------------------------------------------- ===#
# Multiplicative and extremal reductions
# ===---------------------------------------------------------------------- ===#
# `prod`, `min` and `max` are the same walk as `sum` in `routines/statistics
# .mojo` with a different accumulator, so they go through the same two pieces:
# `fold` for the whole matrix, `apply_along_axis` for one lane at a time.
#
# `min` and `max` seed the accumulator with the first element rather than with
# a sentinel, so they work for any dtype without needing to know its bounds,
# and they raise on an empty operand instead of returning a value that no
# element justifies.


def _mul_op[dtype: DType](a: Scalar[dtype], b: Scalar[dtype]) -> Scalar[dtype]:
    return a * b


def _min_op[dtype: DType](a: Scalar[dtype], b: Scalar[dtype]) -> Scalar[dtype]:
    return a if a < b else b


def _max_op[dtype: DType](a: Scalar[dtype], b: Scalar[dtype]) -> Scalar[dtype]:
    return a if a > b else b


def _prod_lane[
    dtype: DType, origin: Origin[mut=False]
](v: MatrixView[Scalar[dtype], origin]) -> Scalar[dtype]:
    return fold[func=_mul_op[dtype]](v, Scalar[dtype](1))


def _min_lane[
    dtype: DType, origin: Origin[mut=False]
](v: MatrixView[Scalar[dtype], origin]) -> Scalar[dtype]:
    return fold[func=_min_op[dtype]](v, v[0, 0])


def _max_lane[
    dtype: DType, origin: Origin[mut=False]
](v: MatrixView[Scalar[dtype], origin]) -> Scalar[dtype]:
    return fold[func=_max_op[dtype]](v, v[0, 0])


def prod[
    dtype: DType, origin: Origin[mut=False]
](m: MatrixView[Scalar[dtype], origin]) -> Scalar[dtype]:
    """Multiplies every element of a matrix or view.

    Parameters:
        dtype: The data type of the matrix elements.
        origin: The origin of the operand.

    Args:
        m: The matrix or view to reduce.

    Returns:
        The product of all elements, or one if the operand is empty.
    """
    return _prod_lane(m)


def prod[
    dtype: DType, origin: Origin[mut=False]
](m: MatrixView[Scalar[dtype], origin], axis: Int) raises -> Matrix[
    Scalar[dtype]
]:
    """Multiplies along one axis.

    Parameters:
        dtype: The data type of the matrix elements.
        origin: The origin of the operand.

    Args:
        m: The matrix or view to reduce.
        axis: The dimension to remove. 0 returns `1 x ncols`, 1 returns
            `nrows x 1`.

    Returns:
        A new matrix holding one product per lane.

    Raises:
        ValueError: If `axis` is neither 0 nor 1.
    """
    if axis == 0:
        return apply_along_axis[axis=0, func=_prod_lane[dtype, origin]](m)
    elif axis == 1:
        return apply_along_axis[axis=1, func=_prod_lane[dtype, origin]](m)
    raise ValueError(function="prod(m, axis)", message="Axis must be 0 or 1.")


def min[
    dtype: DType, origin: Origin[mut=False]
](m: MatrixView[Scalar[dtype], origin]) raises -> Scalar[dtype]:
    """Returns the smallest element of a matrix or view.

    Parameters:
        dtype: The data type of the matrix elements.
        origin: The origin of the operand.

    Args:
        m: The matrix or view to reduce.

    Returns:
        The smallest element.

    Raises:
        ValueError: If the operand is empty.
    """
    if m.size() == 0:
        raise ValueError(
            function="min(m)", message="Cannot reduce an empty matrix."
        )
    return _min_lane(m)


def min[
    dtype: DType, origin: Origin[mut=False]
](m: MatrixView[Scalar[dtype], origin], axis: Int) raises -> Matrix[
    Scalar[dtype]
]:
    """Returns the smallest element along one axis.

    Parameters:
        dtype: The data type of the matrix elements.
        origin: The origin of the operand.

    Args:
        m: The matrix or view to reduce.
        axis: The dimension to remove. 0 returns `1 x ncols`, 1 returns
            `nrows x 1`.

    Returns:
        A new matrix holding one minimum per lane.

    Raises:
        ValueError: If `axis` is neither 0 nor 1, or the operand is empty.
    """
    if m.size() == 0:
        raise ValueError(
            function="min(m, axis)", message="Cannot reduce an empty matrix."
        )
    if axis == 0:
        return apply_along_axis[axis=0, func=_min_lane[dtype, origin]](m)
    elif axis == 1:
        return apply_along_axis[axis=1, func=_min_lane[dtype, origin]](m)
    raise ValueError(function="min(m, axis)", message="Axis must be 0 or 1.")


def max[
    dtype: DType, origin: Origin[mut=False]
](m: MatrixView[Scalar[dtype], origin]) raises -> Scalar[dtype]:
    """Returns the largest element of a matrix or view.

    Parameters:
        dtype: The data type of the matrix elements.
        origin: The origin of the operand.

    Args:
        m: The matrix or view to reduce.

    Returns:
        The largest element.

    Raises:
        ValueError: If the operand is empty.
    """
    if m.size() == 0:
        raise ValueError(
            function="max(m)", message="Cannot reduce an empty matrix."
        )
    return _max_lane(m)


def max[
    dtype: DType, origin: Origin[mut=False]
](m: MatrixView[Scalar[dtype], origin], axis: Int) raises -> Matrix[
    Scalar[dtype]
]:
    """Returns the largest element along one axis.

    Parameters:
        dtype: The data type of the matrix elements.
        origin: The origin of the operand.

    Args:
        m: The matrix or view to reduce.
        axis: The dimension to remove. 0 returns `1 x ncols`, 1 returns
            `nrows x 1`.

    Returns:
        A new matrix holding one maximum per lane.

    Raises:
        ValueError: If `axis` is neither 0 nor 1, or the operand is empty.
    """
    if m.size() == 0:
        raise ValueError(
            function="max(m, axis)", message="Cannot reduce an empty matrix."
        )
    if axis == 0:
        return apply_along_axis[axis=0, func=_max_lane[dtype, origin]](m)
    elif axis == 1:
        return apply_along_axis[axis=1, func=_max_lane[dtype, origin]](m)
    raise ValueError(function="max(m, axis)", message="Axis must be 0 or 1.")


def cumprod[
    dtype: DType, origin: Origin[mut=False]
](m: MatrixView[Scalar[dtype], origin]) raises -> Matrix[Scalar[dtype]]:
    """Returns the running product over every element, in row-major order.

    Parameters:
        dtype: The data type of the matrix elements.
        origin: The origin of the operand.

    Args:
        m: The matrix or view to scan.

    Returns:
        A new C-contiguous matrix with the same shape as the input.
    """
    var result = Matrix[Scalar[dtype]](m.nrows(), m.ncols(), m.ncols(), 1)
    var acc = Scalar[dtype](1)
    var k = 0
    for i in range(m.nrows()):
        for j in range(m.ncols()):
            acc *= m[i, j]
            result._data[k] = acc
            k += 1
    return result^


def cumprod[
    dtype: DType, origin: Origin[mut=False]
](m: MatrixView[Scalar[dtype], origin], axis: Int) raises -> Matrix[
    Scalar[dtype]
]:
    """Returns the running product along one axis.

    Parameters:
        dtype: The data type of the matrix elements.
        origin: The origin of the operand.

    Args:
        m: The matrix or view to scan.
        axis: The dimension to accumulate along. 0 runs down each column,
            1 runs across each row.

    Returns:
        A new C-contiguous matrix with the same shape as the input.

    Raises:
        ValueError: If `axis` is neither 0 nor 1.
    """
    if axis != 0 and axis != 1:
        raise ValueError(
            function="cumprod(m, axis)", message="Axis must be 0 or 1."
        )

    var result = Matrix[Scalar[dtype]](m.nrows(), m.ncols(), m.ncols(), 1)
    if axis == 0:
        for j in range(m.ncols()):
            var acc = Scalar[dtype](1)
            for i in range(m.nrows()):
                acc *= m[i, j]
                result._data[i * m.ncols() + j] = acc
    else:
        for i in range(m.nrows()):
            var acc = Scalar[dtype](1)
            for j in range(m.ncols()):
                acc *= m[i, j]
                result._data[i * m.ncols() + j] = acc
    return result^


# ===---------------------------------------------------------------------- ===#
# Arbitrary-precision element-wise operations
# ===---------------------------------------------------------------------- ===#
# The same routine names again, for the element types that carry their
# arithmetic in `decimo.Numeric` rather than in a vector instruction:
# `BigInt`, `BigDecimal` and `Decimal128`. Each is one overload away from its
# scalar twin, and the `where` clauses are disjoint, so `add(a, b)` is the same
# call whichever kind of element the operands hold.
#
# There is no SIMD path and no `parallelize` here, and that is not an omission.
# A `BigInt` addition allocates, so the loop is memory-bound rather than
# issue-bound: the element-by-element walk below is what the operation costs.


def _elementwise_numeric[
    T: Numeric,
    func: def(T, T) raises thin -> T,
    origin_a: Origin,
    origin_b: Origin,
](a: MatrixView[T, origin_a], b: MatrixView[T, origin_b]) raises -> Matrix[T]:
    """Core element-wise binary operation on two views of the same shape.

    The result is always a freshly allocated, C-contiguous matrix.
    """
    if a.nrows() != b.nrows() or a.ncols() != b.ncols():
        raise ValueError(
            function="_elementwise_numeric()",
            message="Input matrices must have the same shape.",
        )
    var buffer = List[T](capacity=a.nrows() * a.ncols())
    for i in range(a.nrows()):
        for j in range(a.ncols()):
            buffer.append(func(a[i, j], b[i, j]))
    return Matrix[T](buffer^, a.nrows(), a.ncols(), a.ncols(), 1)


def _scalar_elementwise_numeric[
    T: Numeric,
    func: def(T, T) raises thin -> T,
    origin: Origin,
](mat: MatrixView[T, origin], scalar: T) raises -> Matrix[T]:
    """Core scalar-matrix element-wise operation on one view."""
    var buffer = List[T](capacity=mat.nrows() * mat.ncols())
    for i in range(mat.nrows()):
        for j in range(mat.ncols()):
            buffer.append(func(mat[i, j], scalar))
    return Matrix[T](buffer^, mat.nrows(), mat.ncols(), mat.ncols(), 1)


def add[
    T: Numeric, origin_a: Origin, origin_b: Origin
](a: MatrixView[T, origin_a], b: MatrixView[T, origin_b]) raises -> Matrix[T]:
    """Element-wise addition of two matrices or views."""
    return _elementwise_numeric[func=T.__add__](a, b)


def sub[
    T: Numeric, origin_a: Origin, origin_b: Origin
](a: MatrixView[T, origin_a], b: MatrixView[T, origin_b]) raises -> Matrix[T]:
    """Element-wise subtraction of two matrices or views."""
    return _elementwise_numeric[func=T.__sub__](a, b)


def mul[
    T: Numeric, origin_a: Origin, origin_b: Origin
](a: MatrixView[T, origin_a], b: MatrixView[T, origin_b]) raises -> Matrix[T]:
    """Element-wise multiplication of two matrices or views."""
    return _elementwise_numeric[func=T.__mul__](a, b)


def div[
    T: Numeric, origin_a: Origin, origin_b: Origin
](a: MatrixView[T, origin_a], b: MatrixView[T, origin_b]) raises -> Matrix[T]:
    """Element-wise division of two matrices or views.

    On an integral element type this truncates toward zero, as `Int` does.
    """
    return _elementwise_numeric[func=T.__truediv__](a, b)


def scalar_add[
    T: Numeric, origin: Origin
](mat: MatrixView[T, origin], scalar: T) raises -> Matrix[T]:
    """Adds a value to every element of a matrix or view."""
    return _scalar_elementwise_numeric[func=T.__add__](mat, scalar)


def scalar_sub[
    T: Numeric, origin: Origin
](mat: MatrixView[T, origin], scalar: T) raises -> Matrix[T]:
    """Subtracts a value from every element of a matrix or view."""
    return _scalar_elementwise_numeric[func=T.__sub__](mat, scalar)


def scalar_mul[
    T: Numeric, origin: Origin
](mat: MatrixView[T, origin], scalar: T) raises -> Matrix[T]:
    """Multiplies every element of a matrix or view by a value."""
    return _scalar_elementwise_numeric[func=T.__mul__](mat, scalar)


def scalar_div[
    T: Numeric, origin: Origin
](mat: MatrixView[T, origin], scalar: T) raises -> Matrix[T]:
    """Divides every element of a matrix or view by a value."""
    return _scalar_elementwise_numeric[func=T.__truediv__](mat, scalar)


def neg[
    T: Numeric, origin: Origin
](a: MatrixView[T, origin]) raises -> Matrix[T]:
    """Negates every element of a matrix or view."""
    var buffer = List[T](capacity=a.nrows() * a.ncols())
    for i in range(a.nrows()):
        for j in range(a.ncols()):
            buffer.append(-a[i, j])
    return Matrix[T](buffer^, a.nrows(), a.ncols(), a.ncols(), 1)


def _rsub_numeric[T: Numeric](a: T, b: T) raises -> T:
    return b - a


def _rdiv_numeric[T: Numeric](a: T, b: T) raises -> T:
    return b / a


def scalar_rsub[
    T: Numeric, origin: Origin
](mat: MatrixView[T, origin], scalar: T) raises -> Matrix[T]:
    """Subtracts every element of a matrix or view from a value."""
    return _scalar_elementwise_numeric[func=_rsub_numeric[T]](mat, scalar)


def scalar_rdiv[
    T: Numeric, origin: Origin
](mat: MatrixView[T, origin], scalar: T) raises -> Matrix[T]:
    """Divides a value by every element of a matrix or view."""
    return _scalar_elementwise_numeric[func=_rdiv_numeric[T]](mat, scalar)


def matmul[
    T: Numeric, origin_a: Origin, origin_b: Origin
](a: MatrixView[T, origin_a], b: MatrixView[T, origin_b]) raises -> Matrix[T]:
    """Performs matrix multiplication.

    A plain triple loop. The scalar kernel's four layout-specialised SIMD paths
    exist to keep a vector unit fed; here every `+` and `*` allocates, so the
    layout of the operands is not what the time is spent on, and one readable
    loop is the honest implementation.

    Raises:
        ValueError: If the inner dimensions do not match.
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
