# Roadmap <!-- omit in toc -->

Linamo development roadmap. Phases are prioritized for use as the linear
algebra foundation of [stamojo](https://github.com/mojomath/stamojo) (a
statistical modeling library, similar to statsmodels).

Last reviewed: **2026-08-15**

- [Phase 0 — Core Types \& Basic Operations](#phase-0--core-types--basic-operations)
- [Phase 1 — Matrix Fundamentals](#phase-1--matrix-fundamentals)
- [Phase 2 — Decompositions](#phase-2--decompositions)
- [Phase 3 — Solvers \& Inverse](#phase-3--solvers--inverse)
- [Phase 4 — Mojo 1.0.0 Migration](#phase-4--mojo-100-migration)
- [Phase 5 — NuMojo Matrix Consolidation](#phase-5--numojo-matrix-consolidation)
  - [5.1 — Indexing \& iteration](#51--indexing--iteration)
  - [5.2 — Operators](#52--operators)
  - [5.3 — Reductions \& search](#53--reductions--search)
  - [5.4 — Shape \& layout manipulation](#54--shape--layout-manipulation)
  - [5.5 — Creation routines](#55--creation-routines)
  - [5.6 — Element-wise math \& logic](#56--element-wise-math--logic)
  - [5.7 — Linear algebra gaps](#57--linear-algebra-gaps)
  - [5.8 — Interop \& NuMojo hand-off](#58--interop--numojo-hand-off)
- [Phase 6 — Eigenvalue Problems](#phase-6--eigenvalue-problems)
- [Phase 7 — Statistics Primitives](#phase-7--statistics-primitives)
- [Phase 8 — Norms \& Conditioning](#phase-8--norms--conditioning)
- [Phase 9 — Random Matrix Generation](#phase-9--random-matrix-generation)
- [Phase 10 — Performance \& Polish](#phase-10--performance--polish)
- [Review Log](#review-log)

---

## Phase 0 — Core Types & Basic Operations

> **Status: ✓ Done**

| Item                                                   | Module                     | Status |
| ------------------------------------------------------ | -------------------------- | ------ |
| `Matrix` type (dynamic, row/col-major)                 | `types/matrix.mojo`        | ✓      |
| `StaticMatrix` type (compile-time sized)               | `types/static_matrix.mojo` | ✓      |
| `MatrixView` (non-owning view, slicing, view-on-view)  | `types/matrix_view.mojo`   | ✓      |
| `MatrixLike` trait                                     | `traits/matrix_like.mojo`  | ✓      |
| `matrix()` / `smatrix()` creation routines             | `routines/creation.mojo`   | ✓      |
| Element-wise `add`, `sub`, `mul`, `div` (StaticMatrix) | `routines/math.mojo`       | ✓      |
| `matmul` (naive + SIMD/parallel for dynamic)           | `routines/math.mojo`       | ✓      |
| Custom error types (`ValueError`, `IndexError`, etc.)  | `types/errors.mojo`        | ✓      |
| Unit test suite (88 tests, TestSuite.discover_tests)   | `tests/`                   | ✓      |
| CI: GitHub Actions + pre-commit (mojo format)          | `.github/workflows/`       | ✓      |

---

## Phase 1 — Matrix Fundamentals

> **Status: ✓ Done**
>
> *stamojo dependency: blocking — nearly every statistical model needs these.*

| Item                                    | Module                   | stamojo use                         | Status |
| --------------------------------------- | ------------------------ | ----------------------------------- | ------ |
| `transpose()` / `.T` property           | `routines/linalg.mojo`   | Design matrices, X^T X              | ✓      |
| `eye()` / `identity()`                  | `routines/creation.mojo` | Ridge regression, regularization    | ✓      |
| `diag()` (extract / construct diagonal) | `routines/creation.mojo` | Variance extraction from cov matrix | ✓      |
| `trace()`                               | `routines/linalg.mojo`   | Matrix diagnostics                  | ✓      |
| `zeros()` / `ones()` / `full()`         | `routines/creation.mojo` | Convenience constructors            | ✓      |
| Element-wise ops for dynamic            | `routines/math.mojo`     | Residual computation                | ✓      |
| `Matrix` (`add`, `sub`, `mul`, `div`)   |                          |                                     |        |
| Scalar–matrix operations                | `routines/math.mojo`     | Scaling, centering                  | ✓      |
| (`scalar_add/sub/mul/div`)              |                          |                                     |        |
| Operator overloads (`+`, `-`, `*`, `/`) | `types/matrix.mojo`      | Ergonomic element-wise syntax       | ✓      |
| for dynamic `Matrix`                    |                          |                                     |        |

---

## Phase 2 — Decompositions

> **Status: ✓ Done**
>
> *stamojo dependency: blocking — cannot implement OLS, GLS, or WLS without
> these.*

| Item                           | Module                 | stamojo use                            | Status |
| ------------------------------ | ---------------------- | -------------------------------------- | ------ |
| LU decomposition               | `routines/linalg.mojo` | `solve()`, `inv()`, `det()`            | ✓      |
| (with partial pivoting)        |                        |                                        |        |
| Cholesky decomposition         | `routines/linalg.mojo` | Efficient solve for positive-definite  | ✓      |
|                                |                        | (covariance) matrices                  |        |
| QR decomposition (Householder) | `routines/linalg.mojo` | Numerically stable least squares (OLS) | ✓      |

---

## Phase 3 — Solvers & Inverse

> **Status: ✓ Done**
>
> *stamojo dependency: blocking — regression coefficients require `solve` or
> `inv`.*

| Item                               | Module                 | stamojo use            | Status |
| ---------------------------------- | ---------------------- | ---------------------- | ------ |
| `det()` — determinant (via LU)     | `routines/linalg.mojo` | Singularity check      | ✓      |
| `solve()` — solve Ax = b           | `routines/linalg.mojo` | Linear system solving  | ✓      |
| `inv()` — matrix inverse           | `routines/linalg.mojo` | β̂ = (X^T X)^{-1} X^T y | ✓      |
| `lstsq()` — least squares (via QR) | `routines/linalg.mojo` | OLS regression         | ✓      |

---

## Phase 4 — Mojo 1.0.0 Migration

> **Status: ✓ Done**
>
> *Was blocking: the codebase did not compile on Mojo 1.x.*

Mojo 1.0.0 (released 2026-08-11) landed a large set of breaking changes. See the
[v1.0.0 release notes](https://mojolang.org/releases/v1.0.0/).

| Item                                                                       | Scope                         | Status |
| -------------------------------------------------------------------------- | ----------------------------- | ------ |
| Pin toolchain to `mojo >=1.0.0,<1.1.0` on the **stable** channel           | `pixi.toml`                   | ✓      |
| Add `max >=26.5.0,<26.6` (`parallelize()` moved to MAX)                    | `pixi.toml`                   | ✓      |
| `fn` removed → `def` (same semantics; `raises` before `->`)                | `src/`, `tests/`, `examples/` | ✓      |
| Stdlib imports must be `std.`-qualified                                    | all modules                   | ✓      |
| `Stringable` removed → conform to `Writable` only                          | all types                     | ✓      |
| `__copyinit__`/`__moveinit__` → `__init__(*, copy:)` / `(*, deinit move:)` | `types/`                      | ✓      |
| Unified closures: drop `@parameter`, `unified` kw; `read` → `imm`          | `routines/math.mojo`          | ✓      |
| Function-type annotations `fn(...) ->` → `def(...) thin ->`                | `routines/math.mojo`          | ✓      |
| Interior origins: `ref[self.data[...]]` return origin                      | `types/matrix.mojo`           | ✓      |
| Typed raises are invariant → raise plain `Error` (see note)                | `types/errors.mojo`, all      | ✓      |
| Pointer ops → `unsafe_load` / `unsafe_store` / `unsafe_offset=`            | `routines/`, `utils/`         | ✓      |
| `memcpy` → `unsafe_memcpy`                                                 | `routines/numpy_interop.mojo` | ✓      |
| `@parameter if` / `@parameter for` → `comptime if` / `comptime for`        | `routines/`                   | ✓      |
| Implicit variable declarations → explicit `var`                            | `routines/`, `types/`         | ✓      |
| `mojo package` + `.mojopkg` → `mojo precompile` + `.mojoc`                 | `pixi.toml`, `.gitignore`     | ✓      |
| List literals now build `Array` → pass explicit dtype                      | `examples/`                   | ✓      |
| Test suite green (214 tests), zero build warnings                          | `tests/`                      | ✓      |
| CI: add `linux-64` so the declared ubuntu leg can actually solve           | `pixi.toml`                   | ✓      |

> **Note — typed raises had to be dropped.** Linamo previously declared
> `raises ValueError` (etc.) on its public routines. Mojo 1.0.0 makes typed
> raises strictly **invariant**: a `raises Error` function cannot call a
> `raises ValueError` one, *and vice versa*. That makes a typed-raise public API
> impossible to combine with `std.testing` (which raises `Error`) or with any
> downstream caller such as stamojo.
>
> Resolution: the error kinds in `types/errors.mojo` (`ValueError`,
> `IndexError`, …) changed from **type aliases** into **factory functions** that
> build a `LinamoError` payload and wrap it in a plain `Error`. Every
> `raise ValueError(file=..., function=..., message=...)` call site is
> unchanged, and the rich traceback formatting is preserved, because `Error` is
> constructed from a `Writable`. Only the signatures changed:
> `raises ValueError ->` became `raises ->`. Revisit if Mojo adds error-type
> widening.

---

## Phase 5 — NuMojo Matrix Consolidation

> **Status: 🚧 In progress — 5.1 and 5.2 done**
>
> NuMojo is dropping its `Matrix` type (`numojo/core/matrix/`), and it lives
> here from now on. So Linamo needs to cover what NuMojo users are losing.

We're after the *functionality*, not the API. NuMojo's `Matrix` handed out
pointer-backed sub-matrices; Linamo splits ownership into `Matrix` (owning)
and `MatrixView` (non-owning, origin-tracked), and that split is the point of
the library. Nothing below should reintroduce an `UnsafePointer` in a public
signature, and every view has to carry its `origin` so the borrow checker can
do the work that runtime flags would otherwise have to.

Where a port would fight that model, the API changes and we write down why.

### 5.1 — Indexing & iteration

| Item                                         | Module                   | Status |
| -------------------------------------------- | ------------------------ | ------ |
| `__len__` (row count)                        | `types/matrix.mojo`      | ✓      |
| Row / column iterators                       | `types/matrix_iter.mojo` | ✓      |
| `load[width]` / `store[width]` (SIMD access) | `types/matrix.mojo`      | ✓      |
| Region assignment (`fill`, `assign`)         | `routines/mutation.mojo` | ✓      |
| Mutable views via `ref self`                 | `types/matrix.mojo`      | ✓      |
| `to_matrix()` (materialise a view)           | `types/matrix_view.mojo` | ✓      |

Three things came out differently than the original sketch, all of them forced
by the language rather than chosen:

**Mutable views had to be built before anything else.** `view()` took `self`
as an immutable borrow, so every view it produced was read-only — the `mut`
parameter on `MatrixView` could never be `True`. Taking `ref self` instead lets
the caller's mutability flow into the origin, which is what makes `store` and
region assignment possible at all. A mutable matrix now yields writable views,
a borrowed one yields read-only views, and writing through a read-only view
fails to compile. (Both `view()` and slice indexing were given the same
treatment here, and both had to be walked back in 5.2 — see *Slicing had to
become read-only* below. Mutable views survive, but only as `view_mut` in
`routines/mutation.mojo`; no method returns one.)

**Bulk writes on views can't be methods.** `MatrixView` is generic over
`origin`, and Mojo checks a method body against every instantiation including
the read-only one, so anything writing through `self.data` is rejected where
it's defined. Neither a `where Self.mut` clause nor a constrained `self`
refines it. The bulk write operations therefore live in `routines/mutation.mojo`
as free functions pinned to `Origin[mut=True]`, which moves the requirement
into the signature — passing a read-only view is a compile error at the call
site. Single-element writes are unaffected: `v[i, j] = x` goes through the
reference `__getitem__` returns.

**Region assignment isn't `__setitem__`.** Mojo routes `a[i:j, k:l] = rhs`
through `__getitem__`, so `rhs` has to be convertible to whatever `__getitem__`
returns — here, a view carrying the target's *own* origin. That makes assigning
from any other matrix inexpressible as subscript sugar, so it's spelled
`fill(...)` and `assign(...)` instead.

Two smaller notes: the iterator is parameterised on axis and direction rather
than hardwired to forward rows, because that's the traversal `apply_along_axis`
needs in 5.3. And Mojo's builtin `reversed()` only accepts specific stdlib
containers, so it won't dispatch to `__reversed__` — use `rows[False]()`.

### 5.2 — Operators

| Item                                                             | Module                | Status |
| ---------------------------------------------------------------- | --------------------- | ------ |
| In-place ops `+=`, `-=`, `*=`, `/=`, `//=`, `%=`                 | `types/matrix.mojo`   | ✓      |
| `__pow__`, `__floordiv__`, `__mod__`                             | `types/matrix.mojo`   | ✓      |
| Reflected ops `__radd__`, `__rsub__`, `__rmul__`                 | `types/matrix.mojo`   | ✓      |
| Comparison ops `<`, `<=`, `>`, `>=`, `==`, `!=` → `Matrix[bool]` | `routines/logic.mojo` | ✓      |

Four notes on how this landed.

**In-place operators exist on `Matrix` only.** They write back through the
matrix's own strides instead of allocating, so a transposed or column-major
matrix keeps its layout. `MatrixView` gets no `+=`: it's generic over `origin`
and Mojo checks the body against the read-only instantiation too, which is the
same wall 5.1 hit. Mutating a view still goes through
`routines/mutation.mojo`. Aliasing like `a += a[:, :]` never compiles — the
borrow checker won't hand out a mutable reference to `a` while a view of it is
alive, which is exactly the guarantee the two-type split is for.

**Comparisons return a mask, not a verdict.** `a == b` is an element-wise
`Matrix[DType.bool]`, as in NumPy, so `Matrix` deliberately does not conform to
`EqualityComparable`. Asking whether two matrices are wholly identical stays a
separate question, answered by `assert_matrices_equal`. The comparison kernels
went into a new `routines/logic.mojo` rather than `math.mojo`, which is where
5.3 wants `all` / `any` anyway.

**`__pow__` is element-wise.** `A ** 2` squares each entry, matching NumPy.
Matrix exponentiation is a different operation and will get a named routine
rather than an operator.

**Slicing had to become read-only.** 5.1 gave slice indexing `ref self`, along
with `view()`, so `a[0:2, 0:2]` on a `var` matrix produced a *mutable* view.
That turns out to be unusable once operators exist. A mutable view is an
exclusive borrow, and Mojo refuses to pass two values that both borrow the same
memory into one call — so `a[0:1, :] - a[1:2, :]`, `a + a[0:2, 0:2]` and
`a[0:2, 0:2] @ a[0:2, 0:2]` were all rejected by the compiler. Nothing caught
it because every view test until now paired views taken from two *different*
matrices.

Slicing therefore takes `self` by `read` and always yields a read-only view.
Reading one matrix twice at once is always safe, so all of the above compile.

Fixing slicing alone was not enough, and the first attempt kept `view()` and
added `view(x, y)` as a mutable door on both types. That left `m.view()`
returning a *mutable* view of a `var` matrix — a write door behind the most
innocent-looking call in the API — and left the invariant as a set of
remembered exceptions rather than a rule. The design that replaced it states
one rule instead:

> Nothing that carries a borrow in its type is ever handed out mutable, except
> through a function in `linamo.routines.mutation`.

A method can only propagate the caller's mutability by taking `ref self`, so
that rule is checkable with `grep -rn "ref self" src/linamo/types/`, which now
returns exactly one line: element access on `Matrix`. `view()`, `rows()`,
`cols()` and iteration are all `read self`; `view(x, y)` is gone from both
types; `view_mut`, `rows_mut` and `cols_mut` live in `routines/mutation.mojo`
alongside `fill`, `store` and `assign`, so a caller who never imports that
module cannot construct a mutable view at all.
`tests/matrix_view/test_view_aliasing.mojo` exists so this blind spot cannot
reopen.

**Element access had a narrower bug.** `Matrix.__getitem__` returned a
reference whose origin named one computed element,
`self.data[row * row_stride + col * col_stride]`. Forming a second such
reference invalidated the first, so `a[0, 0] + a[1, 1]` did not compile on a
`var` matrix — adding two elements of a matrix. Returning through
`origin_of(self.data)`, the whole buffer, lets any number of element
references coexist.

**`__setitem__` is not available.** The obvious way to make indexing and
slicing symmetric is `m[a:b, c:d] = src` with a `mut self` setter, which lets
no view escape and looked ideal. It cannot be used: merely defining
`__setitem__` on `Matrix` makes the compiler pass `self` to `__getitem__` as a
temporary copy in some positions, so a sliced view carries the origin of a dead
temporary and `a[0:1, :] - a[1:2, :]` stops compiling again. Reproduced in
twenty lines with no linamo involved, for both `Int` and `Slice` setters.
Element writes therefore keep going through the reference `m[i, j]` returns,
and region writes are spelled `assign(...)`.

**`__rtruediv__` came along uninvited.** The table lists three reflected
operators, but shipping `2.0 - A` without `2.0 / A` is a worse API than either
having all four or none, so division is in too.

One deviation worth recording: the reflected and scalar forms are mirrored onto
`MatrixView` as well. The view already had `__add__` and friends, and leaving it
with arithmetic but no comparisons would have been an odd gap in the type that
the library is built around.

### 5.3 — Reductions & search

| Item                                 | Module                     | Status |
| ------------------------------------ | -------------------------- | ------ |
| `sum` / `cumsum` (axis + full)       | `routines/statistics.mojo` | □      |
| `prod` / `cumprod` (axis + full)     | `routines/math.mojo`       | □      |
| `min` / `max` (axis + full)          | `routines/math.mojo`       | □      |
| `argmin` / `argmax` (axis + full)    | `routines/searching.mojo`  | □      |
| `all` / `any`                        | `routines/logic.mojo`      | □      |
| `sort` / `argsort` / `sort_inplace`  | `routines/sorting.mojo`    | □      |
| `apply_along_axis` (generic applier) | `routines/functional.mojo` | □      |

> **On the generic applier.** Rather than hand-writing an axis loop for each of
> the reductions above, port the `apply_along_axis` idea from NuMojo
> (`numojo/routines/functional.mojo`): take the per-lane function as a
> compile-time parameter, walk the axis once, and let each reduction supply
> only its own kernel. The axis iterator from 5.1 is already the traversal it
> needs. Worth specialising on layout — `col_stride == 1`, `row_stride == 1`,
> and a strided fallback — so the same three-way dispatch that matmul uses is
> written once here instead of being repeated per routine. It lands with the
> reductions rather than before them, so it has real callers to be shaped by.

### 5.4 — Shape & layout manipulation

| Item                            | Module                       | Status |
| ------------------------------- | ---------------------------- | ------ |
| `reshape`                       | `routines/manipulation.mojo` | □      |
| `resize`                        | `routines/manipulation.mojo` | □      |
| `flatten`                       | `routines/manipulation.mojo` | □      |
| `contiguous` / `reorder_layout` | `routines/manipulation.mojo` | □      |
| `broadcast_to`                  | `routines/manipulation.mojo` | □      |
| `astype[dtype]`                 | `types/matrix.mojo`          | □      |
| `fill`                          | `types/matrix.mojo`          | □      |

### 5.5 — Creation routines

| Item                                      | Module                   | Status |
| ----------------------------------------- | ------------------------ | ------ |
| `empty`                                   | `routines/creation.mojo` | □      |
| `arange`                                  | `routines/creation.mojo` | □      |
| `linspace`                                | `routines/creation.mojo` | □      |
| `zeros_like` / `ones_like` / `empty_like` | `routines/creation.mojo` | □      |
| `fromlist` / `fromstring`                 | `routines/creation.mojo` | □      |
| `rand` (see also Phase 9)                 | `routines/random.mojo`   | □      |

### 5.6 — Element-wise math & logic

| Item                                                          | Module                | Status |
| ------------------------------------------------------------- | --------------------- | ------ |
| Trig: `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2`    | `routines/math.mojo`  | □      |
| Hyperbolic: `sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh` | `routines/math.mojo`  | □      |
| `round`                                                       | `routines/math.mojo`  | □      |
| `isclose` / `allclose`                                        | `routines/logic.mojo` | □      |
| `isposinf` / `isneginf`                                       | `routines/logic.mojo` | □      |
| `logical_and` / `logical_or` / `logical_not` / `logical_xor`  | `routines/logic.mojo` | □      |

### 5.7 — Linear algebra gaps

| Item                                 | Module                 | Status |
| ------------------------------------ | ---------------------- | ------ |
| `issymmetric`                        | `routines/linalg.mojo` | □      |
| `solve_lu` (explicit LU-based solve) | `routines/linalg.mojo` | □      |
| `eig` (ported from NuMojo — Phase 6) | `routines/linalg.mojo` | □      |

### 5.8 — Interop & NuMojo hand-off

| Item                                                 | Module                        | Status |
| ---------------------------------------------------- | ----------------------------- | ------ |
| `to_numpy` / `matrix_from_numpy`                     | `routines/numpy_interop.mojo` | ✓      |
| `to_ndarray` equivalent (NuMojo `NDArray` bridge)    | TBD — see note                | □      |
| Migration guide for NuMojo users                     | `docs/`                       | □      |
| Coordinate removal of `numojo/core/matrix/` upstream | NuMojo repo                   | □      |

> **Note on `to_ndarray`:** NuMojo's `Matrix.to_ndarray()` converts to NuMojo's
> own `NDArray`. Reproducing it here would make Linamo depend on NuMojo, which
> inverts the intended dependency direction. Preferred resolution: NuMojo grows
> an `NDArray.from_linamo()` (or equivalent) on its side, and Linamo stays
> dependency-free. Decision pending.

---

## Phase 6 — Eigenvalue Problems

> **Status: □ Not started**
>
> *stamojo dependency: important for PCA and diagnostics, not blocking for basic
> regression.*

| Item                                   | Module                 | stamojo use                         |
| -------------------------------------- | ---------------------- | ----------------------------------- |
| `eig()` — eigenvalues + eigenvectors   | `routines/linalg.mojo` | PCA, principal component regression |
| `eigvals()` — eigenvalues only         | `routines/linalg.mojo` | Condition number, multicollinearity |
| `svd()` — singular value decomposition | `routines/linalg.mojo` | Pseudo-inverse, rank, PCA           |

---

## Phase 7 — Statistics Primitives

> **Status: □ Not started**
>
> *stamojo dependency: important — descriptive stats and residual analysis.*
>
> *Overlaps Phase 5.3: `sum`, `mean`, `var`, `std` arrive with the NuMojo
> consolidation; `cov` and `corrcoef` are new work.*

| Item                              | Module                     | stamojo use                          |
| --------------------------------- | -------------------------- | ------------------------------------ |
| `sum()` (along axis / full)       | `routines/statistics.mojo` | Data aggregation                     |
| `mean()` (along axis / full)      | `routines/statistics.mojo` | Centering, descriptive stats         |
| `var()` / `std()` (along axis)    | `routines/statistics.mojo` | Variance estimation, standardization |
| `cov()` — covariance matrix       | `routines/statistics.mojo` | Covariance estimation                |
| `corrcoef()` — correlation matrix | `routines/statistics.mojo` | Correlation analysis                 |

---

## Phase 8 — Norms & Conditioning

> **Status: □ Not started**
>
> *stamojo dependency: useful for diagnostics and numerical stability.*

| Item                                      | Module                 | stamojo use                  |
| ----------------------------------------- | ---------------------- | ---------------------------- |
| `norm()` (Frobenius, L1, L2, inf)         | `routines/linalg.mojo` | Residual norms, convergence  |
| `cond()` — condition number               | `routines/linalg.mojo` | Multicollinearity detection  |
| `rank()` — matrix rank                    | `routines/linalg.mojo` | Rank-deficiency check        |
| `pinv()` — pseudo-inverse (Moore–Penrose) | `routines/linalg.mojo` | Rank-deficient least squares |

---

## Phase 9 — Random Matrix Generation

> **Status: □ Not started**
>
> *stamojo dependency: needed for simulation, bootstrap, MCMC.*

| Item                             | Module                 | stamojo use                 |
| -------------------------------- | ---------------------- | --------------------------- |
| `rand()` — uniform random matrix | `routines/random.mojo` | Monte Carlo simulation      |
| `randn()` — normal random matrix | `routines/random.mojo` | Error simulation, bootstrap |
| `seed()` — set RNG seed          | `routines/random.mojo` | Reproducibility             |

---

## Phase 10 — Performance & Polish

> **Status: □ Not started**

| Item                                   | Module                 | Notes                    |
| -------------------------------------- | ---------------------- | ------------------------ |
| Optimized matmul for all layout combos | `routines/math.mojo`   | See [API.md](API.md)     |
| (C@C, F@F, C@F, F@C, V@*)              |                        |                          |
| Tiled / blocked matmul                 | `routines/math.mojo`   | Cache-friendly           |
| SIMD-optimized decompositions          | `routines/linalg.mojo` | Performance              |
| Parallel row/col operations            | `routines/math.mojo`   | Multi-core utilization   |
| Comprehensive benchmarks               | `benches/`             | Compare vs. NumPy/LAPACK |

---

## Review Log

| Date       | Notes                                                         |
| ---------- | ------------------------------------------------------------- |
| 2026-02-18 | Initial roadmap created. Phase 0 complete.                    |
| 2025-07-11 | Phase 1 complete: creation, linalg, elementwise & scalar ops, |
|            | dunders. 88 tests total.                                      |
| 2026-02-19 | Phase 2 complete: LU (partial pivoting), Cholesky,            |
|            | QR (Householder). 20 new tests.                               |
| 2026-08-15 | Added Phase 4 (Mojo 1.0.0 migration) and Phase 5              |
|            | (NuMojo `Matrix` consolidation); renumbered later phases.     |
| 2026-08-15 | Phase 4 complete: migrated to Mojo 1.0.0.                     |
|            | All tests pass, zero warnings.                                |
| 2026-08-15 | Phase 5.1 done: mutable views via `ref self`, axis iterators, |
|            | SIMD load/store, region assignment. 17 new tests.             |
| 2026-08-15 | Renamed the package from MatMojo to Linamo; import alias      |
|            | `mm` -> `la`.                                                 |
| 2026-08-15 | Phase 5.2 done: in-place, floordiv/mod/pow, reflected and     |
|            | comparison operators; new `routines/logic.mojo`. 35 new       |
|            | tests (266 total).                                            |
| 2026-08-15 | Slicing now yields read-only views, so two views of one       |
|            | matrix compose (`a[0:1, :] - a[1:2, :]`). Added               |
|            | `MatrixView.as_imm()`. 13 new tests (280 total).              |
| 2026-08-16 | Closed the last write door: `view()`, `rows()`, `cols()` and  |
|            | iteration are `read self`; `view(x, y)` removed; `view_mut`,  |
|            | `rows_mut`, `cols_mut` added to `routines/mutation.mojo`.     |
|            | Fixed `a[0, 0] + a[1, 1]`, which did not compile. 7 new       |
|            | tests (287 total).                                            |
| 2026-08-16 | Reworked `types/errors.mojo` against Decimo's version:        |
|            | file and line are captured with `call_location()`, paths are  |
|            | shortened to `./src/...`, tracebacks are ANSI-coloured and    |
|            | chain Python-style. Dropped the hand-written `file=` argument |
|            | from all 33 raise sites. Restored the Apache-2.0 attribution. |
| 2026-08-16 | Consolidated `examples/` into `matrix.mojo`,                  |
|            | `matrix_view.mojo` and `static_matrix.mojo`, one per public   |
|            | type. `matrix_view.mojo` now covers slicing, view-on-view,    |
|            | element and region writes, `view_mut`/`as_imm` and mutable    |
|            | iteration. Added `pixi run examples`, and exported            |
|            | `StaticMatrix` from `linamo/__init__.mojo`.                   |
