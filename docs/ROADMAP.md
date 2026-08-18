# Roadmap <!-- omit in toc -->

Linamo development roadmap. Phases are prioritized for use as the linear
algebra foundation of [stamojo](https://github.com/mojomath/stamojo) (a
statistical modeling library, similar to statsmodels).

Last reviewed: **2026-08-18**

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
  - [5.8 — Interop](#58--interop)
  - [5.9 — API consolidation \& hardening](#59--api-consolidation--hardening)
- [Phase 6 — Eigenvalue Problems](#phase-6--eigenvalue-problems)
- [Phase 7 — Statistics Primitives](#phase-7--statistics-primitives)
- [Phase 8 — Norms \& Conditioning](#phase-8--norms--conditioning)
- [Phase 9 — Random Matrix Generation](#phase-9--random-matrix-generation)
- [Phase 10 — Performance \& Polish](#phase-10--performance--polish)
- [Documentation](#documentation)
- [Release Plan — v0.1.0](#release-plan--v010)
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

> **Note — typed raises had to be dropped.** Linamo used to declare
> `raises ValueError` (etc.) on its public routines. Mojo 1.0.0 makes typed
> raises strictly **invariant**: a `raises Error` function cannot call a
> `raises ValueError` one, *and vice versa*, which makes a typed-raise public
> API impossible to combine with `std.testing` or with any downstream caller
> such as stamojo. The error kinds in `types/errors.mojo` therefore changed from
> type aliases into factory functions that build a `LinamoError` payload and
> wrap it in a plain `Error`. Every
> `raise ValueError(function=..., message=...)` call site is unchanged and the
> rich traceback survives, because `Error` is built from a `Writable`; only the
> signatures changed. Revisit if Mojo adds error-type widening.

---

## Phase 5 — NuMojo Matrix Consolidation

> **Status: 🚧 In progress — 5.1 through 5.4 done**
>
> NuMojo dropped its `Matrix` type (`numojo/core/matrix/`), and it lives here
> from now on. Its API is the checklist for this phase — not because Linamo
> inherits its users, but because it is a known-complete list of what a matrix
> type has to do.

We're after the *functionality*, not the API. NuMojo's `Matrix` handed out
pointer-backed sub-matrices; Linamo splits ownership into `Matrix` (owning) and
`MatrixView` (non-owning, origin-tracked), and that split is the point of the
library. So nothing below reintroduces an `UnsafePointer` in a public signature,
and every view carries its `origin`. Where a port fights that model, the API
changes and the reason is written down.

### 5.1 — Indexing & iteration

| Item                                         | Module                   | Status |
| -------------------------------------------- | ------------------------ | ------ |
| `__len__` (row count)                        | `types/matrix.mojo`      | ✓      |
| Row / column iterators                       | `types/matrix_iter.mojo` | ✓      |
| `load[width]` / `store[width]` (SIMD access) | `types/matrix.mojo`      | ✓      |
| Region assignment (`fill`, `assign`)         | `routines/mutation.mojo` | ✓      |
| Mutable views (`view_mut`, `rows_mut`)       | `routines/mutation.mojo` | ✓      |
| `to_matrix()` (materialise a view)           | `types/matrix_view.mojo` | ✓      |

Three deviations from the sketch, all forced by the language.

**Bulk writes on views can't be methods.** `MatrixView` is generic over
`origin`, and Mojo checks a method body against every instantiation including
the read-only one, so anything writing through `self.data` is rejected where it
is defined. Neither a `where Self.mut` clause nor a constrained `self` refines
it. They live in `routines/mutation.mojo` instead, as free functions pinned to
`Origin[mut=True]`, which puts the requirement in the signature: passing a
read-only view is a compile error at the call site. Single-element writes are
unaffected — `v[i, j] = x` goes through the reference `__getitem__` returns.

**Region assignment isn't `__setitem__`.** Mojo routes `a[i:j, k:l] = rhs`
through `__getitem__`, so `rhs` would have to be a view carrying the target's
own origin, which makes assigning from any other matrix inexpressible as
subscript sugar. Spelled `fill(...)` and `assign(...)`.

**Mutable views were built here and locked away in 5.2.** `view()` and slicing
took `ref self` so the caller's mutability could reach the origin; that had to
be walked back once operators existed. `view_mut` in `routines/mutation.mojo`
is now the only source of one.

Two smaller notes. The iterator is parameterised on axis and direction rather
than hardwired to forward rows, because that is the traversal
`apply_along_axis` needs in 5.3. And Mojo's builtin `reversed()` only accepts
specific stdlib containers, so it will not dispatch to `__reversed__` — use
`rows[False]()`.

### 5.2 — Operators

| Item                                                             | Module                | Status |
| ---------------------------------------------------------------- | --------------------- | ------ |
| In-place ops `+=`, `-=`, `*=`, `/=`, `//=`, `%=`                 | `types/matrix.mojo`   | ✓      |
| `__pow__`, `__floordiv__`, `__mod__`                             | `types/matrix.mojo`   | ✓      |
| Reflected ops `__radd__`, `__rsub__`, `__rmul__`                 | `types/matrix.mojo`   | ✓      |
| Comparison ops `<`, `<=`, `>`, `>=`, `==`, `!=` → `Matrix[bool]` | `routines/logic.mojo` | ✓      |

**In-place operators exist on `Matrix` only.** They write back through the
matrix's own strides instead of allocating, so a transposed or column-major
matrix keeps its layout. `MatrixView` gets no `+=`, for the reason 5.1 hit:
the body would have to type-check against the read-only instantiation too.
Mutating a view goes through `routines/mutation.mojo`.

**Comparisons return a mask, not a verdict.** `a == b` is an element-wise
`Matrix[DType.bool]`, as in NumPy, so `Matrix` deliberately does not conform to
`EqualityComparable`; whether two matrices are wholly identical stays a separate
question that `assert_matrices_equal` answers. `__pow__` is element-wise for the
same reason — matrix exponentiation will get a named routine, not an operator.
The comparison kernels went into a new `routines/logic.mojo`, which is where 5.3
wants `all` / `any` anyway. `__rtruediv__` came along uninvited: shipping
`2.0 - A` without `2.0 / A` is worse than having all four or none. All of these
are mirrored onto `MatrixView`.

**Slicing had to become read-only.** 5.1 gave slicing and `view()` `ref self`,
so `a[0:2, 0:2]` on a `var` matrix produced a mutable view. A mutable view is an
exclusive borrow and Mojo refuses to pass two of them into one call, so
`a[0:1, :] - a[1:2, :]`, `a + a[0:2, 0:2]` and `a[0:2, 0:2] @ a[0:2, 0:2]` were
all rejected. Nothing caught it because every view test until then paired views
from two *different* matrices. The first fix kept `view()` and added a mutable
`view(x, y)` beside it, which left a write door behind the most innocent call in
the API. The rule that replaced it:

> Nothing that carries a borrow in its type is ever handed out mutable, except
> through a function in `linamo.routines.mutation`.

That is checkable: `grep -rn "ref self" src/linamo/types/` returns exactly one
line, element access on `Matrix`. `view_mut`, `rows_mut` and `cols_mut` live in
the mutation module, so a caller who never imports it cannot construct a mutable
view at all. `tests/matrix_view/test_view_aliasing.mojo` keeps the blind spot
closed.

Two related fixes. `Matrix.__getitem__` used to return a reference whose origin
named one computed element, and forming a second invalidated the first, so
`a[0, 0] + a[1, 1]` did not compile; it now returns through
`origin_of(self.data)`, the whole buffer. And `__setitem__` stays absent:
merely defining it makes the compiler pass `self` to `__getitem__` as a
temporary copy in some positions, so a sliced view carries the origin of a dead
temporary and `a[0:1, :] - a[1:2, :]` breaks again. Reproduced in twenty lines
with no Linamo involved, for both `Int` and `Slice` setters.

**Four overloads per operation became one.** Each binary routine carried
`(M, M)`, `(M, V)`, `(V, M)` and `(V, V)`, three of them one-line forwarders —
57 redundant overloads once comparisons landed, with every 5.3 reduction set to
add more. `MatrixView` now has an `@implicit` constructor from `Matrix`, so a
routine declares the view × view signature only:

```mojo
@implicit
def __init__[d: DType](
    out self: MatrixView[d, ImmOrigin(origin_of(m.data))], ref m: Matrix[d]
):
```

Two details are load-bearing. `ref m` is required, because under `imm`, `read`
or the default convention `origin_of(m.data)` names the callee's own parameter
slot and no call site can satisfy the result. And `ImmOrigin(...)` is required
so a `var` matrix converts to a *read-only* view — without it `add(a, a)` is two
mutable borrows of one matrix. It also can never satisfy
`routines/mutation.mojo`'s `Origin[mut=True]`, so `fill(m, ...)` stays a compile
error.

Net effect: 57 overloads removed, `math.mojo` 1089 → 823 lines and `logic.mojo`
462 → 261, with no call site changed. Done before 5.3 so that each reduction
below is one signature instead of two. The operators themselves still carry the
redundancy — see 5.9.

> **Not the `MatrixLike` trait.** `def add[M: MatrixLike, N: MatrixLike](a, b)`
> cannot work, and not only for want of parameterised traits: the
> `M -> MatrixView` conversion must produce a type whose `origin` depends on the
> *borrow of the argument*, and no trait method can name that. `out self` can.

### 5.3 — Reductions & search

| Item                                 | Module                     | Status |
| ------------------------------------ | -------------------------- | ------ |
| `sum` / `cumsum` (axis + full)       | `routines/statistics.mojo` | ✓      |
| `prod` / `cumprod` (axis + full)     | `routines/math.mojo`       | ✓      |
| `min` / `max` (axis + full)          | `routines/math.mojo`       | ✓      |
| `argmin` / `argmax` (axis + full)    | `routines/searching.mojo`  | ✓      |
| `all` / `any`                        | `routines/logic.mojo`      | ✓      |
| `sort` / `argsort` / `sort_inplace`  | `routines/sorting.mojo`    | ✓      |
| `apply_along_axis` (generic applier) | `routines/functional.mojo` | ✓      |

The applier is two pieces. `fold` reduces a view to one scalar and carries the
three-way layout dispatch — row-contiguous, column-contiguous, strided — so no
reduction repeats it. `apply_along_axis[axis, func]` walks one axis with the 5.1
iterator and calls a per-lane kernel. Each reduction is then an operator, a lane
kernel, and two public overloads; `sum(m)` and `sum(m, axis=0)` share an
implementation rather than resembling one.

`axis` is a compile-time parameter on the applier and a runtime argument on the
public routines. The iterator is parameterised on axis, so traversal has to be
picked at build time, but `sum(m, axis=0)` is the call users expect to write;
the public routines branch onto the two instantiations, and a
`where axis == 0 or axis == 1` clause makes a third value a build error.

`axis` and the iterator index run opposite ways. `axis` follows NumPy and names
the dimension *removed*, so `axis=0` collapses rows and the traversal walks
columns. The inversion happens once, inside `apply_along_axis`. Every axis test
uses a non-square matrix, because an implementation that inverts them still
produces plausible numbers on a square one.

Operands are pinned to `Origin[mut=False]`: a lane kernel has to be specialised
to a concrete origin at the call site, and leaving `mut` free makes the function
type unnameable. It costs nothing, since nothing outside `routines.mutation`
hands out a mutable view and one demotes with `as_imm()`.

Scans and searches keep their own walks. `cumsum`/`cumprod` produce one output
per input rather than one per lane, `argmin`/`argmax` thread two accumulators
where a fold threads one, and `all`/`any` accumulate a `Bool` while the elements
are not — and short-circuit, which a fold could not.

Sorting requires an explicit `axis`. NumPy defaults `sort` to the last axis but
`sum` to a full reduction; carried into a two-dimensional library that would
make `sort(m)` read like `sum(m)` and mean something else. `sort_inplace` writes
through the matrix's own strides so a column-major matrix keeps its layout,
`sort` returns a fresh C-contiguous result, and `argsort` is stable so the two
agree element for element.

Vectorising `fold` is left to Phase 10: it needs a SIMD accumulator and a
horizontal reducer, which the scalar `func` parameter cannot express without
making the function type generic over lane count.

### 5.4 — Shape & layout manipulation

| Item                            | Module                       | Status |
| ------------------------------- | ---------------------------- | ------ |
| `reshape`                       | `routines/manipulation.mojo` | ✓      |
| `reshape_view` (zero-copy)      | `routines/manipulation.mojo` | ✓      |
| `resize`                        | `routines/manipulation.mojo` | ✓      |
| `flatten`                       | `routines/manipulation.mojo` | ✓      |
| `contiguous` / `reorder_layout` | `routines/manipulation.mojo` | ✓      |
| `broadcast_to`                  | `routines/manipulation.mojo` | ✓      |
| `astype[dtype]`                 | `routines/manipulation.mojo` | ✓      |
| `fill` (whole matrix)           | `types/matrix.mojo`          | ✓      |

**Invariant: a matrix's element buffer is fixed at construction.** `reshape`,
`resize`, `flatten` and `astype` return new matrices; nothing here grows,
shrinks or reallocates the `data` of an existing one. That is a safety rule, not
a style preference. A `MatrixView` holds a `Span` over `origin_of(m.data)`,
which captures the `List`'s heap pointer, so growing that `List` leaves every
live view dangling — and Mojo 1.0 will not catch it, because the borrow checker
enforces origins at call sites and a later `m.data.append(...)` is not one.
`local/origin_demos/` (gitignored) has a runnable demonstration.

The module splits by what it returns. `reshape`, `resize`, `flatten`,
`contiguous`, `reorder_layout` and `astype` allocate and return an owning
`Matrix`; `reshape_view` and `broadcast_to` allocate nothing and return a view
carrying the input's origin, so the source stays alive exactly as long as the
result. That second group is what the two-type split buys — NuMojo's equivalents
either copied or handed back a pointer-backed matrix whose lifetime nothing
tracked.

`resize` could not be ported as written. NuMojo's mutates and reallocates when
the shape grows, which is precisely what the invariant forbids, so it returns a
new matrix and reads `a = resize(a, m, n)` at the call site. Semantics are
otherwise unchanged: copy in C order, truncate or zero-pad.

`broadcast_to` returns a stride-0 view — a stretched dimension gets stride zero,
so every index along it lands on the same element and a `1 x n` row broadcast to
`m x n` costs nothing. It is read-only, as in NumPy, and here for a second
reason: many logical positions map onto one element, so a write would show up at
all of them. A zero stride is not contiguous by any definition, so the result
takes the strided path through the routine layer; `to_matrix()` densifies it.

Finally, `order` means *index* order, not memory layout. `reshape(a, m, n, "F")`
reads and writes column-first and still returns a C-contiguous matrix; where the
elements sit is a separate question, asked with `contiguous(a, "F")`. That is
why `contiguous` subsumes NuMojo's `reorder_layout`, which is now just the flip
and raises on a strided input because there is no layout to flip.

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

### 5.8 — Interop

| Item                             | Module                        | Status |
| -------------------------------- | ----------------------------- | ------ |
| `to_numpy` / `matrix_from_numpy` | `routines/numpy_interop.mojo` | ✓      |

> **No NuMojo bridge, and no migration guide.** NuMojo's `Matrix` is gone as of
> its Mojo 1.0.0 release, so there is no installed base to migrate and nothing
> for a guide to bridge from. Nor is there a `to_ndarray` here: reproducing it
> would make Linamo depend on NuMojo and invert the dependency direction. If an
> `NDArray` bridge is ever wanted, it belongs on NuMojo's side as
> `NDArray.from_linamo()`. Linamo is an independent library; what users need
> instead is a guide to *this* one — see [Documentation](#documentation).

### 5.9 — API consolidation & hardening

| Item                                                     | Module                       | Status |
| -------------------------------------------------------- | ---------------------------- | ------ |
| Collapse the operator overloads onto implicit conversion | `types/matrix.mojo`, `_view` | □      |
| Make the layout fields private; rename the accessors     | `types/`                     | □      |
| Assert the layout invariant in the `Matrix` constructors | `types/matrix.mojo`          | □      |
| Stop using `MatrixLike` (keep the file)                  | `traits/matrix_like.mojo`    | □      |

All four are breaking changes to spellings users would already have written, so
they land **before v0.1.0** — see [Release Plan](#release-plan--v010). The
overload collapse should come before 5.6, for the reason the 5.2 collapse came
before 5.3: every routine added meanwhile doubles the work.

**The operators never got the 5.2 treatment.** `types/matrix.mojo` still carries
both `__add__(self, other: Self)` and
`__add__[origin](self, other: MatrixView[...])`, and `types/matrix_view.mojo`
mirrors it with a `Matrix`-argument overload beside each view one. The `Self`
and `Matrix` forms are redundant: implicit conversion fires on the *argument*
(only `self` is beyond its reach), so deleting the `Self` overload leaves
`a + b` compiling and correct — checked on 2026-08-18 by doing it. Roughly 20
overloads per file, across `+ - * / // % ** @` and the six comparisons.

**Fields go private, accessors lose the `get_`.** `m.nrows` and `get_nrows()`
are two spellings of one fact, and the fields are not merely informational:
`nrows`, `ncols`, the two strides and the length of `data` are one invariant
bundle, so assigning to any of them alone produces a matrix that indexes outside
its own buffer. So `_nrows`, `_ncols`, `_row_stride`, `_col_stride`, `_offset`,
`_data`, and `get_nrows()` becomes `nrows()`. `data` goes private with the rest:
users are not expected to write `origin_of(m.data)` themselves, and if that
turns out to be wrong the type grows a public way to spell its own origin rather
than reopening the field. Mojo 1.0 has no access control, so the underscore is a
convention and the real enforcement is the assertion below.

**The constructors accept any strides at all.** `Matrix.__init__` stores
`row_stride` and `col_stride` as given, checking them against nothing. Aliasing
gets through (`row_stride=0` makes every row the same row, so `m[0, 0] = 5` also
changes `m[1, 0]`) and so does overrun (`row_stride=100` on a six-element buffer
indexes past the end — caught by `List` under `-D ASSERT=all`, undefined in
release). A zero stride is a legitimate state for a `MatrixView` now that
`broadcast_to` produces one, and never legitimate for an owning matrix. Two
`debug_assert`s per constructor — positive strides, and
`(nrows - 1) * row_stride + (ncols - 1) * col_stride < len(data)` — state that
executably and cost nothing in release.

**`MatrixLike` stops being used.** Nothing in the library is generic over it:
twelve methods, declared by three types and consumed by none, whose only real
effect is to pin the accessor spellings the item above changes. The conformances
come off `Matrix`, `MatrixView` and `StaticMatrix`, and the accessors stay as
ordinary methods. The file and the `traits/` folder stay in the tree, because
the trait is not a bad idea, only an unused one: Mojo 1.0 supports associated
aliases, so `comptime dtype: DType` plus
`def at(self, r: Int, c: Int) -> Scalar[Self.dtype]` is expressible (probed
2026-08-18) and a later version could carry the read-only algorithms — starting
with `__str__` / `write_to`, duplicated almost line for line between the two
types today. This does not reopen 5.2: operand genericity still cannot go
through a trait.

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

## Documentation

> **Status: 🚧 README and examples only**

| Item                                             | Where                    | Status |
| ------------------------------------------------ | ------------------------ | ------ |
| README: overview, quickstart, project layout     | `README.md`              | ✓      |
| Runnable examples, one per public type           | `examples/`              | ✓      |
| User guide                                       | `docs/GUIDE.md`          | □      |
| `docs/API.md` checked against the actual surface | `docs/API.md`            | 🚧      |
| Documented install path                          | `README.md`, `pixi.toml` | □      |

The guide is the release blocker of the three. Most of Linamo can be guessed at
by someone who knows NumPy — except the one thing the library is built around,
which is that `Matrix` owns and `MatrixView` borrows. A user who does not know
that will not understand why `a[0:2, 0:2]` cannot be written through, why `fill`
lives in `routines.mutation`, or why passing a `Matrix` to a routine that takes
a `MatrixView` compiles at all. So the guide leads with the two types and their
contract, then does the ordinary tour: creating matrices, indexing and slicing,
arithmetic and comparison, reductions with `axis`, shape and layout, the linear
algebra routines, NumPy interop, and how errors are raised and read.

`docs/API.md` is close but drifts. Its `MatrixView` intro still described
`view()` as taking `ref self` and yielding a writable view from a `var` matrix
— the 5.1 behaviour that 5.2 removed — while its own table three sections later
said the opposite. That paragraph is fixed; the rest of the file has not been
walked against the current surface, and 5.4's routines are not in it at all.

The install path is a documentation gap and a packaging one at once: today the
only story is `-I src`, and either the package gets published to
`modular-community` or the `.mojoc` route gets written down.

---

## Release Plan — v0.1.0

**Gate: Phase 5 through 5.6, plus 5.9, plus the user guide.**

Earlier is not shippable. Without 5.5 and 5.6 there is no `arange`, no
`linspace`, no `isclose`, no `sin`, and a new user's opening moves are "build a
test vector, transform it, check it against an expected answer" — only the
middle one works today. Neither phase is large: `fold` and `apply_along_axis`
already exist, and most of the entries are one-liners over them.

Later buys little. `eig`, norms, random generation and the Phase 10 performance
work all *add* signatures rather than change them, so they are 0.2.0 material.
5.9 is the exception that has to land first, since it changes spellings users
would already have written.

The rest of the checklist is [Documentation](#documentation): a user guide, an
install path that is not `-I src`, and `docs/API.md` checked against the actual
surface. Those are what make a release usable rather than merely tagged.

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
| 2026-08-16 | Collapsed the four operand overloads per binary routine into  |
|            | one, via an `@implicit` `Matrix` -> `MatrixView` constructor  |
|            | pinned to an immutable origin. 57 overloads removed;          |
|            | `math.mojo` 1089 -> 823 lines, `logic.mojo` 462 -> 261. No    |
|            | call site changed. 7 new tests (294 total).                   |
| 2026-08-16 | Phase 5.3 done: `fold` + `apply_along_axis` in a new          |
|            | `routines/functional.mojo`, and on top of them `sum`,         |
|            | `cumsum`, `prod`, `cumprod`, `min`, `max`, `argmin`,          |
|            | `argmax`, `all`, `any`, `sort`, `argsort`,                    |
|            | `sort_inplace`. New modules: `statistics.mojo`,               |
|            | `searching.mojo`, `sorting.mojo`. 33 new tests                |
|            | (327 total).                                                  |
| 2026-08-18 | Phase 5.4 done: new `routines/manipulation.mojo` with         |
|            | `reshape`, `reshape_view`, `resize`, `flatten`,               |
|            | `contiguous`, `reorder_layout`, `broadcast_to`,               |
|            | `astype`; whole-matrix `fill` on `Matrix`.                    |
|            | `reshape_view` and `broadcast_to` are zero-copy and           |
|            | origin-tracked; `resize` returns a new matrix rather          |
|            | than reallocating in place. 36 new tests (363 total).         |
| 2026-08-18 | Added 5.9 (API consolidation & hardening) and the             |
|            | v0.1.0 release plan; condensed the Phase 5 write-ups.         |
|            | Decided, both before v0.1.0: stop using `MatrixLike` but      |
|            | keep the file, and make the layout fields private.            |
|            | Verified first that the `Self` operator overloads are         |
|            | redundant under implicit conversion, and that Mojo 1.0        |
|            | traits do support associated aliases.                         |
|            | Dropped the NuMojo migration guide in favour of a user guide; |
|            | added a Documentation                                         |
|            | section and gated v0.1.0 on it.                               |
