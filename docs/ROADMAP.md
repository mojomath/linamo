# Roadmap <!-- omit in toc -->

Linamo development roadmap. Phases are prioritized for use as the linear
algebra foundation of [stamojo](https://github.com/mojomath/stamojo) (a
statistical modeling library, similar to statsmodels).

Last reviewed: **2026-08-22**

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
  - [Mixing `StaticMatrix` with `Matrix`](#mixing-staticmatrix-with-matrix)
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
| Operator overloads (`+`, `-`, `*`, `/`) | `types/matrix.mojo`      | Ergonomic arithmetic syntax         | ✓      |
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

> **Status: 🚧 In progress — 5.1 through 5.5, the gated half of 5.6,
> 5.8 and 5.9 done**
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
question that `assert_matrices_equal` answers. `__pow__` was element-wise for
the same reason; the 2026-08-21 entry below revisits that and gives `*` and `**`
their linear-algebra meanings. The comparison kernels went into a new
`routines/logic.mojo`, which is where 5.3 wants `all` / `any` anyway.
`__rtruediv__` came along uninvited: shipping `2.0 - A` without `2.0 / A` is
worse than having all four or none. All of these are mirrored onto `MatrixView`.

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

| Item                                                    | Module                   | Status |
| ------------------------------------------------------- | ------------------------ | ------ |
| `empty`                                                 | `routines/creation.mojo` | ✓      |
| `arange`                                                | `routines/creation.mojo` | ✓      |
| `linspace`                                              | `routines/creation.mojo` | ✓      |
| `zeros_like` / `ones_like` / `full_like` / `empty_like` | `routines/creation.mojo` | ✓      |
| `from_list` / `from_string`                             | `routines/creation.mojo` | ✓      |
| `rand` / `seed` (see also Phase 9)                      | `routines/random.mojo`   | ✓      |

**Range constructors return a `1 x n` row.** Linamo has no 1-D type, and a row
is what NumPy's 1-D `arange` prints as, so that is what `arange` and `linspace`
produce; `reshape(x, n, 1)` is the column. Both **raise on an empty result**
rather than returning a `1 x 0` matrix: `arange(5, 0)` is a mistake far more
often than a deliberate request for nothing, and a zero-column matrix cannot be
printed, indexed or multiplied by anything. `linspace` pins its last element to
`stop` exactly instead of trusting `start + (num - 1) * step`, which lands a
rounding error short.

**The `*_like` family copies the shape, not the layout.** It follows 5.4's rule
that an owning result is dense in C order, so `zeros_like` of an F-contiguous
matrix is C-contiguous. Reproducing the input's layout would also be undefined
for the case the family most needs to accept --- a strided view, which has no
layout to reproduce. `full_like` was added alongside the three in the sketch,
since its absence next to `full` would be the odd gap.

**`from_string` parses one grammar, and `from_list` is a rename.** Elements are
separated by whitespace or commas and rows by nested brackets, so
`"[[1, 2], [3, 4]]"` is 2x2 and an unnested `"1 2 3"` is one row; a second
overload takes an explicit shape and ignores the bracket structure entirely.
Tokenising is deliberately liberal --- anything that is not a separator or a
bracket accumulates into a token --- so a bad cell is reported by name
(`Cannot parse 'x' as a number`) rather than guessed at by the scanner.
`from_list` is the positional spelling of the keyword-only
`matrix(flat_list=..., nrows=..., ncols=...)` that already existed. Both carry
the underscore rather than NuMojo's `fromlist`/`fromstring`, so that every
constructor named for its source reads as one family with `from_numpy`.

**`seed` came along with `rand`.** A generator that cannot be pinned cannot be
tested, so `seed()` forwards to the standard library's global RNG that `rand`
draws from; the reproducibility test seeds twice and compares element for
element. `randn` and the rest of the distribution family stay in Phase 9.

### 5.6 — Element-wise math & logic

**In the v0.1.0 gate.** These are what a new user needs to check a result, so
without them the opening moves of a session do not close.

| Item                                                         | Module                | Status |
| ------------------------------------------------------------ | --------------------- | ------ |
| `isclose` / `allclose`                                       | `routines/logic.mojo` | ✓      |
| `logical_and` / `logical_or` / `logical_not` / `logical_xor` | `routines/logic.mojo` | ✓      |

**Deferred to 0.2.0.** Each of these only *adds* a signature, so shipping them
later breaks nothing that 0.1.0 users will have written. Decided 2026-08-19.

| Item                                                          | Module                | Status |
| ------------------------------------------------------------- | --------------------- | ------ |
| Trig: `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2`    | `routines/math.mojo`  | □      |
| Hyperbolic: `sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh` | `routines/math.mojo`  | □      |
| `round`                                                       | `routines/math.mojo`  | □      |
| `isposinf` / `isneginf`                                       | `routines/logic.mojo` | □      |

### 5.7 — Linear algebra gaps

| Item                                    | Module                 | Status |
| --------------------------------------- | ---------------------- | ------ |
| `issymmetric`                           | `routines/linalg.mojo` | □      |
| `solve_lu` (explicit LU-based solve)    | `routines/linalg.mojo` | □      |
| `eig` (ported from NuMojo — Phase 6)    | `routines/linalg.mojo` | □      |
| `lu`/`det`/`solve`/`inv` over `Numeric` | `routines/linalg.mojo` | ✓      |

### 5.8 — Interop

| Item                      | Module                        | Status |
| ------------------------- | ----------------------------- | ------ |
| `to_numpy` / `from_numpy` | `routines/numpy_interop.mojo` | ✓      |

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
| Collapse the `linalg` routine overloads                  | `routines/linalg.mojo`       | ✓      |
| Unify the `Matrix` writes onto `set`                     | `types/matrix.mojo`          | ✓      |
| Clamp negative view extents from backwards slices        | `types/matrix_view.mojo`     | ✓      |
| Promote `view_mut` to a `Matrix` method                  | `types/matrix.mojo`          | ✓      |
| Make the whole-matrix scalar write non-raising           | `types/matrix.mojo`          | ✓      |
| Give `fill` and `assign` whole-view forms                | `routines/mutation.mojo`     | ✓      |
| Move the dtype aliases out of the prelude                | `__init__.mojo`, `prelude`   | ✓      |
| Collapse the operator overloads onto implicit conversion | `types/matrix.mojo`, `_view` | ✓      |
| Make the layout fields private; rename the accessors     | `types/`                     | ✓      |
| Assert the layout invariant in the `Matrix` constructors | `types/matrix.mojo`          | ✓      |
| Stop using `MatrixLike` (keep the file)                  | `traits/matrix_like.mojo`    | ✓      |

**5.9 is complete.** Its breaking items were the reason it had to precede
v0.1.0 — see [Release Plan](#release-plan--v010). The overload collapse went
first, for the reason the 5.2 collapse came before 5.3: every operator added
meanwhile would have doubled the work.

**The routine layer collapsed first.** 5.2 did
`math.mojo` and `logic.mojo`, and 5.4's `manipulation.mojo` was written
view-only from the start, which left `linalg.mojo` as the last module carrying
`Matrix`-argument forwarders --- 13 of them, including the three-way
`(M, M)` / `(M, V)` / `(V, M)` sets on `solve` and `lstsq`. Verified redundant
by deleting them and compiling a caller that passes a `Matrix` to every
routine, then confirmed against the full suite: 121 lines gone, `linalg.mojo`
594 -> 472, no call site changed.

**Every write on a `Matrix` is now `set`.** `fill(value)`, `fill(rows, cols,
value)` and `assign(rows, cols, src)` became five `set` overloads that dispatch
on their arguments: a `Self.ElementType` fills, a `Matrix` or `MatrixView`
copies, and `set(row, col, value)` writes one element. Dispatch is unambiguous
because nothing in the library converts a scalar to a matrix, and a `Matrix`
source needs no `.view()` thanks to the implicit conversion. The free `assign`
overload taking a `Matrix` source went the same way as the `linalg` forwarders.

`set` **delegates to `routines/mutation.mojo`** rather than looping over
`self.data`, and the reason is not tidiness. The duplicated loops had already
drifted: both were written in `e41b9e4`, and by 2026-08-18 the method used
`Slice.indices()` plus `range()` and agreed with Python on `m[3:1, :]`, while
the `MatrixView` slicing constructor computed `ceildiv(end - start, step)` and
produced `nrows = -2`. A view with a negative row count, reachable from plain
public slicing, reporting `len(v) == -2`. Both extents are now clamped with
`max(0, ...)`, which leaves genuine reversing slices alone
(`4:0:-1` is `ceildiv(-4, -1) == 4`), and the loop exists once.

None of the three excuses for the duplication survived checking. There is no
circular-import wall — `types/matrix.mojo` imports `routines/mutation.mojo` at
the top level and compiles, and it already imported `routines.math`,
`routines.logic` and `routines.manipulation`. There is no cost —
`MatrixView.__getitem__(i, j)` is an unchecked `offset + i*rs + j*cs`, the same
arithmetic `get_offset` does. And it was not an evolution: `git log -S` puts
both copies in one commit.

`view_mut` stays a free function and does **not** become a method. It is
technically possible — a `mut self` method returning
`MatrixView[dtype, origin_of(self.data)]` compiles — but it would dissolve the
one invariant this design is built on: a caller who never imports
`routines.mutation` cannot construct a mutable view. `set` is different in kind:
it creates a mutable view internally and consumes it before returning, so it
hands nothing out.

**`view_mut` became a method on `Matrix` after the import barrier was challenged
and failed to justify itself. The original argument was that confining mutable
views to `routines.mutation` meant a caller could not construct one without
opting in. That is true but not useful: the thing it protects against is
*accidental* write access, and nobody types `view_mut` by accident. What the
barrier actually cost was an import line on every loop that writes.

Testing settled the only part that was not taste. A method takes `ref self`,
which raised the question of whether it could bind a temporary and hand back a
mutable view onto dead storage - the failure mode that ruled out `__setitem__`.
It cannot. `la.matrix(...).view_mut(...)` binds the temporary immutably, so the
result is read-only and `fill` rejects it at compile time, identically to the
free function. The same holds for a borrowed `Matrix` parameter. Adding the
method also left `a[0:1, :] - a[1:2, :]` compiling, confirming this is unrelated
to the `__setitem__` hazard, which was specific to how that operator perturbs
`__getitem__` resolution.

So the invariant was restated rather than abandoned. It was never really about
the module: mutability is only ever inherited from a `var`, never manufactured,
and every spelling that produces a mutable view carries `_mut` in its name -
`grep -rn "def .*_mut" src/` lists all four. The name is the barrier, and it
is the one that was doing the work all along. The free function stays for views
of views, where there is no owner to call a method on.

`set(value)` is the one write that does not delegate, and it is now
non-raising. A `Matrix` owns exactly `nrows * ncols` elements at offset zero,
so every slot in `data` belongs to it under either layout: the whole-matrix
scalar write is a flat walk of the buffer with no index arithmetic and no
bounds to check. Routing it through the region path had made it `raises` for no
reason, and cost the stride arithmetic too. This is not the duplication that
5.9 removed - it is a different and strictly better algorithm for the total
case, and it is total, which is why it can drop `raises` at all. The other four
overloads keep it: they can fail on a shape mismatch or an out-of-range index,
and that is real.

`fill` and `assign` gained whole-view forms for the same reason. The writing
table in the manual had `m.set(value)` on the left against
`fill(v, rows, cols, value)` on the right, which meant "write the whole thing"
and "fill a region" were the same call spelled once - the slices were pure
ceremony whenever they covered the view. `fill(v, value)` and `assign(v, src)`
now mirror the `Matrix` side one for one, and `fill(v, value)` is non-raising
on the same argument as `set(value)`: it visits every index of the view, so
none of them can be out of range. `assign(v, src)` keeps `raises`, because a
shape mismatch is real.

**The element-type aliases moved out of the prelude.** `float64`, `int32` and
the rest were defined in `prelude.mojo`, so `from linamo.prelude import *` put
twelve bare names into the caller's global scope - including `int` and `uint`,
which no library has any business claiming unqualified, and which shadow
silently rather than erroring. They now live in `__init__.mojo` and are reached
as `la.float64`, qualified like every other name the library exports. Each type
also gained a short form, so `la.f64`, `la.i32` and `la.u8` are the same
aliases under Rust-style names; they cost nothing now that they are namespaced,
and the two spellings suit the two audiences the library has. The prelude is
down to `import linamo as la`, and stays that way as a placeholder (decided
2026-08-19). It is the right home for whatever a future version decides every
user should have unqualified, and keeping the file means that decision will
not also have to re-add a public module.

A related question, and the answer is no: `view_mut(v, x, y)` stays. The worry
is that a mutable sub-view of a mutable view puts two writable views on one
matrix at once. It does - but so do `var a = m.view_mut(...)` and
`var b = m.view_mut(...)`, and so does `var b = a`, since `MatrixView` is an
`ImplicitlyCopyable` handle. Removing the sub-view form would leave both of
those doors open while breaking `fill` and `assign`, which are built on it, and
any blocked algorithm that writes into a quadrant of a view it was handed.
What actually guards against aliasing is Mojo, one level down: two mutable
views of the same matrix in a single expression is a compile error
(`aliasing values passed mutably to 'self' ... and to 'other'`), which is the
case that could produce a wrong answer. Holding two handles and writing through
them in sequence is well-defined.

**The operators have now had the 5.2 treatment.** `types/matrix.mojo` carried
both `__add__(self, other: Self)` and
`__add__[origin](self, other: MatrixView[...])`, and `types/matrix_view.mojo`
mirrored it with a `Matrix`-argument overload beside each view one. Both
redundant sets are gone: 32 overloads deleted — 19 from `matrix.mojo`, 13 from
`matrix_view.mojo` — across `+ - * / // % ** @`, the six in-place operators and
the six comparisons. 205 lines removed against 56 added.

The load-bearing fact had to be verified rather than assumed: implicit
conversion fires on the *argument*, and it fires **under operator sugar**, not
only in direct calls. With `__add__(self, other: Self)` deleted, `A + B`
resolves to the view-operand overload. Confirmed for all four operand
permutations of every operator, plus the scalar and reflected forms, before
touching the rest.

Alone among the 5.9 items this one is **not** a breaking change: every spelling
that compiled before still compiles, and `A += A` is still rejected by the
borrow checker with a byte-identical diagnostic. What stays is what the
conversion cannot reach — `self` is fixed by the receiver, so both types keep
their own dunders, and the scalar overloads stay because nothing converts a
scalar to a matrix.

**The fields are private and the accessors have lost the `get_`.** `m.nrows`
and `get_nrows()` were two spellings of one fact, and the fields are not merely
informational: `nrows`, `ncols`, the two strides and the length of `data` are
one invariant bundle, so assigning to any of them alone produces a matrix that
indexes outside its own buffer. They are now `_nrows`, `_ncols`,
`_row_stride`, `_col_stride`, `_offset`, `_data`, and `get_nrows()` is
`nrows()`. Mojo 1.0 has no access control, so the underscore is a convention
and the real enforcement is the constructor assertion above.

Checked first, because it decided the shape of the change: **Mojo 1.0 has no
`@property`** (`@property` is not a known declaration). So `m.nrows` is either
a public field or a method call; there is no third spelling that keeps the
field private and the parentheses off. The parentheses are the price of the
underscore.

Routines call the accessor rather than reaching for `_nrows`, so the underscore
means "the type itself, nobody else" instead of merely "not for users". That
costs nothing: every accessor is `@always_inline`, and a kernel that reads
`nrows()`, `ncols()`, both strides and `offset()` in a loop compiles to
assembly containing **no call instructions at all** — the accessors and the
enclosing function are inlined into `main` and no accessor symbol survives.
Verified with `mojo build --emit asm`.

`data` is the exception at both ends. Internally it stays a raw field: kernels
in `linalg.mojo` and `statistics.mojo` index and write the buffer directly, and
`numpy_interop.mojo` needs `mat._data._data` for the memcpy. Externally it
keeps a public `data()` returning a `Span`, which is what the examples use to
show a C- and an F-ordered matrix having different buffers. The one thing
`data()` cannot do is name an origin: `origin_of(a.data())` is the origin of a
temporary. The public spelling for that turns out to already exist and to be
better than the old one — `type_of(a.view()).origin` says "the origin a view of
`a` carries" without mentioning a buffer at all.

Three findings the plan did not anticipate:

- **`StaticMatrix` cannot follow the same naming, and does not need to.** Its
  `nrows` and `ncols` are struct *parameters* and its strides are `comptime`
  aliases, so `def nrows(self)` is an `invalid redefinition`. It needs no
  accessor either: `m.nrows` already reads on an instance as a compile-time
  constant that nothing can assign to. Its four shape accessors were deleted
  rather than renamed, leaving only `data()`, `offset()` and `size()`.
- **The `data()` accessor collides with the constructors' `data` argument.**
  Inside the body, the bare name resolves to the method. The argument is now
  `buffer`, which is also the more accurate word for it: flat storage, not the
  matrix's view of it.
- **`MatrixAxisIter` carries the same bundle** and was privatised with it.
  Nothing outside its own file ever read those fields, so this half of the
  change has no API surface at all.

**The constructors now assert their layout.** `Matrix.__init__` used to store
`row_stride` and `col_stride` as given, checking them against nothing. Both
failure modes were reproduced before the fix: aliasing (`row_stride=0` makes
every row the same row, so `m[0, 0] = 99` also wrote `m[1, 0]` and `m[2, 0]`)
and overrun (`row_stride=100` on a six-element buffer reached offset 201 and
returned garbage without crashing).

Two predicates in `utils/indexing.mojo` state the invariant, and the two
stride-taking constructors `debug_assert` both. `layout_fits_buffer` is the
bound the plan called for. `layout_is_dense` came out **stronger** than the
planned positivity test, and the strengthening was free: positivity alone
still admits `(1, 1)` on a 2x2, where `[0, 1]` and `[1, 0]` both land on
offset 1. Requiring C-major (`row_stride == ncols * col_stride`) or F-major
(`col_stride == nrows * row_stride`) rules that out, and the whole suite plus
every example passes under `-D ASSERT=all` with it in place — which is the
evidence that the library builds nothing else. If a padded owning matrix is
ever wanted, this assertion is the thing that will have to be relaxed
deliberately rather than discovered by accident.

The checks are independent, and each catches what the other misses: `(2, 1)`
on a 3x2 is dense and overruns a four-element buffer; `(0, 1)` fits a
six-element buffer comfortably and aliases. Both cases are covered in
`tests/matrix/test_matrix_layout_invariant.mojo` (15 tests), which tests the
predicates rather than the assertions, since `debug_assert` aborts and cannot
be caught by `assert_raises`. The copying and moving constructors take their
layout from a matrix that already satisfies the invariant, so they assert
nothing.

**`MatrixLike` is no longer used.** Nothing in the library was generic over it:
twelve methods, declared by three types and consumed by none, whose only real
effect is to pin the accessor spellings the item above changes. The conformances
came off `Matrix`, `MatrixView` and `StaticMatrix` and the accessors stayed as
ordinary methods; no call site moved and the suite was green on the first run.
The file and the `traits/` folder stay in the tree, because
the trait is not a bad idea, only an unused one: Mojo 1.0 supports associated
aliases, so `comptime dtype: DType` plus
`def at(self, r: Int, c: Int) -> Scalar[Self.dtype]` is expressible (probed
2026-08-18) and a later version could carry the read-only algorithms — the
per-type loop that reads elements into cells for `utils/formatting.mojo` is
the one that remains, printing itself having been shared out on 2026-08-21.
This does not reopen 5.2: operand genericity still cannot go through a trait.

Keeping an unimported module means keeping a module the compiler never looks
at, so `tests/traits/test_matrix_like.mojo` imports it and nothing else. The
import is the test.

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

> **Status: 🚧 `rand` and `seed` landed with 5.5; `randn` remains**
>
> *stamojo dependency: needed for simulation, bootstrap, MCMC.*

| Item                             | Module                 | stamojo use                 | Status |
| -------------------------------- | ---------------------- | --------------------------- | ------ |
| `rand()` — uniform random matrix | `routines/random.mojo` | Monte Carlo simulation      | ✓      |
| `randn()` — normal random matrix | `routines/random.mojo` | Error simulation, bootstrap | □      |
| `seed()` — set RNG seed          | `routines/random.mojo` | Reproducibility             | ✓      |

---

## Phase 10 — Performance & Polish

> **Status: □ Not started**

| Item                                   | Module                 | Notes                      |
| -------------------------------------- | ---------------------- | -------------------------- |
| Optimized matmul for all layout combos | `routines/math.mojo`   | See [MANUAL.md](MANUAL.md) |
| (C@C, F@F, C@F, F@C, V@*)              |                        |                            |
| Tiled / blocked matmul                 | `routines/math.mojo`   | Cache-friendly             |
| SIMD-optimized decompositions          | `routines/linalg.mojo` | Performance                |
| Parallel row/col operations            | `routines/math.mojo`   | Multi-core utilization     |
| Comprehensive benchmarks               | `benches/`             | Compare vs. NumPy/LAPACK   |

---

## Documentation

> **Status: ✓ Done**

| Item                                         | Where                    | Status |
| -------------------------------------------- | ------------------------ | ------ |
| README: overview, quickstart, project layout | `README.md`              | ✓      |
| Runnable examples, one per public type       | `examples/`              | ✓      |
| User manual                                  | `docs/MANUAL.md`         | ✓      |
| Manual checked against the actual surface    | `docs/MANUAL.md`         | ✓      |
| Documented install path                      | `README.md`, `docs/`     | ✓      |

The manual was the release blocker. Most of Linamo can be guessed at by someone
who knows NumPy — except the one thing the library is built around, which is
that `Matrix` owns and `MatrixView` borrows. A user who does not know that will
not understand why `a[0:2, 0:2]` cannot be written through, why `fill` lives in
`routines.mutation`, or why passing a `Matrix` to a routine that takes a
`MatrixView` compiles at all. So the manual leads with the two types and their
contract, then does the ordinary tour: creating matrices, indexing and slicing,
arithmetic and comparison, mutation, iteration, reductions with `axis`, shape
and layout, the linear algebra routines, SIMD access, NumPy interop, and how
errors are raised and read.

The user guide and the API document are one file rather than two. `docs/API.md`
was never a symbol reference — it was a prose account of the design, and its
mutability, conversion and copying tables are the most useful pages in the repo
for a user. Folding the guide into it, rather than writing a second document
that would repeat those tables and drift from them, keeps one place to update.
The design rationale that a *user* does not need — the three-layer collapse,
the kernel inventory, the `@implicit` constructor — moved to an appendix. The
per-symbol reference is generated from docstrings with `mojo doc`, which is
verified to work on the whole package, so it was never this file's job.

The manual is written and every snippet has been compiled and run against the
current source. It covers the gated half of 5.6 in a *Comparing matrices*
chapter --- exact comparison, `isclose`/`allclose` with the asymmetry of their
formula and the floating-point-only requirement, the `logical_*` connectives,
and how a mask collapses. Appendix B now lists only what 0.2.0 will add, each
entry a signature that changes nothing already written.

The install path is documented as `pixi add linamo` from `modular-community`,
with the source route beside it for the window before the v0.1.0 tag reaches
the channel. Both are written down rather than one: `-I src` recompiles the
library on every build, so a project that imports it repeatedly wants the
`.mojoc` that `pixi run pack` produces, and that route was verified by running
a program against `tests/linamo.mojoc` with no `src` on the path at all. What
remains is the publish itself, which is a `recipe.yaml` on the
`modular-community` side, not a change in this repository.

---

## Release Plan — v0.1.0

**Gate: Phase 5 through 5.5, the gated half of 5.6, plus 5.9, plus the user
guide.**

**The gate is met.** Every item above is ✓, the suite is green under
`-D ASSERT=all`, `pixi run examples` runs clean and `pixi run pack` builds
without a warning. One step is left and it is not a code change: publishing the
package to `modular-community`, which the README and the manual already
document as the install path.

Earlier is not shippable. Without 5.5 and the comparison half of 5.6 there is
no `arange`, no `linspace` and no `isclose`, and a new user's opening moves are
"build a test vector, transform it, check it against an expected answer" — only
the middle one works today. Neither is large: `fold` and `apply_along_axis`
already exist, and most of the entries are one-liners over them.

5.6 is **split** rather than taken whole. `isclose`/`allclose` and the
`logical_*` family are how a user confirms a result, so their absence blocks
the first thing anyone does. Trig, hyperbolic, `round` and the infinity
predicates only add signatures; deferring them to 0.2.0 breaks nothing written
against 0.1.0 and takes real work off the gate.

Later buys little. `eig`, norms, random generation and the Phase 10 performance
work all *add* signatures rather than change them, so they are 0.2.0 material.
5.9 is the exception that has to land first, since it changes spellings users
would already have written.

The rest of the checklist was [Documentation](#documentation): the user manual
kept level with 5.5–5.7, and an install path that is not `-I src`. Those are
what make a release usable rather than merely tagged, and both are now in
place.

### Mixing `StaticMatrix` with `Matrix`

`StaticMatrix` shares no operator with `Matrix` or `MatrixView`, so `m + s`
does not compile. The bridge is one method, `s.to_matrix()`, which copies a
static matrix into a newly allocated dense one; from there everything in the
library applies. The name is not new: `MatrixView.to_matrix()` already does the
same job for the other non-owning type, walking a source whose layout is not
dense and producing owned C-contiguous storage. Here what is walked is the
power-of-two padding, which does not survive the copy. Seven tests in
`tests/static_matrix/test_static_matrix.mojo` cover it.

An `@implicit` conversion was considered and rejected. Mojo applies a single
implicit conversion and never chains two, and every operator and routine here
takes a `MatrixView`, so a conversion landing on `Matrix` stops one step short
of all of them. Reaching the operators would instead mean letting a
wrong-shaped `StaticMatrix` convert and fall through to the dynamic kernel,
which turns a compile-time shape error into a runtime `ValueError` --- the
compile-time shape check is what `StaticMatrix` is for. Writing the hop at the
call site keeps it.

---

## Review Log

| Date       | Notes                                                         |
| ---------- | ------------------------------------------------------------- |
| 2025-02-18 | Initial roadmap created. Phase 0 complete.                    |
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
| 2026-08-18 | Phase 5.5 done: `empty`, `arange`, `linspace`,                |
|            | `zeros_like`/`ones_like`/`full_like`/`empty_like`,            |
|            | `from_list`, `from_string`, and a new `routines/random.mojo`  |
|            | with `rand` and `seed`. Range constructors return a `1 x n`   |
|            | row and raise rather than yield an empty matrix. 45 new       |
|            | tests (408 total).                                            |
|            | Also removed the 13 `Matrix`-argument forwarders from         |
|            | `linalg.mojo` (121 lines), the last module the 5.2 overload   |
|            | collapse had not reached.                                     |
| 2026-08-18 | 5.9: unified every `Matrix` write onto `set`, five overloads  |
|            | dispatching on their arguments, replacing `fill` and          |
|            | `assign` as methods; removed the free `assign` overload       |
|            | taking a `Matrix` source. `set` delegates to                  |
|            | `routines/mutation.mojo` rather than repeating the loop.      |
|            | Found while checking the two copies against each other: the   |
|            | `MatrixView` slicing constructor computed a **negative**      |
|            | extent for a backwards slice, so `m[3:1, :]` was a `-2 x 6`   |
|            | view with `len(v) == -2`. Both extents clamped to 0.          |
|            | 18 new tests (426 total).                                     |
| 2026-08-18 | Renamed `docs/API.md` to `docs/MANUAL.md` and rewrote it as   |
|            | the user manual, rather than shipping a separate `GUIDE.md`   |
|            | that would duplicate its tables. The internals moved to       |
|            | Appendix A; Appendix B lists what is not documented yet.      |
|            | Every snippet compiled and run. The per-symbol reference is   |
|            | `mojo doc` output, not a hand-written file.                   |
| 2026-08-18 | 5.9: collapsed the operator overloads onto the implicit       |
|            | `Matrix` -> `MatrixView` conversion. 32 dunders deleted       |
|            | (19 in `matrix.mojo`, 13 in `matrix_view.mojo`); 205 lines    |
|            | removed, 56 added. Verified first that the conversion fires   |
|            | under operator sugar, then across all four operand            |
|            | permutations of every operator. Not a breaking change: no     |
|            | call site moved, 435 tests unchanged, and `A += A` still      |
|            | fails with a byte-identical borrow-checker diagnostic.        |
| 2026-08-19 | 5.9 complete. Asserted the `Matrix` layout invariant in the   |
|            | constructors (`layout_is_dense` came out stronger than        |
|            | planned: C-/F-major, not just positive strides, and the whole |
|            | suite passes under `-D ASSERT=all` with it). Dropped the      |
|            | `MatrixLike` conformances. Made the layout fields private and |
|            | renamed `get_nrows()` to `nrows()` across `types/`, all       |
|            | routines, tests and examples. Mojo 1.0 has no `@property`, so |
|            | the parentheses are unavoidable; `@always_inline` makes the   |
|            | layer free, checked in emitted assembly. `StaticMatrix` keeps |
|            | bare `m.nrows` --- parameters, not fields. 457 tests.         |
| 2026-08-19 | Added `StaticMatrix.to_matrix()`, the explicit bridge to      |
|            | `Matrix`, named to match `MatrixView.to_matrix()`. Rejected   |
|            | an `@implicit` conversion first: Mojo applies one implicit    |
|            | conversion and never chains two, so one landing on `Matrix`   |
|            | reaches none of the library's `MatrixView` signatures, and    |
|            | one reaching the operators would let a wrong-shaped           |
|            | `StaticMatrix` through to a dynamic kernel, downgrading a     |
|            | compile-time shape error to a runtime `ValueError`. Explicit  |
|            | keeps the check. 7 tests, 464 total.                          |
| 2026-08-19 | Renamed `matrix_from_numpy` to `from_numpy`, `fromlist` to    |
|            | `from_list` and `fromstring` to `from_string`, all exported   |
|            | from `linamo`. The `matrix_` prefix was the only type tag on  |
|            | a routine returning a `Matrix` --- `zeros`, `arange` and the  |
|            | rest are bare, and only the `StaticMatrix` variants are       |
|            | marked (`smatrix`) --- and it broke the pair with             |
|            | `to_numpy`. The underscore replaces NuMojo's spelling so the  |
|            | three read as one family. No alias kept: pre-1.0, and every   |
|            | call site is in-tree.                                         |
| 2026-08-20 | The matrix types now take an element **type** rather than a   |
|            | `DType`: `Matrix[Float64]` beside `List[Float64]`, and        |
|            | `Matrix[BigInt]` becomes expressible at all. `Float64` *is*   |
|            | `Scalar[DType.float64]`, so nothing about the scalar case is  |
|            | lost --- routines match `Matrix[Scalar[d]]` and `d` is        |
|            | inferred --- while the two-parameter alternative would have   |
|            | carried a vestigial dtype (`Matrix[DType.float64, BigInt]`)   |
|            | that lies in every introspection path. Methods needing the    |
|            | dtype are gated with `where Self.T == Scalar[d]`; because a   |
|            | `where` clause decides availability without refining `T` in   |
|            | the body, each type has one `_simd_view`/`_as_simd` bridge    |
|            | and no `rebind` anywhere else. `la.f64` and friends now name  |
|            | types, and `la.bool_` is the mask element the stdlib has no   |
|            | name for. `StaticMatrix[Float64, 2, 3]` is spelled the same   |
|            | way but still accepts scalars only: its buffer is one SIMD    |
|            | register, so `utils/element.dtype_of` recovers the dtype at   |
|            | compile time and rejects anything else. 464 tests, unchanged. |
| 2026-08-20 | `Matrix[BigInt]` works. Decimo gained a `Numeric` trait ---   |
|            | `zero`, `one`, `+`, `-`, `*`, `/`, unary `-` --- which        |
|            | `BigInt`, `BigDecimal` and `Decimal128` declare, plus         |
|            | `BigInt.__truediv__`, which truncates toward zero exactly as  |
|            | Mojo's own `Int` does (`-7 / 2 == -3`, `-7 // 2 == -4`), so   |
|            | integer division is closed and one trait covers all three.    |
|            | The arithmetic over it lives in `linamo.decimo`, because      |
|            | every kernel in `routines/` is written against `Scalar[d]`    |
|            | and an arbitrary-precision element has no vector instruction  |
|            | to lower to. Everything structural stayed in core Linamo and  |
|            | is now generic: slicing, `transpose`, `reshape`, `flatten`,   |
|            | `contiguous`, `reorder_layout`, `broadcast_to` need only to   |
|            | move elements, and `sort`, `argsort`, `argmin`, `argmax` ride |
|            | the *stdlib* `Comparable`. Generalising them surfaced a real  |
|            | bug class: routines allocated `unsafe_uninit_length` and      |
|            | wrote at computed offsets, which for a heap-owning element    |
|            | runs a destructor over uninitialised memory. Those buffers    |
|            | are now filled front to back. `tools/ensure_decimo.sh`        |
|            | resolves the dependency (local path, conda, or a pinned       |
|            | commit) and must precompile it with this workspace's own      |
|            | `mojo`: a `.mojoc` from another toolchain crashes the         |
|            | compiler rather than failing to load. 20 new tests, 484       |
|            | total, plus `examples/decimo_examples.mojo`.                  |
| 2026-08-20 | `StaticMatrix` shape parameters renamed `nrows`/`ncols` ->    |
|            | `num_rows`/`num_cols`, and the stride aliases                 |
|            | `row_stride`/`col_stride` -> `ROW_STRIDE`/`COL_STRIDE`, to    |
|            | free the names for `nrows()`, `ncols()`, `row_stride()` and   |
|            | `col_stride()` methods: a Mojo parameter and a method cannot  |
|            | share a name. All three matrix types now spell their shape    |
|            | and stride queries alike, which is what `MatrixLike` would    |
|            | need. Verified that `StaticMatrix` conforms to that trait as  |
|            | written and that `Matrix` and `MatrixView` do not, blocked    |
|            | solely by the trait's `__str__`, which on those two carries   |
|            | `where conforms_to(Self.T, Writable)`. Conformance is still   |
|            | not declared.                                                 |
| 2026-08-20 | `tools/ensure_decimo.sh` pins the upstream commit carrying    |
|            | `decimo.Numeric`, so the fallback resolves with no local      |
|            | checkout, and the clone is blobless. CI provisions decimo     |
|            | before the suite and now also runs `examples/run_all.sh`:     |
|            | without the include path, `tests/decimo` and the decimo       |
|            | example fail to compile rather than skip.                     |
| 2026-08-20 | `linamo.decimo` is gone, folded into `linamo.routines`.       |
|            | Decimo is a hard dependency now, not an optional corner: it   |
|            | always was one in practice --- `mojo precompile src/linamo`   |
|            | never resolved without it --- and the submodule bought a      |
|            | separation that only source-mode builds could use, at the     |
|            | price of a second namespace and function-call syntax for      |
|            | arithmetic. `Matrix` and `MatrixView` carry the operators for |
|            | both element families on one method name, `Self.T ==          |
|            | Scalar[d]` selecting the SIMD kernels and                     |
|            | `conforms_to(Self.T, Numeric)` the loops; the clauses are     |
|            | disjoint, so `a + b`, `a @ b` and `la.eye[BInt](3)` read the  |
|            | same whatever the element. Mojo 1.0 has no `extension`, so    |
|            | this had to go on the structs themselves --- there was no     |
|            | third option. Decimo's three `Numeric` types are re-exported  |
|            | from `la`; `BigUInt`, `Rational` and `BigFloat` are not,      |
|            | since a matrix of them would have no arithmetic. Also:        |
|            | `transpose()` as a method (no `.T` --- a parameter and a      |
|            | method cannot share a name), unary `-` on both types for      |
|            | scalars too, `routines.mutation` re-exported from `la`, and   |
|            | the lowercase element aliases dropped now that `Float64`      |
|            | names the element type itself --- `bool_` stays, having       |
|            | nothing in the stdlib to name it. `tools/ensure_decimo.sh`    |
|            | built its conda probe inside `temp/`, where Mojo found the    |
|            | fallback `decimo.mojoc` next to it and deleted it as a stale  |
|            | shadow; the probe now builds in a scratch directory and the   |
|            | clone lives outside the include path. 523 tests.              |
| 2026-08-21 | Everything the compiler can deduce now sits behind a `//` and |
|            | is unwritable: 209 signatures across the whole library, so a  |
|            | bracket list means one thing everywhere --- these are the     |
|            | decisions Linamo cannot make for you.                         |
|            | `la.sum[DType.float64](m)` is now *unexpected parameter*      |
|            | rather than a second way to spell `la.sum(m)`. The marker     |
|            | also renumbers the slots, which is what makes `fold[my_op]`   |
|            | and `apply_along_axis[0, lane]` positional; `func` and `axis` |
|            | moved to the back of those two signatures and of the eight    |
|            | element-wise helpers to allow it, invisibly, since every call |
|            | site already named them by keyword. Left writable: the        |
|            | element type of a matrix conjured from nothing (`zeros[T]`),  |
|            | a SIMD width, an axis, a kernel passed as a `func=` value,    |
|            | and `_simd_view[d]` / `_as_simd[d]`, whose `d` comes from a   |
|            | `where` clause rather than an argument and so cannot be       |
|            | inferred at a call site inside a generic method --- `Self.T`  |
|            | is not concrete there yet. Also: `from_string` now takes      |
|            | arbitrary-precision elements. Building one by hand always     |
|            | worked --- `matrix[Dec128]([[Dec128("0.1")]])` --- but a list |
|            | of `Float64` cannot be handed over: `Dec128` and `BDec` have  |
|            | no implicit constructor from one, since routing a literal     |
|            | through it would round to a binary float first. That needed a |
|            | capability the `Numeric` trait does not carry, so decimo      |
|            | gained `Parsable` (one static `from_string`) beside it, and   |
|            | `numeric.mojo` became `traits.mojo` now that it holds two.    |
|            | The bracket walk in `_tokenize_rows` is shared: the element   |
|            | type enters only through a `parse` parameter. 528 tests.      |
| 2026-08-21 | Matrix printing reworked and shared. `Matrix`, `MatrixView`   |
|            | and `StaticMatrix` had three copies of the same tab-separated |
|            | loop; they now print through `utils/formatting.mojo` and      |
|            | differ in their header line alone --- the duplication Phase 5 |
|            | flagged, removed without the trait it was waiting on. Cells   |
|            | are padded either side of the decimal point so the points of  |
|            | a column stand in one line, per-column, the way NumPy does    |
|            | it: flush-left and flush-right both line up an edge of a      |
|            | number rather than its scale. The header dropped its strides  |
|            | and offset unless they are not the ones a fresh matrix has,   |
|            | so a view of part of a matrix announces its layout and        |
|            | nothing else does. `write_to` no longer ends in a newline,    |
|            | which was printing a blank line under every matrix. Rows      |
|            | elide by element count and columns by line width --- a count  |
|            | cannot protect a terminal when one element can be wider than  |
|            | a line --- under a `MIN_COLS_SHOWN` floor, since one column   |
|            | says less about a matrix than an overlong line does. A long   |
|            | fractional part is trimmed and marked with `…` rather than    |
|            | rounded, and the integer part is never touched, so a printed  |
|            | magnitude is always the magnitude held. Widths are counted in |
|            | code points, not bytes: `…` is three bytes and would have     |
|            | skewed every column holding a trimmed value. The nine         |
|            | appearance knobs are comptime aliases in one place, which is  |
|            | what a configuration type would carry when Mojo has global    |
|            | variables. 13 new tests (541 total), asserting on whole lines |
|            | --- a substring check cannot see a column that slipped by one |
|            | space.                                                        |
| 2026-08-21 | `*` and `**` now carry their linear-algebra meanings. `A * B` |
|            | is the matrix product, the same call as `A @ B`; `A ** n` is  |
|            | repeated multiplication by squaring, `A ** 0` the identity, a |
|            | negative power inverting first. The element-wise three became |
|            | methods beside the routines they already had --- `a.mul(b)`,  |
|            | `a.div(b)`, `a.pow(b)`. `A / B` is gone rather than           |
|            | reinterpreted: division by a matrix is a solve, and which     |
|            | side is being solved on is not a thing one character should   |
|            | decide. Five of the nine matmul-`*` libraries surveyed leave  |
|            | it undefined for that reason; only the MATLAB lineage makes   |
|            | it a solve, and that lineage has `\` for the other side. `*=` |
|            | and `/=` keep the scalar operand and lose the matrix one,     |
|            | because the matrix product cannot be written in place ---     |
|            | every element of the target is read after the point it would  |
|            | have been overwritten. The risk here is unlike the rest of    |
|            | the phase: on square operands both readings of `*` type-check |
|            | and return the right shape, so the old meaning fails silently |
|            | rather than loudly. One test in the suite was asserting the   |
|            | Hadamard product and had to be caught by running it, not by   |
|            | compiling it; `test_mul_respects_the_order_of_the_operands`   |
|            | is there so the next one is caught by name. `matrix_power` is |
|            | new in `routines/linalg.mojo`, with negative exponents on the |
|            | SIMD path alone --- `inv` has no `Numeric` counterpart to     |
|            | invert with. 549 tests.                                       |
| 2026-08-21 | `lu`, `det`, `solve` and `inv` gained `Numeric` overloads, so |
|            | a `Matrix[Dec128]` or `Matrix[BDec]` decomposes and inverts   |
|            | through the same names a `Matrix[Float64]` uses, and          |
|            | `A ** -1` works there --- the gap the previous entry left     |
|            | open. The bound is `Numeric & Comparable`, because partial    |
|            | pivoting has to rank candidates by magnitude; an `abs`        |
|            | requirement on top of that would be redundant, since          |
|            | `-x if x < zero else x` is already available. No trait was    |
|            | added to keep `BigInt` out. Its `/` truncates toward zero and |
|            | an integer matrix has no integer inverse in general, so       |
|            | elimination over it answers nothing --- but a marker trait    |
|            | would not have stopped anyone either, since any type at all   |
|            | can be a matrix element. The caveat lives in the docs and in  |
|            | two tests that pin both sides of it: `[[1, 1], [0, 1]]`       |
|            | inverts exactly because nothing truncates, and                |
|            | `det([[1, 2], [3, 4]])` misses -2 because something does.     |
|            | Singular input parts company with the scalar path, which has  |
|            | an infinity to return and returns it; a decimal has none, so  |
|            | `inv` and `solve` raise while `det` still reports zero. The   |
|            | MANUAL claimed the scalar path raised as well --- it yields   |
|            | `-0.0` and `inf`, and the page now says which path does what. |
|            | 16 new tests (565 total).                                     |
| 2026-08-22 | v0.1.0 gate closed. The 5.6 gate routines were written but    |
|            | not reachable: `routines/logic.mojo` held `isclose`,          |
|            | `allclose` and the `logical_*` family, while                  |
|            | `linamo/__init__.mojo` re-exported `all` and `any` alone, so  |
|            | `la.isclose` did not resolve --- only a module-path import    |
|            | found them. Thirteen names added, and the line the export     |
|            | list had been following without saying so is now written at   |
|            | the top of the file: a routine an operator already spells is  |
|            | reached by its module path. That is why `mul`, `div` and      |
|            | `pow` are exported --- `*` became the matrix product and left |
|            | the element-wise three without a symbol --- and why `add`,    |
|            | `sub`, `floordiv`, `mod`, `neg` and the six comparisons are   |
|            | not. The `logical_*` and closeness families have no operator  |
|            | at all, so their `scalar_*` forms are exported beside their   |
|            | matrix forms; those of the comparisons are not, since         |
|            | `A > 0.0` is already the call. MANUAL gained a *Comparing     |
|            | matrices* chapter for them, and Appendix B lost the three     |
|            | entries it was claiming absent. Install path documented both  |
|            | ways: `pixi add linamo` from `modular-community` as the       |
|            | route, with the source and `.mojoc` routes for the window     |
|            | before the tag lands --- verified by running a program        |
|            | against `tests/linamo.mojoc` with no `src` on the import      |
|            | path. The README quickstart opened `fn main() raises:`,       |
|            | which Mojo 1.0 rejects outright (`'fn' has been removed`),    |
|            | so the first program a visitor copied did not compile. Also   |
|            | cleared the three docstring warnings `pixi run pack` emitted  |
|            | --- two `smatrix` parameter lists ordered against the         |
|            | declaration, one undocumented `T` in `from_numpy` --- which   |
|            | matter because `mojo doc` output is the symbol reference.     |
|            | A `tests/test_exports.mojo` pins the surface: every           |
|            | other test file imports from a module path, so none of        |
|            | them could see a name missing from the alias. 7 new tests     |
|            | (572 total); no behaviour changed.                            |
