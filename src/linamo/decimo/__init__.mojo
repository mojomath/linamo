"""
Linear algebra over Decimo's arbitrary-precision numbers.

`Matrix[BigInt]`, `Matrix[BigDecimal]` and `Matrix[Decimal128]` are ordinary
matrices (the types took an element *type* rather than a `DType` precisely
so that they could be) but their arithmetic cannot live in
`linamo.routines`. Every routine there is written against `Scalar[d]`, whose
`+` the compiler lowers to a vector instruction; an arbitrary-precision element
has no such instruction and no dtype to name.

So the arithmetic lives here instead, written once against `decimo.Numeric`.
The split is the whole reason this is a submodule rather than a handful of
extra overloads next door:

- **Core Linamo never mentions decimo.** `pixi run test` builds and runs the
  whole scalar library with decimo nowhere on the include path.
- **Importing Linamo does not import decimo.** Only `from linamo.decimo import
  ...` pulls it in.
- **It is the seam.** Shipping `linamo-decimo` as a separate package later is
  a directory move, not a refactor.

What it is *not* is a way to avoid the dependency while the two ship together:
Mojo has no conditional imports, so `pixi run pack` compiles this directory
too and needs decimo resolvable. `tools/ensure_decimo.sh` is what makes it so.

Not everything needs to be here. Anything that only moves elements --- slicing,
`transpose`, `reshape`, `flatten`, `contiguous`, iteration --- and anything
that only compares them (`sort`, `argsort`, `argmax`, `argmin`, which ride
the *stdlib* `Comparable`) is generic in core Linamo already and works for a
`BigInt` matrix with no import from here at all.
"""

from linamo.decimo.math import (
    add,
    sub,
    mul,
    neg,
    matmul,
    scalar_add,
    scalar_mul,
    trace,
    total,
)
from linamo.decimo.creation import zeros, ones, identity, eye, diag
