"""
Matrices of arbitrary-precision numbers.

A `Matrix` takes an element *type*, so `Matrix[BigInt]` is an ordinary matrix
and everything structural about it -- slicing, transposition, reshaping,
sorting -- comes from core Linamo unchanged. What cannot come from there is the
arithmetic: every kernel in `linamo.routines` is written against `Scalar[d]`,
whose `+` lowers to a vector instruction that an arbitrary-precision element
has no equivalent of. That lives in `linamo.decimo` instead.

Run with:

```bash
pixi run decimo   # once, to build the decimo package into temp/
pixi run examples
```
"""

import linamo as la
import linamo.decimo as lad
from decimo import BigInt, BigDecimal


def _banner(title: String):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def main() raises:
    _banner("A MATRIX OF ARBITRARY-PRECISION INTEGERS")

    # The element type is written where `Float64` would go, and nothing else
    # about the call changes.
    var a = la.matrix[BigInt](
        [
            [BigInt(1), BigInt(2)],
            [BigInt(3), BigInt(4)],
        ]
    )
    print("a =\n", a)
    print("a @ a =\n", lad.matmul(a, a))
    print("trace(a) =", lad.trace(a))

    _banner("WHY: A VALUE NO DType CAN HOLD")

    # 60! is about 8.3e81. `UInt64` stops at 1.8e19 and `Float64` would have
    # kept 53 bits of it; a `BigInt` keeps every digit.
    var factorial = BigInt.one()
    for k in range(1, 61):
        factorial = factorial * BigInt(k)
    var big = lad.diag([factorial.copy(), BigInt(1)])
    print("diag(60!, 1) =\n", big)
    print("its trace    =", lad.trace(big))

    _banner("STRUCTURE COMES FROM CORE LINAMO")

    # None of the following is imported from `linamo.decimo`. They move or
    # compare elements, which needs no arithmetic and therefore no trait.
    var m = la.matrix[BigInt](
        [
            [BigInt(30), BigInt(10), BigInt(20)],
            [BigInt(3), BigInt(1), BigInt(2)],
        ]
    )
    print("m =\n", m)
    print("transpose(m) =\n", la.transpose(m))
    print("m[0:1, :] (a view, nothing copied) =\n", m[0:1, :])
    print("sort(m, axis=1) =\n", la.sort(m, 1))
    print("argsort(m, axis=1) =\n", la.argsort(m, 1))
    print("reshape(m, 3, 2) =\n", la.reshape(m, 3, 2))

    _banner("EXACT DECIMAL ARITHMETIC")

    # The same routines, over a type that keeps decimal fractions exactly --
    # 0.1 + 0.2 is 0.3 here, which it is not in binary floating point.
    var prices = la.matrix[BigDecimal](
        [
            [BigDecimal("0.1"), BigDecimal("0.2")],
            [BigDecimal("1.05"), BigDecimal("2.10")],
        ]
    )
    print("prices =\n", prices)
    print("prices + prices =\n", lad.add(prices, prices))
    print("total(prices) =", lad.total(prices))
    print("Float64 for comparison:", Float64(0.1) + Float64(0.2))
