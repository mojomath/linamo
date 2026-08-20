"""
Matrices of arbitrary-precision numbers.

A `Matrix` takes an element *type*, so `Matrix[BInt]` is an ordinary matrix
with the ordinary operators. Everything structural -- slicing, transposition,
reshaping, sorting -- is the same code that serves `Matrix[Float64]`, and the
arithmetic differs only in where it comes from: a `Float64` gets it from a
vector instruction, a `BInt` from its own `+`. `a + b` and `a @ b` are spelled
the same either way.

The element types are re-exported by Linamo, so this file imports nothing but
`linamo`.

Run with:

```bash
pixi run examples
```
"""

import linamo as la
from linamo import BInt, Decimal


def _banner(title: String):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def main() raises:
    _banner("A MATRIX OF ARBITRARY-PRECISION INTEGERS")

    # The element type is written where `Float64` would go, and nothing else
    # about the call changes -- integer literals convert to `BInt` in the
    # nested list just as they would to a scalar.
    var a = la.matrix[BInt]([[1, 2], [3, 4]])
    print("a =\n", a)
    print("a @ a =\n", a @ a)
    print("a * 3 =\n", a * BInt(3))
    print("trace(a) =", la.trace(a))

    _banner("WHY: A VALUE NO DType CAN HOLD")

    # 60! is about 8.3e81. `UInt64` stops at 1.8e19 and `Float64` would have
    # kept 53 bits of it; a `BInt` keeps every digit.
    var factorial = BInt.one()
    for k in range(1, 61):
        factorial = factorial * BInt(k)
    var big = la.diag([factorial.copy(), BInt(1)])
    print("diag(60!, 1) =\n", big)
    print("its trace    =", la.trace(big))

    _banner("STRUCTURE IS THE SAME CODE AS FOR A SCALAR MATRIX")

    # None of the following touches the elements' arithmetic. These routines
    # move or compare elements, so they were generic over the element type
    # before `Numeric` existed and stay that way.
    var m = la.matrix[BInt]([[30, 10, 20], [3, 1, 2]])
    print("m =\n", m)
    print("m.transpose() =\n", m.transpose())
    print("m[0:1, :] (a view, nothing copied) =\n", m[0:1, :])
    print("sort(m, axis=1) =\n", la.sort(m, 1))
    print("argsort(m, axis=1) =\n", la.argsort(m, 1))
    print("reshape(m, 3, 2) =\n", la.reshape(m, 3, 2))

    _banner("EXACT DECIMAL ARITHMETIC")

    # The same operators, over a type that keeps decimal fractions exactly --
    # 0.1 + 0.2 is 0.3 here, which it is not in binary floating point.
    var prices = la.matrix[Decimal](
        [
            [Decimal("0.1"), Decimal("0.2")],
            [Decimal("1.05"), Decimal("2.10")],
        ]
    )
    print("prices =\n", prices)
    print("prices + prices =\n", prices + prices)
    print("sum(prices) =", la.sum(prices))
    print("Float64 for comparison:", Float64(0.1) + Float64(0.2))
