"""
Matrices of complex numbers.

`Matrix[CFloat64]` is an ordinary matrix with the ordinary operators. What
changes is where the arithmetic comes from and, in three places, what a
transpose means.

The element type is Linamo's `Complex`, which is a wrapper over the stdlib's
`std.complex.ComplexSIMD` and forwards every operator to it. It exists for one
line the stdlib could not write: conformance to `decimo.Numeric`, a trait
declared after `ComplexSIMD` was, which Mojo's nominal rule allows only at the
struct's own definition. That single declaration is what puts a complex matrix
through the same `zeros`, `eye`, `@`, `lu` and `det` a `Matrix[BInt]` goes
through.

Run with:

```bash
pixi run examples
```
"""

import linamo as la
from linamo import CFloat64


def _banner(title: String):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def main() raises:
    _banner("A MATRIX OF COMPLEX NUMBERS")

    # The element type is written where `Float64` would go. `1+2i` is the
    # literal syntax, and it is also what the matrix prints, so the printed
    # form reads back in.
    var a = la.from_string[CFloat64]("[[1+2i, 3-1i], [2i, 4]]")
    print("a =\n", a)
    print("a + a =\n", a + a)
    print("a @ a =\n", a @ a)
    print("a.mul(a)  (element-wise) =\n", a.mul(a))
    print("trace(a) =", la.trace(a))

    _banner("THE ARITHMETIC IS COMPLEX, NOT COMPONENT-WISE")

    # `i * i == -1` is the whole difference. A component-wise product would
    # give `0+0i` for the square of `i` and the matrix product would be wrong
    # everywhere.
    print("i * i =", CFloat64(0, 1) * CFloat64(0, 1))
    print("1 / i =", CFloat64.one() / CFloat64(0, 1), " (not the conjugate)")
    print("(3+4i).norm() =", CFloat64(3, 4).norm())
    print("(3+4i).conj() =", CFloat64(3, 4).conj())
    print("sqrt(3+4i) =", CFloat64(3, 4).sqrt())
    print("sqrt(-1) =", CFloat64(-1, 0).sqrt())

    _banner("A LITERAL MAY LEAVE OUT WHAT IT IMPLIES")

    # `3` is real, `2i` is imaginary, `-i` has no coefficient. A mostly-real
    # matrix does not have to write `+0i` in every cell.
    print(
        "[[1, 2i], [-i, 3.5-0.5i]] =\n",
        la.from_string[CFloat64]("[[1, 2i], [-i, 3.5-0.5i]]"),
    )

    _banner("STRUCTURE IS THE SAME CODE AS FOR A SCALAR MATRIX")

    # None of these touches the elements' arithmetic, so they were generic
    # over the element type long before complex numbers existed here.
    var m = la.from_string[CFloat64]("[[1+1i, 2, 3-1i], [4i, 5, 6]]")
    print("m =\n", m)
    print("m.transpose() =\n", m.transpose())
    print("m[0:1, :] (a view, nothing copied) =\n", m[0:1, :])
    print("reshape(m, 3, 2) =\n", la.reshape(m, 3, 2))

    _banner("TRANSPOSE IS NOT THE TRANSPOSE YOU WANT")

    # The distinction that every complex algorithm below turns on. The plain
    # transpose moves elements; the conjugate transpose also negates the
    # imaginary parts, and it is the one that makes `A @ A^H` Hermitian: a
    # real, non-negative diagonal and conjugate off-diagonal pairs.
    print("a =\n", a)
    print("transpose(a) =\n", la.transpose(a))
    print("conj_transpose(a) =\n", la.conj_transpose(a))
    print("a @ conj_transpose(a)  (Hermitian) =\n", a @ la.conj_transpose(a))
    print("a @ transpose(a)  (not) =\n", a @ la.transpose(a))

    # Conjugation is the identity on a real element, so the same name works
    # there and an algorithm written with it stays correct over both.
    var real = la.matrix[Float64]([[1.0, 2.0], [3.0, 4.0]])
    print(
        "conj_transpose of a real matrix is just its transpose =\n",
        la.conj_transpose(real),
    )

    _banner("ELIMINATION, PIVOTING ON MAGNITUDE")

    # `lu`, `det`, `solve` and `inv` reach a complex matrix through the same
    # names a `Float64` matrix uses. Partial pivoting ranks candidates by
    # `|z|`: there is no ordering on the complex plane, but a magnitude is a
    # real number and does have one.
    var b = la.from_string[CFloat64]("[[1+1i, 2], [3, 4-1i]]")
    print("b =\n", b)
    print("det(b) =", la.det(b), " (by hand: -1+3i)")
    print("inv(b) =\n", la.inv(b))
    print("b @ inv(b) =\n", b @ la.inv(b))
    print("b ** -1 (the same matrix) =\n", b**-1)

    var rhs = la.from_string[CFloat64]("[[1], [1i]]")
    var x = la.solve(b, rhs)
    print("solve(b, [[1], [i]]) =\n", x)
    print("b @ x  (the right-hand side again) =\n", b @ x)

    _banner("HERMITIAN FACTORISATIONS")

    # These three needed a real algorithm change rather than a new pivot rule:
    # every transpose in them is a conjugate transpose. `A = L L^H` is what
    # makes the Cholesky diagonal a sum of squared magnitudes, hence real.
    var h = la.from_string[CFloat64]("[[4, 1-1i], [1+1i, 3]]")
    print("h (Hermitian, positive-definite) =\n", h)
    var L = la.cholesky(h)
    print("cholesky(h) = L =\n", L)
    print("L @ conj_transpose(L)  (h again) =\n", L @ la.conj_transpose(L))

    # Q comes out unitary rather than orthogonal: `Q^H Q` is the identity.
    var tall = la.from_string[CFloat64]("[[1+2i, 3-1i], [2i, 4], [1, -1i]]")
    var factored = la.qr(tall)
    ref Q = factored[0]
    ref R = factored[1]
    print("qr on a 3x2: R (upper triangular) =\n", R)
    print("conj_transpose(Q) @ Q  (the identity) =\n", la.conj_transpose(Q) @ Q)
    print("Q @ R  (the input again) =\n", Q @ R)

    # Least squares projects with `Q^H`, so the residual it minimises is a sum
    # of squared magnitudes.
    var design = la.from_string[CFloat64]("[[1, 0], [0, 1], [1, 1]]")
    var target = la.from_string[CFloat64]("[[1+1i], [2], [0]]")
    var fit = la.lstsq(design, target)
    print("lstsq =\n", fit)
    print(
        "conj_transpose(A) @ (A x - b)  (zero: the residual is orthogonal) =\n",
        la.conj_transpose(design) @ ((design @ fit) - target),
    )

    _banner("COMPONENT WIDTH")

    # Named after the components, as `Float64` is: a `CFloat64` is a pair of
    # `Float64`. NumPy counts the bits of the pair and calls this `complex128`.
    print("CFloat32 element =", la.CFloat32(1, 2))
    print("CFloat64 element =", CFloat64(1, 2))
    print("a 32-bit complex matrix =\n", la.zeros[la.CFloat32](2, 2))
