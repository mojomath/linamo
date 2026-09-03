"""An element type that has a conjugate.

The axis `conj_transpose` dispatches on. Conjugation is the identity on a real
number, so the useful question is not "what is the conjugate of this element"
--- every element type can answer that --- but "does this element type have a
conjugate that is not itself". Only a type that says yes needs the extra pass.

Two things follow from Mojo's nominal conformance, and both are why this trait
is shaped the way it is.

It cannot be a requirement on all element types. `Float64` and `BigInt` belong
to the stdlib and to decimo, so Linamo can never declare conformance for them,
and a `Conjugable` bound on `conj_transpose` would exclude exactly the element
types the library is built around. So the trait marks the *smaller* set --- the
types that do conjugate --- and the routine's other overload takes everything
else through `where not conforms_to(T, Conjugable)`.

That negation is also what keeps the two overloads disjoint, which they have to
be. Both take a `MatrixView`, and a `Matrix` argument converts implicitly, so
if the bounds overlapped every call from a `Matrix` would need one implicit
conversion on each candidate and the compiler would rank neither above the
other. Every other pair of overloads in this library is disjoint by accident of
what conforms to what --- no scalar is `decimo.Numeric`, no complex number is
`Comparable`. This pair is disjoint by construction.
"""


trait Conjugable(Copyable, Deinitable):
    """An element type whose conjugate differs from itself.

    Conform a type to this and `conj_transpose` conjugates it; leave it
    unconformed and `conj_transpose` is the plain transpose. Only types Linamo
    owns can conform, which is the whole population that needs to.
    """

    def conj(self) -> Self:
        """Returns the conjugate of this value.

        Returns:
            The conjugate.
        """
        ...
