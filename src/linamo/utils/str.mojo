"""
String helpers shared by the matrix types.
"""

from std.reflection import reflect


def element_type_name[T: AnyType]() -> String:
    """Returns the display name of a matrix element type.

    `reflect` spells a scalar element in full, as `SIMD[DType.float64, 1]`.
    That is the honest name of the type but not the one anybody writes it by,
    so a scalar is reported by its dtype and every other element type by its
    unqualified struct name:

    | `T`                        | result             |
    |----------------------------|--------------------|
    | `Float64`                  | `float64`          |
    | `Int32`                    | `int32`            |
    | `BigInt`                   | `BigInt`           |
    | `Complex[DType.float64]`   | `Complex[float64]` |

    A parameterised element keeps its parameters: they are what separates
    `Complex[float64]` from `Complex[float32]`, and a header that printed both
    as `Complex` would name two different matrices the same. Only the module
    path is dropped, and `DType.` inside the parameters, so that a component
    dtype is spelled the way a scalar element is.

    Parameters:
        T: The element type to name.

    Returns:
        The name to print in a matrix header.
    """
    comptime _PREFIX = "SIMD[DType."
    comptime _SUFFIX = ", 1]"
    comptime full = reflect[T].name()
    comptime if full.startswith(_PREFIX) and full.endswith(_SUFFIX):
        return String(
            full[
                byte = _PREFIX.byte_length() : full.byte_length()
                - _SUFFIX.byte_length()
            ]
        )
    else:
        # `reflect.name()` is fully qualified and parameterised
        # (`linamo.types.complex.Complex[DType.float64]`), while `base_name()`
        # is neither (`Complex`). The useful name is in between, so it is cut
        # here: the module path is everything up to the last `.` that precedes
        # the parameter list, and dropping it leaves the parameters intact.
        comptime bracket = full.find("[")
        comptime head_end = bracket if bracket != -1 else full.byte_length()
        comptime dot = String(full[byte=0:head_end]).rfind(".")
        return String(full[byte = dot + 1 :]).replace("DType.", "")
