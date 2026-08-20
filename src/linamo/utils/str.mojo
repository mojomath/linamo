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

    | `T`         | result     |
    |-------------|------------|
    | `Float64`   | `float64`  |
    | `Int32`     | `int32`    |
    | `BigInt`    | `BigInt`   |

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
        return String(reflect[T].base_name())
