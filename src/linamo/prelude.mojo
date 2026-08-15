"""
Prelude module for Linamo.
"""

# [Mojo Miji]
# The prelude module serves as a central place to define the most commonly used
# types, functions, and constants in Linamo. By importing the prelude, users
# can access all the essential components of the library without needing to
# import multiple modules separately. This design choice is inspired by the
# concept of a "prelude" in other programming languages, which provides a
# convenient and efficient way to access core functionality.
# However, we should be careful not to put too much content in the prelude, as
# it will potentially pollute the global namespace and make it harder for users
# to understand where specific functions or types are coming from.

import linamo as la

comptime float64 = DType.float64
"""Alias for 64-bit floating point data type."""
comptime float32 = DType.float32
"""Alias for 32-bit floating point data type."""
comptime int64 = DType.int64
"""Alias for 64-bit integer data type."""
comptime int32 = DType.int32
"""Alias for 32-bit integer data type."""
comptime int16 = DType.int16
"""Alias for 16-bit integer data type."""
comptime int8 = DType.int8
"""Alias for 8-bit integer data type."""
comptime uint64 = DType.uint64
"""Alias for 64-bit unsigned integer data type."""
comptime uint32 = DType.uint32
"""Alias for 32-bit unsigned integer data type."""
comptime uint16 = DType.uint16
"""Alias for 16-bit unsigned integer data type."""
comptime uint8 = DType.uint8
"""Alias for 8-bit unsigned integer data type."""
comptime int = DType.int
"""Alias for the default integer data type (int)."""
comptime uint = DType.uint
"""Alias for the default unsigned integer data type (uint)."""
