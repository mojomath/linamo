"""
The error kinds Linamo raises.

Every constructor here comes from `decimo.errors`, and this module exists to
be the name the rest of Linamo imports them under. Decimo is a hard dependency
either way, so the indirection buys nothing at build time; what it buys is one
edit instead of thirty when the set changes. A kind Linamo needs but Decimo
does not have is added here beside the re-exports, and no call site moves.

A re-export is not a wrapper: these stay the same `@always_inline` functions,
so `call_location()` still resolves to the Linamo file and line that raised,
not to this module.

The six below are the kinds the manual's error table promises. They are
constructor *functions* returning a plain `Error`, not distinct types --- Mojo
has no typed exceptions, so catching is `except e:` and inspection is on the
message.
"""

from decimo.errors import (
    ConversionError,
    IndexError,
    KeyError,
    OverflowError,
    ValueError,
    ZeroDivisionError,
)
