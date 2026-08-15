# ===----------------------------------------------------------------------=== #
# Portions of this file are adapted from Decimo (Copyright 2025-2026 Yuhao Zhu),
# licensed under the Apache License, Version 2.0.
# https://github.com/forfudan/decimo/blob/main/src/decimo/errors.mojo
# ===----------------------------------------------------------------------=== #

"""
Implements error handling for Linamo.

The error messages follow the Python traceback format as closely as possible:

```
Traceback (most recent call last):
  File "./src/linamo/routines/math.mojo", line 197, in _elementwise_view()
ValueError: Input matrices must have the same shape.
```

File name and line number are captured automatically at the raise site with
`call_location()`, and the absolute path is shortened to a `./`-relative one so
that a traceback does not leak the build machine's directory layout. The
function name still has to be passed by hand, because Mojo has no way to ask
for the name of the enclosing function at runtime.

The error kinds, the traceback formatting, the ANSI colouring and the path
shortening are adapted from Decimo's `errors.mojo`. Two things differ. Decimo
exposes each kind as a type alias (`comptime ValueError = BaseError[...]`);
here they are constructor functions that return a plain `Error`, for the reason
given below. And `_USE_COLOUR` makes the ANSI escapes switchable, which matters
more for a library whose errors surface inside other people's test output.
"""

from std.reflection import call_location


# ===----------------------------------------------------------------------=== #
# ANSI colour codes
#
# Mimics the traceback colouring of Python's `rich`. Set `_USE_COLOUR` to False
# to emit plain text - worth doing if the escapes ever show up as literal
# `\033[1m` noise in a log file rather than as colour in a terminal.
# ===----------------------------------------------------------------------=== #

comptime _USE_COLOUR = True

comptime _RESET = "\033[0m" if _USE_COLOUR else ""
comptime _BOLD = "\033[1m" if _USE_COLOUR else ""
comptime _DIM = "\033[2m" if _USE_COLOUR else ""
comptime _RED = "\033[31m" if _USE_COLOUR else ""
comptime _GREEN = "\033[32m" if _USE_COLOUR else ""
comptime _YELLOW = "\033[33m" if _USE_COLOUR else ""
comptime _MAGENTA = "\033[35m" if _USE_COLOUR else ""

# Semantic aliases for error formatting.
comptime _CLR_ERROR_TYPE = _BOLD + _RED  # Error type name, e.g. ValueError
comptime _CLR_TRACEBACK = _BOLD  # "Traceback (most recent call last):"
comptime _CLR_FILE_PATH = _MAGENTA  # File path
comptime _CLR_LINE_NUM = _GREEN  # Line number
comptime _CLR_FUNC_NAME = _YELLOW  # Function name
comptime _CLR_MSG_TEXT = _BOLD  # Error message text
comptime _CLR_CHAIN_MSG = _DIM  # Chained error separator


# ===----------------------------------------------------------------------=== #
# Path shortening
# ===----------------------------------------------------------------------=== #


@always_inline
def _shorten_path(full_path: String) -> String:
    """Shortens an absolute file path to a `./`-prefixed relative one.

    Looks for the directory markers `src/`, `tests/` and `benches/` and returns
    the path from the rightmost marker found. If none is present, returns just
    the file name.

    Args:
        full_path: The absolute file path to shorten.

    Returns:
        A shortened relative path.

    Notes:

    Forwards to a `@no_inline` implementation so that the `rfind` and slicing
    work stays out of every inlined raise site.
    """
    return _shorten_path_implementation(full_path)


@no_inline
def _shorten_path_implementation(full_path: String) -> String:
    var src_idx = full_path.rfind("src/")
    var tests_idx = full_path.rfind("tests/")
    var benches_idx = full_path.rfind("benches/")

    # A path may contain more than one marker, as in
    # `.../tests/.../src/...` or `.../benches/.../src/...`.
    # The rightmost one gives the shortest relative path.
    var idx = src_idx
    if tests_idx > idx:
        idx = tests_idx
    if benches_idx > idx:
        idx = benches_idx

    if idx >= 0:
        return "./" + String(full_path[byte=idx:])
    var last_slash = full_path.rfind("/")
    if last_slash >= 0:
        return String(full_path[byte = last_slash + 1 :])
    return full_path


# ===----------------------------------------------------------------------=== #
# Error kinds
#
# [Mojo Miji]
# Mojo 1.0.0 made typed raises (`raises ValueError`) strictly invariant: a
# `raises Error` function cannot call a `raises ValueError` one, and vice
# versa. That makes a typed-raise public API impossible to combine with
# `std.testing` (which raises `Error`) or with any downstream caller.
#
# So the error kinds below are constructor functions that build a `LinamoError`
# payload and wrap it in a plain `Error`. Call sites are unchanged --
# `raise ValueError(function=..., message=...)` still works - and the rich
# traceback formatting from `LinamoError.write_to` is preserved, because
# `Error` is constructed from a `Writable`.
#
# Each one is `@always_inline` so that the `call_location()` inside reports the
# `raise` site rather than a line in this file.
# ===----------------------------------------------------------------------=== #


@always_inline
def OverflowError(
    *,
    function: String,
    message: String,
    previous_error: Optional[Error] = None,
) -> Error:
    """Builds an `Error` describing overflow errors in Linamo.

    Args:
        function: The function where the error occurred.
        message: A message describing the error.
        previous_error: An optional previous error that caused this error.

    Returns:
        An `Error` carrying a `LinamoError["OverflowError"]` payload.
    """
    var loc = call_location()
    return Error(
        LinamoError[error_type="OverflowError"](
            _shorten_path(String(loc.file_name())),
            loc.line(),
            function,
            message,
            previous_error,
        )
    )


@always_inline
def IndexError(
    *,
    function: String,
    message: String,
    previous_error: Optional[Error] = None,
) -> Error:
    """Builds an `Error` describing index errors in Linamo.

    Args:
        function: The function where the error occurred.
        message: A message describing the error.
        previous_error: An optional previous error that caused this error.

    Returns:
        An `Error` carrying a `LinamoError["IndexError"]` payload.
    """
    var loc = call_location()
    return Error(
        LinamoError[error_type="IndexError"](
            _shorten_path(String(loc.file_name())),
            loc.line(),
            function,
            message,
            previous_error,
        )
    )


@always_inline
def KeyError(
    *,
    function: String,
    message: String,
    previous_error: Optional[Error] = None,
) -> Error:
    """Builds an `Error` describing key errors in Linamo.

    Args:
        function: The function where the error occurred.
        message: A message describing the error.
        previous_error: An optional previous error that caused this error.

    Returns:
        An `Error` carrying a `LinamoError["KeyError"]` payload.
    """
    var loc = call_location()
    return Error(
        LinamoError[error_type="KeyError"](
            _shorten_path(String(loc.file_name())),
            loc.line(),
            function,
            message,
            previous_error,
        )
    )


@always_inline
def ValueError(
    *,
    function: String,
    message: String,
    previous_error: Optional[Error] = None,
) -> Error:
    """Builds an `Error` describing value errors in Linamo.

    Args:
        function: The function where the error occurred.
        message: A message describing the error.
        previous_error: An optional previous error that caused this error.

    Returns:
        An `Error` carrying a `LinamoError["ValueError"]` payload.
    """
    var loc = call_location()
    return Error(
        LinamoError[error_type="ValueError"](
            _shorten_path(String(loc.file_name())),
            loc.line(),
            function,
            message,
            previous_error,
        )
    )


@always_inline
def ZeroDivisionError(
    *,
    function: String,
    message: String,
    previous_error: Optional[Error] = None,
) -> Error:
    """Builds an `Error` describing divided-by-zero errors in Linamo.

    Args:
        function: The function where the error occurred.
        message: A message describing the error.
        previous_error: An optional previous error that caused this error.

    Returns:
        An `Error` carrying a `LinamoError["ZeroDivisionError"]` payload.
    """
    var loc = call_location()
    return Error(
        LinamoError[error_type="ZeroDivisionError"](
            _shorten_path(String(loc.file_name())),
            loc.line(),
            function,
            message,
            previous_error,
        )
    )


@always_inline
def ConversionError(
    *,
    function: String,
    message: String,
    previous_error: Optional[Error] = None,
) -> Error:
    """Builds an `Error` describing conversion errors in Linamo.

    Args:
        function: The function where the error occurred.
        message: A message describing the error.
        previous_error: An optional previous error that caused this error.

    Returns:
        An `Error` carrying a `LinamoError["ConversionError"]` payload.
    """
    var loc = call_location()
    return Error(
        LinamoError[error_type="ConversionError"](
            _shorten_path(String(loc.file_name())),
            loc.line(),
            function,
            message,
            previous_error,
        )
    )


# ===----------------------------------------------------------------------=== #
# The payload
# ===----------------------------------------------------------------------=== #


struct LinamoError[error_type: String = "LinamoError"](Writable):
    """Base type for all Linamo errors.

    The formatted output mimics a Python traceback:

    ```
    Traceback (most recent call last):
      File "./src/linamo/routines/math.mojo", line 197, in _elementwise_view()
    ValueError: Input matrices must have the same shape.
    ```

    File name and line number are captured at the raise site; the function name
    is supplied by the caller.

    Parameters:
        error_type: The type of the error, e.g., "OverflowError", "IndexError".
    """

    var file: String
    """The source file where the error occurred (captured automatically)."""
    var line: Int
    """The line number where the error occurred (captured automatically)."""
    var function: String
    """The function where the error occurred."""
    var message: String
    """A message describing the error."""
    var previous_error: Optional[String]
    """The formatted previous error that caused this one, if any."""

    def __init__(
        out self,
        file: String,
        line: Int,
        function: String,
        message: String,
        previous_error: Optional[Error],
    ):
        """Creates a `LinamoError`.

        Args:
            file: The file where the error occurred, already shortened.
            line: The line where the error occurred.
            function: The function where the error occurred.
            message: A message describing the error.
            previous_error: An optional previous error that caused this one.
        """
        self.file = file
        self.line = line
        self.function = function
        self.message = message
        if previous_error is None:
            self.previous_error = None
        else:
            self.previous_error = String(previous_error.value())

    def write_to[W: Writer](self, mut writer: W):
        """Writes a Python-style traceback to a writer.

        A chained error is printed first, as Python does:

        ```
        Traceback (most recent call last):
          File "./src/linamo/routines/creation.mojo", line 40, in eye()
        ValueError: inner error message

        The above exception was the direct cause of the following exception:

        Traceback (most recent call last):
          File "./src/linamo/routines/linalg.mojo", line 92, in inv()
        LinamoError: outer error message
        ```

        Parameters:
            W: A type conforming to the `Writer` interface.

        Args:
            writer: The writer instance.
        """
        # The chained error comes first, as in Python.
        if self.previous_error is not None:
            writer.write(self.previous_error.value())
            writer.write("\n")
            writer.write(_CLR_CHAIN_MSG)
            writer.write(
                "The above exception was the direct cause of the following"
                " exception:"
            )
            writer.write(_RESET)
            writer.write("\n\n")

        writer.write(_CLR_TRACEBACK)
        writer.write("Traceback (most recent call last):")
        writer.write(_RESET)
        writer.write("\n")

        writer.write('  File "')
        writer.write(_CLR_FILE_PATH)
        writer.write(self.file)
        writer.write(_RESET)
        writer.write('", line ')
        writer.write(_CLR_LINE_NUM)
        writer.write(String(self.line))
        writer.write(_RESET)
        writer.write(", in ")
        writer.write(_CLR_FUNC_NAME)
        writer.write(self.function)
        writer.write(_RESET)
        writer.write("\n")

        writer.write(_CLR_ERROR_TYPE)
        writer.write(Self.error_type)
        writer.write(_RESET)
        writer.write(": ")
        writer.write(_CLR_MSG_TEXT)
        writer.write(self.message)
        writer.write(_RESET)
        writer.write("\n")
