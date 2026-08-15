"""
Implements error handling for MatMojo.
"""

from std.pathlib.path import cwd

# [Mojo Miji]
# Mojo 1.0.0 made typed raises (`raises ValueError`) strictly invariant: a
# `raises Error` function cannot call a `raises ValueError` one, and vice
# versa.  That makes a typed-raise public API impossible to combine with
# `std.testing` (which raises `Error`) or with any downstream caller.
#
# So the error *kinds* below are constructor functions that build a
# `MatMojoError` payload and wrap it in a plain `Error`.  Call sites are
# unchanged -- `raise ValueError(file=..., function=..., message=...)` still
# works -- and the rich traceback formatting from `MatMojoError.write_to` is
# preserved, because `Error` is constructed from a `Writable`.


def OverflowError(
    *,
    file: String,
    function: String,
    message: Optional[String] = None,
    previous_error: Optional[Error] = None,
) -> Error:
    """Builds an `Error` describing overflow errors in MatMojo.

    Args:
        file: The file where the error occurred.
        function: The function where the error occurred.
        message: An optional message describing the error.
        previous_error: An optional previous error that caused this error.

    Returns:
        An `Error` carrying a `MatMojoError["OverflowError"]` payload.
    """
    return Error(
        MatMojoError[error_type="OverflowError"](
            file, function, message, previous_error
        )
    )


def IndexError(
    *,
    file: String,
    function: String,
    message: Optional[String] = None,
    previous_error: Optional[Error] = None,
) -> Error:
    """Builds an `Error` describing index errors in MatMojo.

    Args:
        file: The file where the error occurred.
        function: The function where the error occurred.
        message: An optional message describing the error.
        previous_error: An optional previous error that caused this error.

    Returns:
        An `Error` carrying a `MatMojoError["IndexError"]` payload.
    """
    return Error(
        MatMojoError[error_type="IndexError"](
            file, function, message, previous_error
        )
    )


def KeyError(
    *,
    file: String,
    function: String,
    message: Optional[String] = None,
    previous_error: Optional[Error] = None,
) -> Error:
    """Builds an `Error` describing key errors in MatMojo.

    Args:
        file: The file where the error occurred.
        function: The function where the error occurred.
        message: An optional message describing the error.
        previous_error: An optional previous error that caused this error.

    Returns:
        An `Error` carrying a `MatMojoError["KeyError"]` payload.
    """
    return Error(
        MatMojoError[error_type="KeyError"](
            file, function, message, previous_error
        )
    )


def ValueError(
    *,
    file: String,
    function: String,
    message: Optional[String] = None,
    previous_error: Optional[Error] = None,
) -> Error:
    """Builds an `Error` describing value errors in MatMojo.

    Args:
        file: The file where the error occurred.
        function: The function where the error occurred.
        message: An optional message describing the error.
        previous_error: An optional previous error that caused this error.

    Returns:
        An `Error` carrying a `MatMojoError["ValueError"]` payload.
    """
    return Error(
        MatMojoError[error_type="ValueError"](
            file, function, message, previous_error
        )
    )


def ZeroDivisionError(
    *,
    file: String,
    function: String,
    message: Optional[String] = None,
    previous_error: Optional[Error] = None,
) -> Error:
    """Builds an `Error` describing divided-by-zero errors in MatMojo.

    Args:
        file: The file where the error occurred.
        function: The function where the error occurred.
        message: An optional message describing the error.
        previous_error: An optional previous error that caused this error.

    Returns:
        An `Error` carrying a `MatMojoError["ZeroDivisionError"]` payload.
    """
    return Error(
        MatMojoError[error_type="ZeroDivisionError"](
            file, function, message, previous_error
        )
    )


def ConversionError(
    *,
    file: String,
    function: String,
    message: Optional[String] = None,
    previous_error: Optional[Error] = None,
) -> Error:
    """Builds an `Error` describing conversion errors in MatMojo.

    Args:
        file: The file where the error occurred.
        function: The function where the error occurred.
        message: An optional message describing the error.
        previous_error: An optional previous error that caused this error.

    Returns:
        An `Error` carrying a `MatMojoError["ConversionError"]` payload.
    """
    return Error(
        MatMojoError[error_type="ConversionError"](
            file, function, message, previous_error
        )
    )


comptime HEADER_OF_ERROR_MESSAGE = """
---------------------------------------------------------------------------
MatMojoError                             Traceback (most recent call last)
"""


struct MatMojoError[error_type: String = "MatMojoError"](Writable):
    """Base type for all MatMojo errors.

    Parameters:
        error_type: The type of the error, e.g., "OverflowError", "IndexError".

    Fields:

    file: The file where the error occurred.
    function: The function where the error occurred.
    message: An optional message describing the error.
    previous_error: An optional previous error that caused this error.
    """

    var file: String
    var function: String
    var message: Optional[String]
    var previous_error: Optional[String]

    def __init__(
        out self,
        file: String,
        function: String,
        message: Optional[String],
        previous_error: Optional[Error],
    ):
        self.file = file
        self.function = function
        self.message = message
        if previous_error is None:
            self.previous_error = None
        else:
            self.previous_error = "\n".join(
                String(previous_error.value()).split("\n")[3:]
            )

    def __str__(self) -> String:
        if self.message is None:
            return (
                "Traceback (most recent call last):\n"
                + '  File "'
                + self.file
                + '"'
                + " in "
                + self.function
                + "\n\n"
            )

        else:
            return (
                "Traceback (most recent call last):\n"
                + '  File "'
                + self.file
                + '"'
                + " in "
                + self.function
                + "\n\n"
                + String(Self.error_type)
                + ": "
                + self.message.value()
                + "\n"
            )

    def write_to[W: Writer](self, mut writer: W):
        writer.write("\n")
        writer.write(("-" * 80))
        writer.write("\n")
        writer.write(Self.error_type.ascii_ljust(47, " "))
        writer.write("Traceback (most recent call last)\n")
        writer.write('File "')
        try:
            writer.write(String(cwd()))
        except e:
            pass
        finally:
            writer.write("/")
        writer.write(self.file)
        writer.write('"\n')
        writer.write("----> ")
        writer.write(self.function)
        if self.message is None:
            writer.write("\n")
        else:
            writer.write("\n\n")
            writer.write(Self.error_type)
            writer.write(": ")
            writer.write(self.message.value())
            writer.write("\n")
        if self.previous_error is not None:
            writer.write("\n")
            writer.write(self.previous_error.value())
