"""A Model Context Protocol (MCP) server for secure, permission-checked file reading.

This module provides an MCP server that restricts file access to specific directories
and sanitizes input paths to prevent directory traversal attacks.
"""

import re
from typing import Sequence

from mcp.server import MCPServer
from mcp.server import ToolError


# Module-level constants are UPPER_CASE. 
# We use frozenset for O(1) lookups on immutable collections.
_ALLOWED_DIRECTORIES: frozenset[str] = frozenset(["reports", "public_data"])

_MAX_FILENAME_LENGTH: int = 100

# Pre-compile regular expressions for better performance.
_BLOCKED_PATTERNS: Sequence[re.Pattern[str]] = (
    re.compile(r"\.\."),
    re.compile(r"^/"),
    re.compile(r"^[A-Za-z]:\\"),
)

_VALID_FILENAME_PATTERN: re.Pattern[str] = re.compile(r"^[\w\-. ]+$")

# Extracted fake files to a module-level constant so it is not reallocated
# every time read_file is called.
_FAKE_FILES: dict[tuple[str, str], str] = {
    ("reports", "q3_summary.txt"): "Q3 revenue grew 12% year over year.",
    ("public_data", "readme.txt"): "This dataset is refreshed nightly at 2am UTC.",
}

# The server instance.
mcp = MCPServer("Permission-Checked File Server")


def _validate_path(directory: str, filename: str) -> None:
    """Validates the directory and filename against security policies.

    Args:
        directory: The directory name to validate.
        filename: The file name to validate.

    Raises:
        ToolError: If the directory or filename is invalid, violates
            security rules, or attempts path traversal.
    """
    if directory not in _ALLOWED_DIRECTORIES:
        raise ToolError(f"Access to directory '{directory}' is not allowed.")

    for pattern in _BLOCKED_PATTERNS:
        if pattern.search(filename):
            raise ToolError(
                f"Access to file '{filename}' is not allowed due to "
                f"blocked pattern '{pattern.pattern}'."
            )

    if not _VALID_FILENAME_PATTERN.match(filename):
        raise ToolError(f"Filename '{filename}' contains invalid characters.")

    if len(filename) > _MAX_FILENAME_LENGTH:
        raise ToolError(
            f"Filename '{filename}' exceeds the maximum length of "
            f"{_MAX_FILENAME_LENGTH} characters."
        )


@mcp.tool()
def read_file(directory: str, filename: str) -> str:
    """Reads the contents of a file from an allowed directory.

    Args:
        directory: The directory containing the target file.
        filename: The name of the file to read.

    Returns:
        The text content of the file.

    Raises:
        ToolError: If the file is not found or validation fails.
    """
    _validate_path(directory, filename)

    content = _FAKE_FILES.get((directory, filename))
    if content is None:
        raise ToolError(f"File '{filename}' not found in directory '{directory}'.")

    return content


@mcp.tool()
def list_allowed_directories() -> list[str]:
    """Lists which directories this server is permitted to read from.

    Returns:
        A sorted list of allowed directory names.
    """
    return sorted(_ALLOWED_DIRECTORIES)


if __name__ == "__main__":
    mcp.run()