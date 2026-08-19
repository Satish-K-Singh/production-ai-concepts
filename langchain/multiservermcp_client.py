from __future__ import annotations

import os
import shutil
import sys

from langchain_mcp_adapters.client import MultiServerMCPClient

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
DB_PATH = os.path.join(DATA_DIR, "shopease.db")


def _sqlite_command() -> tuple[str, list[str]]:
    """Resolve the SQLite MCP server command."""

    override = os.getenv("MCP_SQLITE_COMMAND")
    if override:
        if not os.path.exists(override) and not shutil.which(override):
            raise FileNotFoundError(
                f"MCP_SQLITE_COMMAND could not be found: {override}"
            )

        return override, ["--db-path", DB_PATH]

    uvx_path = shutil.which("uvx")
    if uvx_path:
        return uvx_path, [
            "mcp-server-sqlite",
            "--db-path",
            DB_PATH,
        ]

    sqlite_path = shutil.which("mcp-server-sqlite")
    if sqlite_path:
        return sqlite_path, [
            "--db-path",
            DB_PATH,
        ]

    # Run the installed Python module through the active virtual environment.
    return sys.executable, [
        "-m",
        "mcp_server_sqlite",
        "--db-path",
        DB_PATH,
    ]


def _filesystem_command() -> tuple[str, list[str]]:
    """Resolve the Filesystem MCP server command safely on Windows."""

    npx_path = shutil.which("npx")

    if not npx_path:
        raise FileNotFoundError(
            "npx was not found. Install Node.js 18 or newer, restart VS Code, "
            "and verify it using: npx --version"
        )

    server_args = [
        "-y",
        "@modelcontextprotocol/server-filesystem",
        DATA_DIR,
    ]

    # npx is normally npx.cmd on Windows, so run it through cmd.exe.
    if os.name == "nt":
        cmd_path = shutil.which("cmd")
        if not cmd_path:
            raise FileNotFoundError("Windows cmd.exe could not be found.")

        return cmd_path, ["/c", npx_path, *server_args]

    return npx_path, server_args


def build_client() -> MultiServerMCPClient:
    """Create a client connected to SQLite and Filesystem MCP servers."""

    os.makedirs(DATA_DIR, exist_ok=True)

    sqlite_cmd, sqlite_args = _sqlite_command()
    filesystem_cmd, filesystem_args = _filesystem_command()

    print(f"SQLite MCP command    : {sqlite_cmd}")
    print(f"Filesystem MCP command: {filesystem_cmd}")

    return MultiServerMCPClient(
        {
            "shopdb": {
                "command": sqlite_cmd,
                "args": sqlite_args,
                "transport": "stdio",
            },
            "files": {
                "command": filesystem_cmd,
                "args": filesystem_args,
                "transport": "stdio",
            },
        }
    )