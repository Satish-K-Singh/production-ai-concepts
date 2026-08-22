import os
from langchain_mcp_adapters.client import MultiServerMCPClient

REMOTE_URL = "https://api.githubcopilot.com/mcp/"
DEFAULT_TOOLSETS = "repos,issues,pull_requests"

def _require_pat():
    pat = os.environ.get("GITHUB_PAT")
    if not pat:
        raise ValueError("GITHUB_PAT environment variable is not set.")
    return pat

def build_client() -> MultiServerMCPClient:
    pat = _require_pat()
    mode = os.getenv("GITHUB_MODE", "remote").lower()
    toolsets = os.getenv("GITHUB_TOOLSETS", DEFAULT_TOOLSETS)
    read_only = os.getenv("GITHUB_READ_ONLY", "1")

    # Mode A: Docker configuration
    if mode == "docker":
        args = [
            "run", "-i", "--rm",
            "-e", "GITHUB_PERSONAL_ACCESS_TOKEN",
            "-e", "GITHUB_TOOLSETS",
            "-e", "GITHUB_READ_ONLY",
            "ghcr.io/github/github-mcp-server",
        ]
       
        return MultiServerMCPClient(
            {
                "github": {
                    "command": "docker",
                    "args": args,
                    "transport": "stdio",
                    "env": {
                        "GITHUB_PERSONAL_ACCESS_TOKEN": pat,
                        "GITHUB_TOOLSETS": toolsets,
                        "GITHUB_READ_ONLY": read_only
                    }
                }
            }
        )

    # Mode B: Remote configuration
    headers = {
        "Authorization": f"Bearer {pat}",
        "X-GitHub-Toolsets": toolsets,
    }

    if read_only in ("1", "true", "True"):
        headers["X-GitHub-Read-Only"] = "true"

    return MultiServerMCPClient(
        {
            "github": {
                "url": REMOTE_URL,
                "headers": headers,
                "transport": "streamable_http",
            }   
        }
    )

def describe_config() -> str:
    mode = os.getenv("GITHUB_MODE", "remote").lower()
    toolsets = os.getenv("GITHUB_TOOLSETS", DEFAULT_TOOLSETS)
    read_only = os.getenv("GITHUB_READ_ONLY", "1")
    where = REMOTE_URL if mode != "docker" else "ghcr.io/github/github-mcp-server (Docker)"
    
    
    return f"Mode: {mode}, Toolsets: {toolsets}, Read-only: {read_only}, Endpoint: {where}"