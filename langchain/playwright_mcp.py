from __future__ import annotations

import os

from langchain_mcp_adapters.client import MultiServerMCPClient


def build_client() -> MultiServerMCPClient:
    """Create the Playwright MCP client configuration."""

    headless = os.getenv("PLAYWRIGHT_HEADLESS", "1") == "1"
    browser = os.getenv("PLAYWRIGHT_BROWSER", "chromium")

    args = [
        "-y",
        "@playwright/mcp@latest",
        "--browser",
        browser,
        "--isolated",
    ]

    if headless:
        args.append("--headless")

    return MultiServerMCPClient(
        {
            "playwright": {
                "transport": "stdio",
                "command": "npx",
                "args": args,
            }
        }
    )


def describe_config() -> str:
    """Return a readable summary of the Playwright configuration."""

    headless = os.getenv("PLAYWRIGHT_HEADLESS", "1") == "1"
    browser = os.getenv("PLAYWRIGHT_BROWSER", "chromium")

    return (
        f"@playwright/mcp · browser={browser} · "
        f"headless={headless} · isolated"
    )