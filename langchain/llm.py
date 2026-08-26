from __future__ import annotations

import os


# A tool-calling-capable model is REQUIRED — the agent drives every MCP server
# through the model's tool-call interface. These defaults are safe choices.
OPENROUTER_DEFAULT_MODEL = "openai/gpt-4o-mini"
ANTHROPIC_DEFAULT_MODEL = "claude-3-5-sonnet-latest"

# Keep requests affordable and cheap. Override per provider via env if needed.
DEFAULT_MAX_TOKENS = 4096


def _max_tokens(env_name: str) -> int:
    raw = os.getenv(env_name) or os.getenv("LLM_MAX_TOKENS")
    try:
        return int(raw) if raw else DEFAULT_MAX_TOKENS
    except ValueError:
        return DEFAULT_MAX_TOKENS


def build_llm(temperature: float = 0.0):
    """Return a configured chat model instance ready to hand to create_agent().

    Raises a clear error if neither provider key is configured.
    """
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")

    if openrouter_key:
        from langchain_openai import ChatOpenAI

        model = os.getenv("OPENROUTER_MODEL", OPENROUTER_DEFAULT_MODEL)
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=openrouter_key,
            base_url="https://openrouter.ai/api/v1",
            max_tokens=_max_tokens("OPENROUTER_MAX_TOKENS"),
        )

    if anthropic_key:
        from langchain_anthropic import ChatAnthropic

        model = os.getenv("ANTHROPIC_MODEL", ANTHROPIC_DEFAULT_MODEL)
        return ChatAnthropic(
            model=model,
            temperature=temperature,
            api_key=anthropic_key,
            max_tokens=_max_tokens("ANTHROPIC_MAX_TOKENS"),
        )

    raise RuntimeError(
        "No LLM key found. Set OPENROUTER_API_KEY (preferred) or ANTHROPIC_API_KEY "
        "in your .env file. Copy .env.example to .env and fill one in."
    )


def active_provider() -> str:
    """Human-readable string describing which provider will be used."""
    if os.getenv("OPENROUTER_API_KEY"):
        model = os.getenv("OPENROUTER_MODEL", OPENROUTER_DEFAULT_MODEL)
        return f"OpenRouter ({model}, max_tokens={_max_tokens('OPENROUTER_MAX_TOKENS')})"
    if os.getenv("ANTHROPIC_API_KEY"):
        model = os.getenv("ANTHROPIC_MODEL", ANTHROPIC_DEFAULT_MODEL)
        return f"Anthropic ({model}, max_tokens={_max_tokens('ANTHROPIC_MAX_TOKENS')})"
    return "NONE (no key configured)"
