from mcp.server import MCPServer

mcp = MCPServer("First MCP Server")

@mcp.tool()
def add(x: int, y: int) -> int:
    """Add two numbers."""
    return x + y

@mcp.tool()
def word_count(text: str) -> int:
    """Count the number of words in a string."""
    return len(text.split())

@mcp.tool()
def reverse_string(text: str) -> str:
    """Reverse a string."""
    return text[::-1]

@mcp.resource("resource://server-info")
def server_info()-> str:
    """Return server information."""
    return f"First MCP server- exposed to add, word_count, and reverse_string tools."

if __name__ == "__main__":
    mcp.run()


