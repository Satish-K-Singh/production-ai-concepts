"""A command-line research assistant agent using LangChain.

This script initializes an LLM agent equipped with web search and
Python execution tools to answer user queries interactively.
"""

import sys
from typing import Any, Dict
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_experimental.tools import PythonREPLTool
from langchain_tavily import TavilySearch

_MODEL_NAME = "gpt-4o"
_MAX_SEARCH_RESULTS = 5
_SYSTEM_PROMPT = (
    "You are a research assistant. Use web search to find current "
    "information, and use the code tool to perform any calculations "
    "or data processing needed to answer the question."
)
_SEPARATOR = "=" * 60

def create_search_agent()->any:
    """Initializes and returns the research agent with necessary tools.
    
    Returns:
        An initialized LangChain agent.
    """
    search_tool = TavilySearch(max_results=_MAX_SEARCH_RESULTS, topic="general")
    code_tool = PythonREPLTool()
    tools = [search_tool, code_tool]
    agent = create_agent(
        model_name=_MODEL_NAME,
        tools=tools,
        system_prompt=_SYSTEM_PROMPT,
    )
    return agent

def main() -> None:
    """Main function to run the interactive research assistant."""
    load_dotenv()  

    try:
        agent = create_search_agent()

    except Exception as e:
        print(f"Error initializing the agent: {e}")
        sys.exit(1)

    while True:
        question = input("Ask a question: ").strip()

        if question.lower() in ("quit", "exit"):
            print("Goodbye!")
            break

        if not question:
            continue

        print("\nWorking on it...\n")

        result = agent.invoke({"messages": [{"role": "user", "content": question}]})

        print("="*60)
        print("FINAL ANSWER")
        print("="*60)
        print(result["messages"][-1].content)
        print("="*60 + "\n")
