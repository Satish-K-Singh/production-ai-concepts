"""A SQL database agent using LangChain to query SQLite databases interactively.

This module initializes an LLM agent equipped with tools to list tables,
view schemas, and execute queries against a local SQLite database.
"""
import sqlite3
import sys
from contextlib import closing
from typing import Any, Dict

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain.agents.middleware import ToolRetryMiddleware

# Constant
_DB_PATH = "Test.db"
_MODEL_NAME = "gpt-4o"
_SYSTEM_PROMPT = """
You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct sqlite query to run,
then look at the results and return the answer. Always limit your query to
at most 5 results unless the user asks for more.

You MUST look at the tables in the database first, then check the schema of
the relevant tables, before writing a query.

DO NOT make any INSERT, UPDATE, DELETE, or DROP statements.
"""

#Tools

@tool
def sql_db_list_tables() -> str:
    """Lists all tables in the SQLite database."""
    try:
        conn = sqlite3.connect(_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall() if not row[0].startswith("sqlite_")]
        conn.close()
        return ", ".join(tables)
    except Exception as e:
        return f"Error listing tables: {e}"

@tool
def sql_db_schema(table_name: str) ->str:
    """Returns the schema for a comma-separated list of table names.
    
    Args:
        table_names: A comma-separated string of table names.
        
    Returns:
        The SQL schema definitions or an error message.
    """
    con = sqlite3.connect("Test.db")  
    cursor = con.cursor()
    result = []
    for table in table_name.split(","):
        table = table.strip()
        cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name=?;", (table,))
        row = cursor.fetchone()
        if row:
            result.append(row[0])

    con.close()
    return "\n".join(result)

@tool
def sql_db_query(query: str) -> str:
    """Executes a SQL query against the database and returns the results.
    
    Args:
        query: The SQLite query to execute.
        
    Returns:
        A string representation of the query results or an error message.
    """
    try:
        with closing(sqlite3.connect(_DB_PATH)) as con:
            cursor = con.cursor()
            cursor.execute(query)
            rows = cursor.fetchall()
            return str(rows)
    except Exception as e:
        return f"Error executing query: {e}"

def create_sql_agent() -> Any:
    """Initializes and returns the SQL database agent with necessary tools.
    
    Returns:
        An initialized LangChain agent.
    """
    tools = [sql_db_list_tables, sql_db_schema, sql_db_query]
    retry_middleware = ToolRetryMiddleware(
        max_retries=3,
        backoff_factor=2.0,
        initial_delay=1.0,
        on_failure="continue",
    )
    agent = create_agent(
        model_name=_MODEL_NAME,
        tools=tools,
        system_prompt=_SYSTEM_PROMPT,
        middleware=[retry_middleware]
    )
    return agent


