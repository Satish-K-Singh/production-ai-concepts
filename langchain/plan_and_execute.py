from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

import operator
import sqlite3
from typing import Dict, Union, Annotated
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send, interrupt, Command
from langchain.agents import create_agent, Tool
from langchain.agents.middleware import ToolRetryMiddleware
from langchain_tavily import TavilySearch
from langgraph.checkpoint.sqlite import SqliteSaver

model = ChatOpenAI(model_name="gpt-4", temperature=0.0, max_tokens=2000)

executor = create_agent(
    model = model,
    tools = [TavilySearch(max_results=3)],
    system_prompt = "Execute exactly the one task given to you. Be concise and factual.",
    middleware = [ToolRetryMiddleware(max_retries=3, backoff_factor=2, onfailure="continue")],
)

class Step(BaseModel):
    id : str = Field(description="Unique identifier for the step.")
    task : str = Field(description="The task to be executed in this step.")
    depends_on : list[str] = Field(default_factory=list, description="IDs of steps that must finish first.")
    sensitive : bool = Field(default=False, description="True if this step needs human approval before running.")

class Plan(BaseModel):
    steps: list[Step]

class Response(BaseModel):
    response : str
    confidence : float = Field(ge=0.0, le=1.0, description="0-1 confidence this fully answers the objective.")

class Act(BaseModel):
    action : Union[Response, Plan]

planner_model = model.with_structured_output(Plan)
replanner_model = model.with_structured_output(Act)

class PlanExecuteState(TypedDict):
    objective : str
    plan : list[Dict]
    completed_ids : Annotated[list[str], operator.add]
    past_steps: Annotated[list[tuple], operator.add]
    respone : str
    confidence : float

def plan_step(state: PlanExecuteState) -> dict:
    plan = planner_model.invoke(
        f"Break this objective into an ordered set of concrete steps. "
        f"Mark depends_on for any step that needs another step's result first. "
        f"Mark sensitive=true only for steps involving irreversible or risky actions.\n\n"
        f"Objective: {state['objective']}"

    )
    return {"plan": [s.model_dump() for s in plan.steps]}

def ready_steps(state : PlanExecuteState) -> list[Step]:
    """Steps whose dependencies are all already completed, and not yet done."""
    done= set(state["completed_ids"])
    return [
        s for s in state["plan"] if s["id"] not in done and set(s["depends_on"]).issubset(done)
    ]

