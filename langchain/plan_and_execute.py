import operator
import sqlite3
import uuid
from typing import Annotated, Dict, Union
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send, interrupt, Command
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.sqlite import SqliteSaver

load_dotenv()

# Models & Tools
model = ChatOpenAI(model_name="gpt-4o", temperature=0.0)
tools = [TavilySearchResults(max_results=3)]
executor = create_react_agent(model=model, tools=tools)

# Pydantic Schemas
class Step(BaseModel):
    id: str = Field(description="Unique identifier for the step.")
    task: str = Field(description="The task to be executed in this step.")
    depends_on: list[str] = Field(default_factory=list, description="IDs of steps that must finish first.")
    sensitive: bool = Field(default=False, description="True if this step needs human approval before running.")

class Plan(BaseModel):
    steps: list[Step]

class Act(BaseModel):
    """Return either a final response or a new plan."""
    response: str | None = Field(default=None, description="Final response answering the objective.")
    confidence: float | None = Field(default=None, ge=0.0, le=1.0, description="0-1 confidence score.")
    steps: list[Step] | None = Field(default=None, description="Next planned steps if not yet finished.")

planner_model = model.with_structured_output(Plan)
replanner_model = model.with_structured_output(Act)

# States
class PlanExecuteState(TypedDict):
    objective: str
    plan: list[Dict]
    completed_ids: Annotated[list[str], operator.add]
    past_steps: Annotated[list[tuple], operator.add]
    response: str
    confidence: float

class ExecuteStepState(TypedDict):
    step: Dict

# Graph Nodes
def plan_step(state: PlanExecuteState) -> dict:
    plan = planner_model.invoke(
        f"Break this objective into an ordered set of concrete steps. "
        f"Mark depends_on for any step that needs another step's result first. "
        f"Mark sensitive=true only for steps involving irreversible or risky actions.\n\n"
        f"Objective: {state['objective']}"
    )
    return {"plan": [s.model_dump() for s in plan.steps]}

def ready_steps(state: PlanExecuteState) -> list[dict]:
    done = set(state.get("completed_ids", []))
    return [
        s for s in state.get("plan", []) 
        if s["id"] not in done and set(s.get("depends_on", [])).issubset(done)
    ]

def fan_out_ready_steps(state: PlanExecuteState):
    steps = ready_steps(state)
    if not steps:
        return "replan"
    return [Send("execute_step", {"step": s}) for s in steps]

def execute_step(state: ExecuteStepState) -> dict:
    step = state["step"]

    if step.get("sensitive", False):
        decision = interrupt({
            "question": "Approve this sensitive step?",
            "step_id": step["id"],
            "step_task": step["task"]
        })
        if decision != "approve":
            return {
                "completed_ids": [step["id"]],
                "past_steps": [(step["task"], "SKIPPED — rejected by user.")],
            }

    result = executor.invoke({"messages": [{"role": "user", "content": step["task"]}]})
    output = result["messages"][-1].content
    return {
        "completed_ids": [step["id"]],
        "past_steps": [(step["task"], output)],
    }

def replan_step(state: PlanExecuteState) -> dict:
    summary = "\n".join(f"{task} => {result}" for task, result in state.get("past_steps", []))
    remaining_steps = [s for s in state.get("plan", []) if s["id"] not in state.get("completed_ids", [])]

    act: Act = replanner_model.invoke(
        f"Objective: {state['objective']}\n\n"
        f"Completed so far:\n{summary}\n\n"
        f"Remaining planned steps: {remaining_steps}\n\n"
        "If the objective is now fully satisfied, provide a final response and confidence score. "
        "Otherwise provide the updated list of steps."
    )

    if act.response:
        return {
            "response": act.response,
            "confidence": act.confidence or 1.0,
        }

    return {
        "plan": [s.model_dump() for s in (act.steps or [])],
    }

def route_after_replan(state: PlanExecuteState):
    if state.get("response"):
        return END
    return fan_out_ready_steps(state)

# Graph Assembly
builder = StateGraph(PlanExecuteState)
builder.add_node("planner", plan_step)
builder.add_node("execute_step", execute_step)
builder.add_node("replan", replan_step)

builder.add_edge(START, "planner")
builder.add_conditional_edges("planner", fan_out_ready_steps, ["execute_step", "replan"])
builder.add_edge("execute_step", "replan")
builder.add_conditional_edges("replan", route_after_replan, ["execute_step", "replan", END])

conn = sqlite3.connect("plan_checkpoints.sqlite", check_same_thread=False)
graph = builder.compile(checkpointer=SqliteSaver(conn))

# CLI Runner
if __name__ == "__main__":
    print("Advanced Plan-and-Execute Agent (type 'quit' to exit)\n")

    while True:
        objective = input("Enter an objective to achieve: ").strip()
        if objective.lower() == "quit":
            break
        if not objective:
            continue

        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 25}

        result = graph.invoke({
            "objective": objective,
            "plan": [],
            "completed_ids": [],
            "past_steps": [],
            "response": "",
            "confidence": 0.0
        }, config)

        # Handle Human-in-the-loop interrupts
        while "__interrupt__" in result:
            payload = result["__interrupt__"][0].value
            print(f"\n[APPROVAL NEEDED] Step '{payload['step_id']}': {payload['step_task']}")
            answer = input("Type 'approve' or 'reject': ").strip().lower()
            result = graph.invoke(Command(resume=answer), config)

        print("\n" + "=" * 65)
        print("STEPS TAKEN")
        print("=" * 65)
        for task, output in result.get("past_steps", []):
            print(f"• {task}\n  → {output}\n")

        print("=" * 65)
        print(f"FINAL RESPONSE (confidence: {result.get('confidence', 0.0):.0%})")
        print("=" * 65)
        print(result.get("response"))
        print("=" * 65 + "\n")