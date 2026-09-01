from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

import operator
from typing import Annotated, Any, Callable, Dict, List, Optional, Tuple, Union
from typing_extensions import TypedDict
from pydantic import BaseModel, Field, create_model
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

model = ChatOpenAI(model_name="gpt-4", temperature=0.0, max_tokens=2000)

class Critique(BaseModel):
     """Structured, multi-dimensional grading of a draft."""
     clarity: int = Field(ge=1,le=10,description="How clear and readable is it?")
     accuracy: int = Field(ge=1,le=10,description="How factually sound is it?")
     completeness: int = Field(ge=1,le=10,description="Does it fully address the task?")
     depth: int = Field(ge=1,le=10,description="How deep and insightful is it?")
     issues: list[str] = Field(description="Specific, actionable problems found. Empty if none.")

     @property
     def overall_score(self) -> float:
          return round((self.clarity + self.accuracy + self.completeness + self.depth) / 4, 2)

critical_model = model.with_structured_output(Critique)

class HistoryEntry(TypedDict):
     iteration: int
     draft: str
     scores: dict
     overall_score: float

class ReflectionState(TypedDict):
     task: str
     draft : str
     iteration: int
     max_iterations: int
     pass_threshold: float
     history: Annotated[List[HistoryEntry], operator.add]
     stop_reason : str



def generate(state: ReflectionState)-> dict:
     if state["iteration"] == 0:
          prompt = f"Write a respone to this task:\n\n {state['task']}"
     else:
          last = state["history"][-1]
          issue_text = "\n".join([f"- {issue}" for issue in last["scores"]["issues"]])
          prompt = (
               f"Original task : {state['task']}\n\n"
               f"Your previous draft:\n{last['draft']}\n\n"
               f"Feedback from previous iteration:\n{issue_text}"
               "Revise your draft to address every issue above. Do not reintroduce old problems."
        )
     response = model.invoke(prompt)
     return {"draft": response.content, "iteration": state["iteration"] + 1}

def critique(state: ReflectionState)-> dict:
     result = critical_model.invoke(
          f"Task give to the writer:\n{state['task']}\n\n"
          f"Their draft:\n{state['draft']}\n\n"
          "Score this draft honestly on all four dimensions. List every real issue, "
          "however minor. Do not inflate scores to be polite."
     )   
     entry: HistoryEntry = {
          "iteration": state["iteration"],
          "draft": state["draft"],
          "scores": result.dict(),
          "overall_score": result.overall_score
     }
     return {"history": state["history"] + [entry]}

def route_after_critique(state: ReflectionState)-> str:
     history = state["history"]
     latest = history[-1]
     if latest["overall_score"] >= state["pass_threshold"]:
          return "done_threshold"
     if state["iteration"] >= state["max_iterations"]:
          return "done_maxed"
     if len(history) >=2:
         improvement = latest["overall_score"] - history[-2]["overall_score"]
         if improvement < 0.2:
             return "done_plateau"

     return "revise"


def finalize(state: ReflectionState)-> dict:
     return {}

builder = StateGraph(ReflectionState)
builder.add_node("generate", generate)
builder.add_node("critique", critique)

builder.add_edge(START, "generate")
builder.add_edge("generate", "critique")
builder.add_conditional_edges(
     "critique", route_after_critique, {
          "done_threshold": END,
          "done_maxed": END,
          "done_plateau": END,
          "revise": "generate"     
     }, 
)

graph = builder.compile()

def print_score_bar(label: str, score: int):
    filled = "#" * score
    empty = "-" * (10 - score)
    print(f"  {label:14s} [{filled}{empty}] {score}/10")


print("Advanced Reflection Loop — structured, multi-dimensional (type 'quit' to exit)\n")

while True:
    task = input("What should the agent write? ").strip()
    if task.lower() in ("quit", "exit"):
        print("Goodbye!")
        break
    if not task:
        continue

    try:
        max_iter = int(input("Max revision rounds (e.g. 4): ").strip() or "4")
    except ValueError:
        max_iter = 4
    try:
        threshold = float(input("Pass threshold out of 10 (e.g. 8.5): ").strip() or "8.5")
    except ValueError:
        threshold = 8.5

    result = graph.invoke({
        "task": task, "draft": "", "iteration": 0,
        "max_iterations": max_iter, "pass_threshold": threshold,
        "history": [], "stop_reason": "",
    })

    print("\n" + "=" * 65)
    print("REVISION HISTORY")
    print("=" * 65)
    for entry in result["history"]:
        print(f"\n--- Round {entry['iteration']} (overall: {entry['overall_score']}/10) ---")
        s = entry["scores"]
        print_score_bar("Clarity", s["clarity"])
        print_score_bar("Accuracy", s["accuracy"])
        print_score_bar("Completeness", s["completeness"])
        print_score_bar("Depth", s["depth"])
        if s["issues"]:
            print("  Issues found:")
            for issue in s["issues"]:
                print(f"    • {issue}")

    print("\n" + "=" * 65)
    print("FINAL DRAFT")
    print("=" * 65)
    print(result["draft"])
    print("=" * 65 + "\n")

