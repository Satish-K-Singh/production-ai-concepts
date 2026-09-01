from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

import operator
from typing import Annotated, Any, Callable, Dict, List, Optional, Tuple, Union
from typing_extensions import TypedDict
from pydantic import BaseModel, Field, create_model
from langchain.chat_models import ChatOpenAI
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
     response = model.invoke([prompt])
     return {"draft": response[0].content, "iteration": state["iteration"] + 1}

def critique(state: ReflectionState)-> dict:   


