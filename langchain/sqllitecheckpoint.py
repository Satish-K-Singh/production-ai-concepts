from dotenv import load_dotenv
load_dotenv()

import os
import sys
import time
import uuid
import json
import sqlite3
from typing import TypedDict, Any, Final, Sequence
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver

DB_PATH = "checkpoints.sqlite"
THREAD_FILE = "thread_id.txt"

class TaskState(TypedDict):
    """Represents the state of a task in the graph."""
    task_name : str
    steps : list[str]
    current_step : int
    steps_completed : list[str]


def show_progress(label : str, duration: float = 1.5):
    spinner = "|/-\\"
    end_time = time.time() + duration
    i = 0
    while time.time() < end_time:
        sys.stdout.write(f"\r  {spinner[i % len(spinner)]} {label}...")
        sys.stdout.flush()
        time.sleep(0.1)
        i += 1
    sys.stdout.write("\r" + " " * (len(label) + 15) + "\r")


def run_step(state:TaskState) -> dict:
    idx = state["current_step"]
    step_name = state["steps"][idx]
    show_progress(f"Running step {idx + 1}/{len(state['steps'])}: {step_name}")
    return {
        "steps_completed": state["steps_completed"] + [step_name],
        "current_step": idx + 1
    }

def route_text(state: TaskState) -> str:
    return "continue" if state["current_step"] < len(state["steps"]) else "done"

builder = StateGraph(TaskState)
builder.add_node("run_steps", run_step)
builder.add_edge(START, "run_steps")
builder.add_conditional_edges("run_steps", route_text, {
    "continue": "run_steps",
    "done": END
})  

conn = sqlite3.connect(DB_PATH, check_same_thread=False)
graph = builder.compile(checkpointer=SqliteSaver(conn))

def print_bar(done: int, total: int):
    width = 30
    filled = int(width * done / total)
    bar = "█" * filled + "░" * (width - filled)
    pct = int(100 * done / total)
    print(f"  [{bar}] {pct}%  ({done}/{total} steps)")

def start_new_run():
    task_name = input("Enter a name for the new task: ")
    if not task_name:
        print("Task name cannot be empty.")
        return

    raw_steps = input("Enter steps for the task, separated by commas: ")
    steps = [step.strip() for step in raw_steps.split(",") if step.strip()]
    if not steps:
        print("No valid steps provided.")
        return
    crash_after = input(
        f"Simulate a crash after which step? {steps} (or press Enter for no crash): "
    ).strip()
    thread_id = str(uuid.uuid4())
    with open(THREAD_FILE, "w") as f:
        json.dump({"thred_id": thread_id, "task_name": task_name}, f)

    config = {"config"}
        
   
