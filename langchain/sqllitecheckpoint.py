"""A task execution graph utilizing LangGraph and SQLite checkpointing.

This script allows users to define a series of steps, execute them sequentially,
simulate crashes, and resume from the last saved state using a SQLite backend.
"""

import argparse
import json
import os
import sqlite3
import sys
import time
import uuid
from typing import Any, Dict, List, TypedDict

from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

# Load environment variables
load_dotenv()

# Constants
DB_PATH = "checkpoints.sqlite"
THREAD_FILE = "thread_id.txt"


class TaskState(TypedDict):
    """Represents the state of a task in the execution graph.

    Attributes:
        task_name: The name of the current task.
        steps: A list of step descriptions to be executed.
        current_step: The index of the step currently being executed.
        steps_completed: A list of step descriptions that have finished.
    """
    task_name: str
    steps: List[str]
    current_step: int
    steps_completed: List[str]


def show_progress(label: str, duration: float = 1.5) -> None:
    """Displays a CLI spinner to simulate work being done.

    Args:
        label: The text to display next to the spinner.
        duration: The number of seconds to run the spinner.
    """
    spinner = "|/-\\"
    end_time = time.time() + duration
    i = 0
    while time.time() < end_time:
        sys.stdout.write(f"\r  {spinner[i % len(spinner)]} {label}...")
        sys.stdout.flush()
        time.sleep(0.1)
        i += 1
    # Clear the spinner line
    sys.stdout.write("\r" + " " * (len(label) + 15) + "\r")


def run_step(state: TaskState) -> Dict[str, Any]:
    """Executes a single step in the task graph.

    Args:
        state: The current state of the task graph.

    Returns:
        A dictionary containing state updates to be merged.
    """
    idx = state["current_step"]
    step_name = state["steps"][idx]
    
    show_progress(f"Running step {idx + 1}/{len(state['steps'])}: {step_name}")
    
    return {
        "steps_completed": state["steps_completed"] + [step_name],
        "current_step": idx + 1
    }


def route_text(state: TaskState) -> str:
    """Determines the next node to transition to in the graph.

    Args:
        state: The current state of the task graph.

    Returns:
        The name of the next edge ('continue' or 'done').
    """
    return "continue" if state["current_step"] < len(state["steps"]) else "done"


def print_bar(done: int, total: int) -> None:
    """Prints a progress bar to standard output.

    Args:
        done: The number of completed items.
        total: The total number of items.
    """
    if total == 0:
        return
        
    width = 30
    filled = int(width * done / total)
    bar = "█" * filled + "░" * (width - filled)
    pct = int(100 * done / total)
    print(f"  [{bar}] {pct}%  ({done}/{total} steps)")


class TaskManager:
    """Manages the creation, execution, and resumption of task graphs."""

    def __init__(self, db_path: str, thread_file: str) -> None:
        """Initializes the TaskManager with database and file paths.

        Args:
            db_path: Path to the SQLite database file.
            thread_file: Path to the text file storing the current thread ID.
        """
        self.db_path = db_path
        self.thread_file = thread_file
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.graph = self._build_graph()

    def _build_graph(self) -> CompiledStateGraph:
        """Builds and compiles the state graph.

        Returns:
            A compiled StateGraph instance with a SQLite checkpointer.
        """
        builder = StateGraph(TaskState)
        builder.add_node("run_steps", run_step)
        builder.add_edge(START, "run_steps")
        builder.add_conditional_edges(
            "run_steps", 
            route_text, 
            {
                "continue": "run_steps",
                "done": END
            }
        )
        return builder.compile(checkpointer=SqliteSaver(self.conn))

    def start_new_run(self) -> None:
        """Prompts the user for a new task, builds it, and begins execution."""
        task_name = input("Enter a name for the new task: ").strip()
        if not task_name:
            print("Error: Task name cannot be empty.")
            return

        raw_steps = input("Enter steps for the task, separated by commas: ")
        steps = [step.strip() for step in raw_steps.split(",") if step.strip()]
        if not steps:
            print("Error: No valid steps provided.")
            return
            
        crash_after = input(
            f"Simulate a crash after which step? {steps} "
            "(or press Enter for no crash): "
        ).strip()
        
        thread_id = str(uuid.uuid4())
        
        # Save thread metadata
        try:
            with open(self.thread_file, "w") as f:
                json.dump({"thread_id": thread_id, "task_name": task_name}, f)
        except OSError as e:
            print(f"Error writing to {self.thread_file}: {e}")
            return

        config = {"configurable": {"thread_id": thread_id}}
        initial_state = {
            "task_name": task_name,
            "steps": steps,
            "current_step": 0,
            "steps_completed": []
        }
        
        print(f"\nStarting '{task_name}' (thread_id={thread_id})\n")
        
        for update in self.graph.stream(initial_state, config=config, stream_mode="updates"):
            completed = update["run_steps"]["steps_completed"][-1]
            done_count = len(update["run_steps"]["steps_completed"])
            
            print(f"✅ Checkpoint saved after: {completed}")
            print_bar(done_count, len(steps))

            if crash_after and completed == crash_after:
                print(f"\n💥 Simulating crash after step: {completed}\n")
                sys.exit(1)

        print(f"\nTask '{task_name}' finished with no crash. Steps done: {steps}")

    def resume_run(self) -> None:
        """Resumes a previously halted task using the saved thread ID."""
        if not os.path.exists(self.thread_file):
            print("No previous run found. Run with 'start' first.")
            return

        try:
            with open(self.thread_file, "r") as f:
                saved = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            print(f"Error reading {self.thread_file}: {e}")
            return

        thread_id = saved.get("thread_id")
        task_name = saved.get("task_name", "Unknown Task")
        
        if not thread_id:
            print("Error: Invalid thread file format.")
            return

        config = {"configurable": {"thread_id": thread_id}}
        print(f"\nResuming '{task_name}' (thread_id={thread_id})\n")

        state = self.graph.get_state(config)
        if not state or not hasattr(state, 'values') or not state.values:
             print("Could not find a valid checkpoint in the database.")
             return

        already_done = state.values["steps_completed"]
        total_steps = len(state.values["steps"])
        
        print(f"Recovered from SQLite — already completed: {already_done}")
        print_bar(len(already_done), total_steps)
        print()

        # Resume streaming from checkpoint
        for update in self.graph.stream(None, config, stream_mode="updates"):
            completed = update["run_steps"]["steps_completed"][-1]
            done_count = len(update["run_steps"]["steps_completed"])
            
            print(f"✅ Checkpoint saved after: {completed}")
            print_bar(done_count, total_steps)

        final_state = self.graph.get_state(config)
        print(f"\nAll steps completed: {final_state.values['steps_completed']}")

    def close(self) -> None:
        """Closes the underlying SQLite connection."""
        if self.conn:
            self.conn.close()


def main() -> None:
    """Parses command-line arguments and routes to the correct operation."""
    parser = argparse.ArgumentParser(
        description="Run or resume LangGraph checkpoint tasks."
    )
    parser.add_argument(
        "mode",
        choices=["start", "resume"],
        nargs="?",
        default="start",
        help="Whether to start a new graph run or resume an existing one."
    )
    
    args = parser.parse_args()

    manager = TaskManager(DB_PATH, THREAD_FILE)
    try:
        if args.mode == "start":
            manager.start_new_run()
        elif args.mode == "resume":
            manager.resume_run()
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
    finally:
        manager.close()


if __name__ == "__main__":
    main()