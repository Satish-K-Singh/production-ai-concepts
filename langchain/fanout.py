import warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
load_dotenv()

import time
import operator
from typing import Annotated
from typing_extensions import TypedDict
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

model = init_chat_model("openrouter:anthropic/claude-sonnet-4.5")


class ResearchState(TypedDict):
    topics: list[str]
    summaries: Annotated[list[str], operator.add]   
    durations: Annotated[list[float], operator.add] 
    final_report: str


def research_topic(state: dict) -> dict:

    topic = state["topic"]
    start = time.time()
    response = model.invoke(f"In 2 sentences, explain: {topic}")
    elapsed = time.time() - start
    return {
        "summaries": [f"**{topic}**: {response.content}"],
        "durations": [elapsed],
    }

def fan_out(state: ResearchState):
   
    return [Send("research_topic", {"topic": t}) for t in state["topics"]]

def compile_report(state: ResearchState) -> dict:
    return {"final_report": "\n\n".join(state["summaries"])}


builder = StateGraph(ResearchState)
builder.add_node("research_topic", research_topic)
builder.add_node("compile_report", compile_report)
builder.add_conditional_edges(START, fan_out, ["research_topic"])
builder.add_edge("research_topic", "compile_report")
builder.add_edge("compile_report", END)

graph = builder.compile()

def print_bar(done: int, total: int):
    width = 30
    filled = int(width * done / total)
    bar = "#" * filled + "-" * (width - filled)
    pct = int(100 * done / total)
    print(f"  [{bar}] {pct}%  ({done}/{total} workers done)")


print("Parallel Research Fan-Out Demo (type 'quit' to exit)\n")

while True:
    raw = input("Enter topics, comma-separated (any number): ").strip()

    if raw.lower() in ("quit", "exit"):
        print("Goodbye!")
        break

    if not raw:
        continue

    topics = [t.strip() for t in raw.split(",") if t.strip()]
    total = len(topics)

    print(f"\nFanning out to {total} parallel workers...\n")

    done = 0
    durations = []
    final_report = ""
    wall_start = time.time()


    for update in graph.stream(
        {"topics": topics, "summaries": [], "durations": [], "final_report": ""},
        stream_mode="updates",
    ):
        if "research_topic" in update:
            done += 1
            summary_line = update["research_topic"]["summaries"][0]
            worker_time = update["research_topic"]["durations"][0]
            durations.append(worker_time)
            topic_name = summary_line.split("**")[1]

            elapsed = time.time() - wall_start
            print(f"✅ Worker {done}/{total} finished: '{topic_name}'  ({worker_time:.1f}s, t+{elapsed:.1f}s)")
            print_bar(done, total)

        elif "compile_report" in update:
            final_report = update["compile_report"]["final_report"]

    wall_time = time.time() - wall_start
    sequential_estimate = sum(durations)

    print("\n" + "=" * 60)
    print("FINAL REPORT")
    print("=" * 60)
    print(final_report)
    print("=" * 60)
    print(f"Parallel wall-clock time : {wall_time:.1f}s")
    print(f"Sequential estimate      : {sequential_estimate:.1f}s  (sum of each worker's own time)")
    print(f"Speedup from fan-out     : {sequential_estimate / wall_time:.1f}x")
    print("=" * 60 + "\n")