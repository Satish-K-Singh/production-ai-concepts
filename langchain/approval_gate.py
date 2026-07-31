"""A LangGraph workflow demonstrating a human-in-the-loop approval gate.

This module sets up a state graph that proposes an action, pauses for human
approval, and then routes to execution or cancellation based on the decision.
"""

import uuid
from typing import TypedDict, Any, Final, Sequence

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command

from dotenv import load_dotenv
load_dotenv()

class ActionState(TypedDict):
    """Represents the shared state of the approval gate graph.

    Attributes:
        action: The action to be evaluated.
        decision: A string indicating the human decision (e.g., 'approve').
        result: A string providing the final outcome message.
    """
    action: str
    decision: str
    reason: str

def propose_action(state: ActionState) -> dict[str, str]:
    """Proposes an action based on the current state.

    Args:
        state: The current state of the approval process.

    Returns:
        A dictionary containing state updates.
    """
    return f"Proposed action: {state['action']}"

def human_approval(state: ActionState) -> dict[str, str]:
    """Simulates human approval or rejection of the proposed action.

    Args:
        state: The current state of the approval process.

    Returns:
        A dictionary containing the human's decision update.
    """
    decision = interrupt({
        "question": "Approve this action",
        "Action": state["action"],
    })
    return {"decision" : decision}

def execute_action(state: ActionState) -> dict[str, str]:
    """Executes the action if approved.

    Args:
        state: The current state of the approval process.

    Returns:
        A dictionary containing the result of the action execution.
    """
    return {"result": f"Executed: {state['action']}"}

def cancel_action(state: ActionState) -> dict[str, str]:
    """Cancels the action if not approved.

    Args:
        state: The current state of the approval process.

    Returns:
        A dictionary containing the cancellation message.
    """
    return {"result": f"Action '{state['action']}' was canceled."}

def route_decision(state: ActionState) -> str:
    """Routes the action based on the approval decision.

    Args:
        state: The current state of the approval process.

    Returns:
        A string indicating the next node to route to.
    """
    return execute_action(state) if state["decision"] == "approved" else cancel_action(state)

def build_graph() -> Any:
    """Builds and compiles the LangGraph workflow.

    Returns:
        A compiled LangGraph StateGraph runnable.
    """
    builder = StateGraph(ActionState)
    
    builder.add_node("propose", propose_action)
    builder.add_node("human_approval", human_approval)
    builder.add_node("execute_action", execute_action)
    builder.add_node("cancel_action", cancel_action)

    builder.add_edge(START, "propose")
    builder.add_edge("propose", "human_approval")
    builder.add_conditional_edges(
        "human_approval", 
        route_decision,
        {
            "execute": "execute_action",
            "cancel": "cancel_action",
        },
    )
    builder.add_edge("execute_action", END)
    builder.add_edge("cancel_action", END)

    return builder.compile(checkpointer=InMemorySaver())


def main() -> None:
    """Runs the interactive Human Approval Gate demo loop."""
    load_dotenv()
    
    graph = build_graph()

    print("Human Approval Gate Demo (type 'quit' to exit)")
    print("Describe ANY action - it will pause for your approval.\n")

    while True:
        action = input("Action that needs approval: ").strip()

        if action.lower() in ("quit", "exit"):
            print("Goodbye!")
            break

        if not action:
            continue

        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}

        # Initialize the workflow
        result = graph.invoke(
            {"action": action, "decision": "", "result": ""},
            config,
        )

        # Handle the interrupt payload if the workflow paused
        if "__interrupt__" in result:
            payload = result["__interrupt__"][0].value
            
            print("\n" + "=" * 60)
            print("HUMAN APPROVAL NEEDED")
            print("=" * 60)
            print(f"Proposed action: {payload['action']}")
            print("=" * 60)

            answer = input("Type 'approve' or 'reject': ").strip().lower()
            
            # Resume graph execution with the human's command
            result = graph.invoke(Command(resume=answer), config)

        print("\n" + "=" * 60)
        print("FINAL RESULT")
        print("=" * 60)
        print(result.get("result", "No result returned."))
        print("=" * 60 + "\n")


if __name__ == "__main__":
    main()