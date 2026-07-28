"""Customer support triage graph using LangChain and LangGraph.

This module defines a state graph that classifies incoming customer support
messages and routes them to specialized AI personas for handling.
"""

import os
from typing import Any, Callable, Dict, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

#Constants and Personas
CATEGORIES = ["order_status", "refund_policy", "shipping", "product_defect", "other"]

SAMPLES = [
    "Where is my order SE-4471? It was due Tuesday.",
    "My blender stopped working; I opened it 35 days ago. Can I get a refund?",
    "The package arrived soaked and the box was crushed.",
]

REFUND_PERSONA = (
    "You are ShopEase's Refund & Returns specialist. Apply this policy exactly:\n"
    "1) Full refund within 30 days of delivery.\n"
    "2) Day 31-45: store credit only, and only if the item is unopened.\n"
    "3) After day 45: no refund unless damaged/defective (report within 90 days).\n"
    "4) Opened electronics: returnable only within 30 days AND only if defective. "
    "Opened electronics past day 30 are never eligible.\n"
    "5) Final-sale / clearance items are never returnable.\n"
    "Give the customer a clear verdict and name the rule you applied."
)

ORDER_PERSONA = (
    "You are ShopEase's order-status specialist. Help the customer locate their "
    "order. Be concise and friendly; if you need an order ID, ask for it."
)

SHIPPING_PERSONA = (
    "You are ShopEase's shipping specialist. Address delivery delays, tracking, "
    "and carrier issues. Be empathetic and concrete."
)

DEFECT_PERSONA = (
    "You are ShopEase's product-defect specialist. Handle damaged or defective "
    "items and apply the 90-day damaged/defective reporting window."
)

GENERAL_PERSONA = (
    "You are a friendly ShopEase support agent. Help with anything that does not "
    "fit a specific category."
)

PERSONAS = {
    "order_status": ORDER_PERSONA,
    "refund_policy": REFUND_PERSONA,
    "shipping": SHIPPING_PERSONA,
    "product_defect": DEFECT_PERSONA,
    "other": GENERAL_PERSONA,
}

#Types and States
class TriageState(TypedDict):
    """Represents the shared state of the support triage graph."""
    category: str
    message: str
    reply: str

#Initialization of the LLM and loading the OpenAI API key
def _initialize_llm() -> ChatOpenAI:
    """Initializes and returns the ChatOpenAI instance.

    Returns:
        An instantiated ChatOpenAI model.

    Raises:
        ValueError: If OPENAI_API_KEY is not found in the environment variables.
    """
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not found in environment variables. "
            "Please create a .env file with your OpenAI API key."
        )
    
    return ChatOpenAI(
        model="gpt-4o",
        api_key=api_key,
        temperature=0.0,
        max_tokens=500
    )

# Module-level instances
try:
    llm = _initialize_llm()
except ValueError as error:
    print(f"Initialization Error: {error}")
    llm = None  

#Graph and routing functions
def classify(state: TriageState) -> Dict[str, str]:
    """Classifies the incoming message into exactly one support category.
    
    Args:
        state: The current graph state containing the customer message.
        
    Returns:
        A dictionary updating the 'category' state.
    """
    sys = SystemMessage(content=(
        "You are a support-triage classifier for ShopEase. Read the customer "
        "message and reply with EXACTLY ONE label from: " + ", ".join(CATEGORIES) +
        ". Reply with the label only -- no punctuation, no explanation."
    ))
    out = llm.invoke([sys, HumanMessage(content=state["message"])])
    label = out.content.strip().lower()
    return {"category": label if label in CATEGORIES else "other"}

def make_specialist(persona: str)-> Callable[[TriageState], Dict[str, str]]:
    """Returns a reusable specialist node bound to a system persona.
    
    Args:
        persona: The system prompt guiding the specialist's behavior.
        
    Returns:
        A function representing a LangGraph node.
    """
    def specialist(state: TriageState) -> dict:
        """Processes the message using the bound persona."""
        out = llm.invoke([SystemMessage(content=persona),
                          HumanMessage(content=state["message"])])
        return {"reply": out.content.strip()}
    return specialist

def route(state: TriageState) -> str:
    """Routes the message to the appropriate specialist based on its category.
    
    Args:
        state: The current graph state.
        
    Returns:
        The dictionary key for the next node to execute.
    """
    return state["category"]

def build_graph() -> Any:
    """Builds and compiles the triage state graph.
    
    Returns:
        A compiled LangGraph StateGraph ready for invocation.
    """
    graph = StateGraph(TriageState)
    graph.add_node("classify", classify)

    for category in CATEGORIES:
        graph.add_node(category, make_specialist(PERSONAS[category]))

    graph.add_edge(START, "classify")

    graph.add_conditional_edges("classify", route, {category: category for category in CATEGORIES})

    for category in CATEGORIES:
        graph.add_edge(category, END)

    return graph.compile()

graph = build_graph() if llm else None

#Executing and Testing the graph
def run(message: str) -> Dict[str, str]:
    """Runs the compiled LangGraph with a given message.
    
    Args:
        message: The customer support message to process.
        
    Returns:
        The final state of the graph after processing.
    """
    if not graph:
        raise RuntimeError("Graph is not initialized due to missing LLM.")
    
    return graph.invoke({"message": message, "category": "", "reply": ""})

def test_classify_returns_valid_category() -> None:
    """Tests that the classify node always returns a valid category."""
    out = classify({""
            "message": "I want my money back for a broken blender",
            "category": "", 
            "reply": ""
        })
    assert out["category"] in CATEGORIES, "Invalid category returned."

def test_refund_specialist_produces_reply():
    """Tests that a specialist node produces a valid string reply."""
    specialist = make_specialist(REFUND_PERSONA)
    out = specialist({
        "message": "Opened blender, day 35, is it refundable?",
        "category": "refund_policy", 
        "reply": ""})
    assert isinstance(out["reply"], str) and out["reply"],"Empty reply generated."

def main() -> None:
    """Main execution function to run examples and tests."""
    if not llm:
        print("Cannot run examples. Fix environment variables and try again.")
        return

    print("Initializing LLM with OpenAI...\n")

    for msg in SAMPLES:
        result = run(msg)
        print("=" * 72)
        print(f"CUSTOMER : {msg}")
        print(f"CATEGORY : {result['category']}")
        print(f"REPLY    : {result['reply']}\n")

    print("Running tests...")
    test_classify_returns_valid_category()
    test_refund_specialist_produces_reply()
    print("All tests passed.")


if __name__ == "__main__":
    main()