"""Customer support triage workflow and execution script using LangGraph.

This module implements and tests a state graph that classifies customer 
support messages and routes them to the appropriate specialist for a response.
"""
import os
from typing import Annotated, Any, Final, Sequence, TypedDict

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph

#Immutable tuple for allowed categories of customer support messages
_CATEGORIES: Final[Sequence[str]] = (
    "order_status",
    "refund_policy",
    "shipping",
    "product_defect",
    "other",
)

class ChatState(TypedDict):
    """Represents the shared state of the support triage graph.

    Attributes:
        category: The classified category of the customer support message.
        message: A list of messages exchanged in the conversation.
    """
    category: str
    message: Annotated[list[BaseMessage], add_messages]


def _last_user_text(state: ChatState) -> str:
    """Extracts the text of the most recent user message from the state.

    Args:
        state: The current state of the conversation.

    Returns:
        The text of the last HumanMessage. Returns an empty string if no
        HumanMessage is found or if the message list is empty.
    """
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, HumanMessage) and isinstance(msg.content, str):
            return msg.content
    return ""

class SupportTriageWorkflow:
    """Workflow for classifying and responding to support tickets.

    Attributes:
        _llm: The language model used for classification and response generation.
    """
    def __init__(self, llm: BaseChatModel):
        """Initializes the SupportTriageWorkflow with a language model.

        Args:
            llm: An instance of a language model that implements BaseChatModel.
        """
        self._llm = llm

    def classify(self, state: ChatState) -> dict:
        """Classifies the incoming customer support message.

        Args:
            state: The current state of the conversation.

        Returns:
            A dictionary containing the classified category and a routing message.
        """
        sys = SystemMessage(content=(
            "Classify the ShopEase customer message as one of: " + ", ".join(_CATEGORIES) +
            ". Reply with the label only."
        ))
        label = self._llm.invoke([sys, HumanMessage(content=_last_user_text(state))]).content.strip().lower()
        if label not in _CATEGORIES:
            label = "other"
        return {"category": label, "messages": [AIMessage(content=f"(routing to: {label})")]}

    def respond(self, state: ChatState) -> dict:
        """Generates a response from the appropriate specialist based on the classified category.

        Args:
            state: The current state of the conversation.

        Returns:
            A dictionary containing the generated response message.
        """
        sys = SystemMessage(content=(
            f"You are ShopEase's {state['category']} specialist. Answer the customer helpfully."
        ))

        reply = self._llm.invoke([sys, HumanMessage(content=_last_user_text(state))]).content.strip()
        return {"messages": [reply]}

    def build_graph(self) -> StateGraph:
        """Constructs and wires the support triage graph.

        Returns:
            A configured StateGraph ready to be compiled.
        """
        builder = StateGraph(ChatState)
        
        builder.add_node("classify", self.classify)
        builder.add_node("respond", self.respond)
        
        builder.add_edge(START, "classify")
        builder.add_edge("classify", "respond")
        builder.add_edge("respond", END)
        
        return builder

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
        max_tokens=500,
    )

def stream_udpate(graph: CompiledStateGraph, message: ChatState) -> None:
    """Streams updates from the graph execution to the console.

    Args:
        graph: The compiled state graph to execute.
        state: The initial state for the graph execution.
    """
    inputs = {"messages": [HumanMessage(content=message)], "category": ""}

    for chunk in graph.stream(inputs, stream_mode = "update"):
        for node_name, delta in chunk.items():
            print(f"[{node_name} finished] {delta}")

def stream_tokens(graph: CompiledStateGraph, message: str) -> None:
    """Streams token-level updates from the graph execution to the console.

    Args:
        graph: The compiled state graph to execute.
        state: The initial state for the graph execution.
    """
    inputs = {"messages": [HumanMessage(content=message)], "category": ""}

    for token, meta in graph.stream(inputs, stream_mode="messages"):
        if meta.get("langgraph_node") == "respond" and token.content:
            print(token.content, end="", flush=True)
    print()

if __name__ == "__main__":
    # 1. Initialize the OpenAI LLM securely using the new helper
    llm = _initialize_llm()
    
    # 2. Build and compile the graph
    workflow = SupportTriageWorkflow(llm)
    app = workflow.build_graph().compile()
    
    # 3. Run the streaming tests
    msg = "My blender stopped working; I opened it 35 days ago. Can I get a refund?"
    print(f"CUSTOMER: {msg}\n")
    
    stream_udpate(app, msg)
    stream_tokens(app, msg)