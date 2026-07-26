from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.store.base import BaseStore


SYSTEM_PROMPT = (
    "You are ShopEase's support assistant. You help customers with orders, "
    "returns, shipping, and warranty questions.\n\n"
    "Known facts about this customer: {customer_facts}\n\n"
    "Use those facts naturally in your replies. Never ask the customer for "
    "information you already have on file. If you have no facts on file, "
    "treat them as a first-time customer and do not assume anything about "
    "them."
)

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder("history"),
    ]
)

_llm = get_llm()


def _read_customer_facts(store: BaseStore, user_id: str) -> str:
    item = store.get(("customer_facts", user_id), "profile")
    if item is None:
        return "none on file yet"
    return item.value["facts"]


def chat_node(state: MessagesState, config, *, store: BaseStore):
    user_id = config["configurable"]["user_id"]
    facts = _read_customer_facts(store, user_id)
    chain = PROMPT.partial(customer_facts=facts) | _llm
    reply = chain.invoke({"history": state["messages"]})
    return {"messages": [reply]}


def build_graph():
    """Compile a fresh graph with its own checkpointer and store.

    Returns (app, store) -- the store is returned separately so callers
    (CLI or Streamlit) can seed/inspect durable customer facts directly.
    """
    checkpointer = MemorySaver()
    store = InMemoryStore()

    graph = StateGraph(MessagesState)
    graph.add_node("chat", chat_node)
    graph.add_edge(START, "chat")

    app = graph.compile(checkpointer=checkpointer, store=store)
    return app, store


def seed_customer_facts(store: BaseStore, user_id: str, facts: str) -> None:
    """Write a durable, cross-conversation fact set for one customer.

    In production this would be called by a human agent's tool or a CRM
    sync job -- not by the chat turn itself.
    """
    store.put(("customer_facts", user_id), "profile", {"facts": facts})
