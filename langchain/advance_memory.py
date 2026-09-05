from dotenv import load_dotenv
import langchain_openai
load_dotenv()

import uuid
from datetime import datetime
from sentence_transformers import SentenceTransformer
from langgraph.store.sqlite import SqliteStore
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send, interrupt, Command
from langgraph.checkpoint.sqlite import SqliteSaver

# Models & Tools
model = ChatOpenAI(model_name="gpt-4o", temperature=0.0)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

CONSOLIDATE_EVERY = 5          # consolidate after this many new episodes
RECALL_SIMILARITY_FLOOR = 0.35  # ignore memories below this relevance score

def embed_text(texts: list[str]) -> list[list[float]]:
    return embedder.encode(texts).tolist()

def episodic_ns(user_id : str) -> tuple:
    return ("semantic", user_id)

def semantic_ns(user_id : str) -> tuple:
    return ("semantic", user_id)

def remember_episode(store: SqliteStore, user_id: str, text: str) -> None:
    """Store a new episode in the episodic memory."""
    key = str(uuid.uuid4())
    store.put(episodic_ns(user_id), key, {
        "text": text,
        "timestamp": datetime.now().isoformat()
    })

def recall(store, namespace: tuple, query: str, k: int = 4, floor: float = 0.0) -> list[dict]:
    """Recall relevant memories based on a query."""
    results= store.search(namespace, query= query, limit =k)
    return [(r.value, r.score) for r in results if r.score >= floor]


def consolidate_if_needed(store, user_id):
    all_episode = store.search(episodic_ns(user_id), query="", limit=1000)

    if len((all_episode)) < CONSOLIDATE_EVERY:
        return

    episode_texts = "\n".join(f"-{e.value['text']}" for e in all_episode)
    facts_response = model.invoke(
        "Extract distinct, durable facts about the user from these conversation episodes. "
        "One fact per line, no commentary, no duplicates, present tense.\n\n"
        f"{episode_texts}"
    )

    for line in facts_response.content.strip().split("\n"):
        fact = line.strip("- ").strip()
        if fact:
            store.put(semantic_ns(user_id), str(uuid.uuid4()), {
                "text": fact, "timestamp": datetime.now().isoformat(),
            })


    for e in all_episode:
        store.delete(episodic_ns(user_id), e.key)


    print(f"Consolidated {len(all_episode)} episodes into semantic memory for user {user_id}.")


def chat_with_memory(store, user_id: str, user_input: str) -> str:
    episodic_hits = recall(store, episodic_ns(user_id), user_input, k=3, floor=RECALL_SIMILARITY_FLOOR)
    semantic_hits = recall(store, semantic_ns(user_id), user_input, k=5, floor=RECALL_SIMILARITY_FLOOR)

    episodic_block = "\n".join(f"- {t} (relevance {s:.2f})" for t, s in episodic_hits) or "(none above threshold)"
    semantic_block = "\n".join(f"- {t} (relevance {s:.2f})" for t, s in semantic_hits) or "(none above threshold)"

    prompt = (
        f"Known facts about this user (semantic memory):\n{semantic_block}\n\n"
        f"Relevant past conversation moments (episodic memory):\n{episodic_block}\n\n"
        f"Current message: {user_input}\n\n"
        "Respond naturally, using the above only where genuinely relevant."
    )
    response = model.invoke(prompt)

    remember_episode(store, user_id, f"User said: {user_input} | Assistant replied: {response.content}")
    consolidate_if_needed(store, user_id)

    return response.content

def main():
    with SqliteStore.from_conn_string(
        "memory.sqlite",
        index={"dims": 384, "embed": embed_text, "fields": ["text"]},
    ) as store:
        store.setup()

        user_id = input("User ID (any name, used to separate memories per person): ").strip() or "default_user"

        print("\nDual Memory Agent — episodic + semantic, with consolidation (type 'quit' to exit)")
        print("Close this and reopen later with the same User ID to test cross-session recall.\n")

        while True:
            user_input = input("You: ").strip()
            if user_input.lower() in ("quit", "exit"):
                print("Goodbye! Memories are saved in memory.sqlite.")
                break
            if not user_input:
                continue

            reply = chat_with_memory(store, user_id, user_input)
            print(f"Agent: {reply}\n")


if __name__ == "__main__":
    main()