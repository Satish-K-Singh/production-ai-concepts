import warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
load_dotenv()

from typing_extensions import TypedDict
from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

# Models & Tools
model = ChatOpenAI(model_name="gpt-4o", temperature=0.0)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
web_search = TavilySearch(max_results=3)

KNOWLEDGE_BASE = [
    "As of our last internal review, Python 3.11 was the recommended version for our backend services.",
    "Our deployment pipeline uses Docker and Kubernetes for container orchestration.",
    "The company's data retention policy requires deleting user logs after 90 days.",
    "Our internal style guide requires type hints on all new Python functions.",
    "The engineering team holds a retrospective every two weeks on Fridays.",
]

docs = [Document(page_content=text) for text in KNOWLEDGE_BASE]
vectorstore = Chroma.from_documents(docs, embeddings, persist_directory="./chroma_crag")
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

class ChunkVerdict(BaseModel):
    verdict: Literal["correct", "ambiguous", "incorrect"]
    reason: str

class RefinedKnowledge(BaseModel):
    relevant_sentences: list[str] = Field(
        description="Only the specific sentences from the chunk that are actually relevant."
    )

verdict_model = model.with_structured_output(ChunkVerdict)
refine_model = model.with_structured_output(RefinedKnowledge)

class CRAGState(TypedDict):
    question: str
    local_docs: list[str]
    verdicts: list[str]
    refined_knowledge: list[str]
    web_results: list[str]
    used_web_fallback: bool
    answer: str

def retrieve_local(state: CRAGState) -> dict:
    """Retrieve relevant documents from the local knowledge base."""
    docs = retriever.invoke(state["question"])
    return {"local_docs": [doc.page_content for doc in docs]}

def grade_local_doc(state: CRAGState) -> dict:
    """Grade a local document for correctness and relevance."""
    verdict = []
    for chunk in state["local_docs"]:
        result = verdict_model.invoke(
            f"Question: {state['question']}\n\nChunk:\n{chunk}\n\n"
            "Classify this chunk as exactly one of: correct (directly and fully answers), "
            "ambiguous (partially relevant or possibly outdated), "
            "incorrect (not relevant or contradicts the question's premise)."
        )
        verdict.append(result)
    return {"verdicts": verdict}

def refine_ambiguous(state: CRAGState) -> dict:
    """Strip ambiguous chunks down to just their genuinely useful sentences."""
    refined = []
    for chunk, verdict in zip(state["local_docs"], state["verdicts"]):
        if verdict == "correct":
            refined.append(chunk)
        elif verdict == "ambiguous":
            result = refine_model.invoke(
                f"Question: {state['question']}\n\nChunk:\n{chunk}\n\n"
                "Extract ONLY the sentences that are genuinely relevant and reliable "
                "for answering the question. Discard the rest."
            )
            refined.extend(result.relevant_sentences)
    return {"refined_knowledge": refined}

def route_after_grading(state: CRAGState) -> dict:
    """Determine the next step based on the current state."""
    correct_count = state["verdicts"].count("correct")
    incorrect_count = state["verdicts"].count("incorrect")

    if correct_count ==0 or incorrect_count > len(state["verdicts"]) / 2:   
        return "web_search_fallback"

    return "generate"


def web_search_fallback(state: CRAGState) -> dict:
    results = web_search.invoke({"query": state["question"]})
    snippets = [r["content"] for r in results.get("results", [])][:3]
    return {"web_results": snippets, "used_web_fallback": True}

def generate(state: CRAGState) -> dict:
    local_block = "\n".join(f"- [LOCAL KB] {k}" for k in state["refined_knowledge"]) or "(none used)"
    web_block = "\n".join(f"- [WEB] {w}" for w in state.get("web_results", [])) or "(not used)"

    response = model.invoke(
        f"Question: {state['question']}\n\n"
        f"Local knowledge base findings:\n{local_block}\n\n"
        f"Web search findings:\n{web_block}\n\n"
        "Answer the question. When you use a fact, note in parentheses whether it "
        "came from [LOCAL KB] or [WEB]."
    )
    return {"answer": response.content}

builder = StateGraph(CRAGState)
builder.add_node("retrieve_local", retrieve_local)
builder.add_node("grade_local_doc", grade_local_doc)
builder.add_node("refine_ambiguous", refine_ambiguous)
builder.add_node("web_search_fallback", web_search_fallback)
builder.add_node("route_after_grading", route_after_grading)
builder.add_node("generate", generate)

builder.add_edge(START, "retrieve_local")
builder.add_edge("retrieve_local", "grade_local_doc")
builder.add_edge("grade_local_doc", "refine_ambiguous")
builder.add_conditional_edges(
    "refine_ambiguous",route_after_grading, {'web_search_fallback': 'web_search_fallback', 'generate': 'generate'}
)
builder.add_edge("web_search_fallback", "generate")
builder.add_edge("generate", END)

graph = builder.compile()

while True:
    question = input("Ask a question: ").strip()
    if question.lower() in ("quit", "exit"):
        print("Goodbye!")
        break
    if not question:
        continue

    result = graph.invoke({
        "question": question, "local_docs": [], "verdicts": [],
        "refined_knowledge": [], "web_results": [], "used_web_fallback": False, "answer": "",
    })

    print("\n" + "=" * 60)
    fallback_note = " (used web fallback)" if result["used_web_fallback"] else " (local KB only)"
    print(f"ANSWER{fallback_note}")
    print("=" * 60)
    print(result["answer"])
    print("=" * 60 + "\n")