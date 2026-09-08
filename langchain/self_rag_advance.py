import warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
load_dotenv()

from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI

# Models & Tools
model = ChatOpenAI(model_name="gpt-4o", temperature=0.0)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

KNOWLEDGE_BASE = [
    "NimbusCloud's free tier includes 5GB of storage and 100 API calls per day.",
    "NimbusCloud's Pro tier costs $29/month and includes 500GB storage and unlimited API calls.",
    "Refunds are available within 14 days of purchase for any paid NimbusCloud plan.",
    "NimbusCloud's API rate limit for the Pro tier is 1000 requests per minute.",
    "Data exported from NimbusCloud is provided in JSON or CSV format.",
    "NimbusCloud does not currently support real-time collaborative editing.",
    "Two-factor authentication is mandatory for all NimbusCloud Enterprise accounts.",
    "NimbusCloud's Enterprise tier includes a dedicated account manager and SLA guarantees.",
]

docs = [Document(page_content=text) for text in KNOWLEDGE_BASE]
vectorstore = Chroma.from_documents(docs, embeddings,persist_directory="./chroma_self_rag")
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

class ChunkGrade(BaseModel):
    revelant : bool 
    confidence : float = Field(ge=0, le=1)
    reason: str 

class GroundednessGrade(BaseModel):
    grounded : bool 
    confidence : float = Field(ge=0, le=1)
    unsupported_claims: list[str] = Field(default_factory=list)

chunk_grader = model.with_structured_output(ChunkGrade)
groundedness_grader = model.with_structured_output(GroundednessGrade)

class SelfRAGState(TypedDict):
    question: str
    current_query: str
    retrieved_docs : list[str]
    relevant_docs : list[str]
    answer: str
    attempt : int
    max_attempts : int
    final_status : str

def retrieve(state: SelfRAGState) -> dict:
    """Retrieve relevant documents based on the current query."""
    docs = retriever.invoke(state["current_query"])
    return {"retrieved_docs": [doc.page_content for doc in docs]}


def grad_chunks(state: SelfRAGState) -> dict:
    """Grade the retrieved documents for relevance."""
    relevant_docs = []
    for chunk in state["retrieved_docs"]:
        grade = chunk_grader.invoke(
            f"Question: {state['question']}\n\nRetrieved chunk:\n{chunk}\n\n"
            "Is this chunk relevant and useful for answering the question? "
            "Score your confidence 0-1 and explain briefly."
        )
        if grade.revelant and grade.confidence > 0.5:
            relevant_docs.append(chunk)
    return {"relevant_docs": relevant_docs}


def route_after_grading(state: SelfRAGState) -> dict:
    """Determine the next step based on the current state."""
    if len(state["relevant_docs"]) >=1:
        return "generate"

    if state["attempt"] >= state["max_attempts"]:
        return "give_up"

    return "rewrite_query"

def rewrite_query(state: SelfRAGState) -> dict:
    hypothetical  = model.invoke(
        f"Write a short, plausible (possibly wrong) hypothetical answer to:{state['question']}"
    ) 
    return {"current_query": hypothetical.content.strip(), "attempt": state["attempt"] + 1}

def generate(state: SelfRAGState) -> dict:
    context = "\n".join(f"- {d}" for d in state["relevant_docs"])
    response = model.invoke(
        f"Context:\n{context}\n\nQuestion: {state['question']}\n\n"
        "Answer using ONLY the context above. Do not add outside knowledge."
    )
    return {"answer": response.content}

def check_graoundness(state: SelfRAGState) -> dict:
    context = "\n".join(f"- {d}" for d in state["relevant_docs"])
    grade = groundedness_grader.invoke(
        f"Context:\n{context}\n\nGenerated answer:\n{state['answer']}\n\n"
        "Does the answer ONLY make claims supported by the context? "
        "List any unssupported claims found"
    )

    if grade.grounded and grade.confidence > 0.5:
        return {"final_status": f"CONFIDENT (groundedness confidence: {grade.confidence:.0%})"}

    unsupported_claims = "; ".join(grade.unsupported_claims) or "general lack of support"
    return {"final_status": f"UNSUPPORTED CLAIMS: {unsupported_claims} (groundedness confidence: {grade.confidence:.0%})"}


def give_up(state: SelfRAGState) -> dict:
    return {
        "answer": "I don't have enough relevant information in the knowledge base to answer this confidently.",
        "final_status": "GAVE UP — no relevant chunks found after all retrieval attempts",
    }
  
builder = StateGraph(SelfRAGState)
builder.add_node("retrieve", retrieve)
builder.add_node("grade_chunks", grad_chunks)
builder.add_node("rewrite_query", rewrite_query)
builder.add_node("generate", generate)
builder.add_node("check_groundedness", check_graoundness)
builder.add_node("give_up", give_up)

builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "grade_chunks")
builder.add_conditional_edges(
    "grade_chunks", route_after_grading, {
        "generate": "generate",
        "rewrite_query": "rewrite_query",
        "give_up": "give_up"
    }
)
builder.add_edge("rewrite_query", "retrieve")
builder.add_edge("generate", "check_groundedness")
builder.add_edge("check_groundedness", END)
builder.add_edge("give_up", END)

graph = builder.compile()

print("Self-RAG with Conditional Regeneration (type 'quit' to exit)\n")

while True:
    question = input("Ask a question about NimbusCloud: ").strip()
    if question.lower() in ("quit", "exit"):
        print("Goodbye!")
        break
    if not question:
        continue

    result = graph.invoke({
        "question": question, "current_query": question,
        "retrieved_docs": [], "relevant_docs": [], "answer": "",
        "attempt": 0, "max_attempts": 2, "final_status": "",
    })

    print("\n" + "=" * 60)
    print(f"ANSWER  [{result['final_status']}]")
    print("=" * 60)
    print(result["answer"])
    print("=" * 60 + "\n")






    