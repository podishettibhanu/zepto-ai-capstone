"""
Zepto Support Assistant -- a small LangGraph-orchestrated RAG service.

Graded baseline: MOCK_LLM unset (or "1") -> fully deterministic, offline,
no API key, no network call to any LLM provider.
Optional extension: MOCK_LLM=0 -> real LLM calls via Groq's free tier.

Run:
    uvicorn main:app --reload --port 7860
"""

import os
from typing import TypedDict

import chromadb
from fastapi import FastAPI
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

MOCK_LLM = os.environ.get("MOCK_LLM", "1") != "0"

POLICY_KEYWORDS = [
    "delivery",
    "return",
    "refund",
    "membership",
    "tracking",
    "cancel",
    "gift card",
    "support hours",
]

DOCS_DIR = os.path.join(os.path.dirname(__file__), "docs")

# ---------------------------------------------------------------------------
# Structured prompt template (role - context - task - format - length),
# with a negative constraint and a few-shot example. Used by the optional
# MOCK_LLM=0 real-LLM path in retrieve_and_answer / direct_answer.
# ---------------------------------------------------------------------------
PROMPT_TEMPLATE = """\
ROLE: You are Zepto's customer support assistant.

CONTEXT: Use ONLY the following retrieved policy excerpts to answer the
customer's question. Do not use any outside knowledge.
---
{context}
---

TASK: Answer the customer's question below, grounded strictly in the context
above.

NEGATIVE CONSTRAINT: Do not answer using information not present in the
provided context. If the context does not contain the answer, say so
explicitly instead of guessing.

FEW-SHOT EXAMPLE:
Q: "How much is standard delivery?"
Context: "Standard delivery is free on orders over INR 149; orders below
this threshold incur a flat INR 25 delivery fee."
A: "Standard delivery is free on orders over INR 149. Orders below that
incur a flat INR 25 delivery fee."

FORMAT: Respond with a single, direct, plain-text answer (no markdown, no
preamble).

LENGTH: 1-3 sentences.

QUESTION: {question}
"""


# ---------------------------------------------------------------------------
# Embedding + retrieval setup (always real -- no API key needed for this part)
# ---------------------------------------------------------------------------
_embedder = SentenceTransformer("all-MiniLM-L6-v2")
_chroma_client = chromadb.Client()
_collection = _chroma_client.get_or_create_collection("zepto_policies")


def _chunk_document(text: str, max_chars: int = 400) -> list[str]:
    """Simple fixed-size chunking; these docs are short so most become a
    single chunk, but longer ones split cleanly on sentence boundaries."""
    sentences = text.split(". ")
    chunks, current = [], ""
    for s in sentences:
        if len(current) + len(s) > max_chars and current:
            chunks.append(current.strip())
            current = ""
        current += s + ". "
    if current.strip():
        chunks.append(current.strip())
    return chunks


def build_index() -> None:
    if _collection.count() > 0:
        return  # already indexed
    ids, texts, metadatas = [], [], []
    for fname in sorted(os.listdir(DOCS_DIR)):
        if not fname.endswith(".txt"):
            continue
        doc_id = fname.replace(".txt", "")
        with open(os.path.join(DOCS_DIR, fname), encoding="utf-8") as f:
            content = f.read().strip()
        for i, chunk in enumerate(_chunk_document(content)):
            ids.append(f"{doc_id}_chunk{i}")
            texts.append(chunk)
            metadatas.append({"source": doc_id})
    embeddings = _embedder.encode(texts).tolist()
    _collection.add(ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings)


def retrieve_top_k(query: str, k: int = 3) -> list[dict]:
    query_embedding = _embedder.encode([query]).tolist()
    results = _collection.query(query_embeddings=query_embedding, n_results=k)
    hits = []
    for doc_id, doc_text, meta in zip(
        results["ids"][0], results["documents"][0], results["metadatas"][0]
    ):
        hits.append({"id": doc_id, "text": doc_text, "source": meta["source"]})
    return hits


# ---------------------------------------------------------------------------
# Optional real-LLM call (only used when MOCK_LLM=0)
# ---------------------------------------------------------------------------

def call_real_llm(prompt: str) -> str:
    from groq import Groq

    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    resp = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content


# ---------------------------------------------------------------------------
# Structured output schema
# ---------------------------------------------------------------------------
class AskRequest(BaseModel):
    query: str


class AskResponse(BaseModel):
    answer: str
    sources: list[str] = Field(default_factory=list)
    confidence: float


# ---------------------------------------------------------------------------
# LangGraph state + nodes
# ---------------------------------------------------------------------------
class GraphState(TypedDict):
    query: str
    intent: str
    answer: str
    sources: list[str]
    confidence: float


def classify_intent(state: GraphState) -> GraphState:
    query_lower = state["query"].lower()
    if MOCK_LLM:
        intent = (
            "policy_question"
            if any(kw in query_lower for kw in POLICY_KEYWORDS)
            else "general_question"
        )
    else:
        prompt = (
            "Classify this customer query as exactly one word, either "
            "'policy_question' or 'general_question': "
            f"{state['query']}"
        )
        result = call_real_llm(prompt).strip().lower()
        intent = "policy_question" if "policy" in result else "general_question"
    return {**state, "intent": intent}


def route_by_intent(state: GraphState) -> str:
    return "retrieve_and_answer" if state["intent"] == "policy_question" else "direct_answer"


def retrieve_and_answer(state: GraphState) -> GraphState:
    hits = retrieve_top_k(state["query"], k=3)
    source_ids = [h["id"] for h in hits]

    if MOCK_LLM or not hits:
        top_snippet = hits[0]["text"][:200] if hits else ""
        answer = f"Based on the retrieved context: {top_snippet}"
        confidence = 1.0
    else:
        context = "\n\n".join(h["text"] for h in hits)
        prompt = PROMPT_TEMPLATE.format(context=context, question=state["query"])
        answer = call_real_llm(prompt)
        confidence = 0.9

    return {**state, "answer": answer, "sources": source_ids, "confidence": confidence}


def direct_answer(state: GraphState) -> GraphState:
    if MOCK_LLM:
        answer = "I can only answer questions about Zepto policies right now."
        confidence = 1.0
    else:
        prompt = PROMPT_TEMPLATE.format(context="(no context needed)", question=state["query"])
        answer = call_real_llm(prompt)
        confidence = 0.7
    return {**state, "answer": answer, "sources": [], "confidence": confidence}


def build_graph():
    graph = StateGraph(GraphState)
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("retrieve_and_answer", retrieve_and_answer)
    graph.add_node("direct_answer", direct_answer)

    graph.set_entry_point("classify_intent")
    graph.add_conditional_edges(
        "classify_intent",
        route_by_intent,
        {"retrieve_and_answer": "retrieve_and_answer", "direct_answer": "direct_answer"},
    )
    graph.add_edge("retrieve_and_answer", END)
    graph.add_edge("direct_answer", END)
    return graph.compile()


_graph = None


def get_graph():
    global _graph
    if _graph is None:
        build_index()
        _graph = build_graph()
    return _graph


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="Zepto Support Assistant")


@app.on_event("startup")
def startup():
    get_graph()


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    graph = get_graph()
    result = graph.invoke(
        {
            "query": request.query,
            "intent": "",
            "answer": "",
            "sources": [],
            "confidence": 0.0,
        }
    )
    return AskResponse(
        answer=result["answer"],
        sources=result["sources"],
        confidence=result["confidence"],
    )
