# Support Assistant (`/support_assistant`)

## Install & run (graded baseline — fully offline)

```bash
pip install -r requirements.txt
uvicorn main:app --reload --port 7860
```

```bash
curl -X POST localhost:7860/ask -H "Content-Type: application/json" \
  -d '{"query": "What is your delivery fee?"}'
# -> routes through retrieve_and_answer (contains keyword "delivery")

curl -X POST localhost:7860/ask -H "Content-Type: application/json" \
  -d '{"query": "What is the capital of France?"}'
# -> routes through direct_answer (no policy keyword)
```

Paste the two raw JSON responses you get here once you run it, e.g.:

```
POST /ask {"query": "What is your delivery fee?"}
-> {"answer": "Based on the retrieved context: Zepto delivers grocery and
   household essentials to serviceable pin codes within 10 to 30 minutes...",
   "sources": ["doc_01_chunk0", ...], "confidence": 1.0}

POST /ask {"query": "What is the capital of France?"}
-> {"answer": "I can only answer questions about Zepto policies right now.",
   "sources": [], "confidence": 1.0}
```

## Docker

```bash
docker build -t zepto-assistant .
docker run -p 7860:7860 zepto-assistant
```
Builds and serves `POST /ask` locally at `MOCK_LLM=1` (the Dockerfile's
default) — no API key required. This local build/run is the required,
graded baseline.

### Optional, ungraded extension

Set `MOCK_LLM=0` and provide `GROQ_API_KEY` (Groq's free tier,
console.groq.com, no card required) to switch every generation step to a
real LLM call. Not required and not what's graded — the default `MOCK_LLM=1`
path must be fully correct on its own.

## Architecture (ingestion → embedding → retrieval → generation)

1. **Ingestion**: `docs/doc_01.txt … doc_08.txt` hold Zepto's own policy text
   verbatim. `build_index()` in `main.py` reads each file and calls
   `_chunk_document()`, a simple sentence-boundary chunker (≈400 chars/chunk;
   each of these short docs mostly becomes a single chunk).
2. **Embedding**: each chunk is embedded locally with `sentence-transformers`'
   `all-MiniLM-L6-v2` model (`_embedder.encode(...)`) — no API key, no
   network call, runs entirely on-machine.
3. **Storage**: embeddings + chunk text + a `source` (doc id) metadata field
   are stored in a ChromaDB collection named `zepto_policies`
   (`_chroma_client.get_or_create_collection(...)`).
4. **Retrieval**: the LangGraph node `retrieve_and_answer` calls
   `retrieve_top_k()`, which embeds the incoming query and asks ChromaDB for
   the top-3 most similar chunks by cosine similarity. This step always runs
   for real, in both `MOCK_LLM` states, since it needs no API key.
5. **Generation**: only the final answer-generation line inside
   `retrieve_and_answer` (and inside `direct_answer`) branches on
   `MOCK_LLM`. At the default `MOCK_LLM=1`, it returns a deterministic
   templated string built from the top retrieved chunk (or a fixed string for
   `direct_answer`) — no LLM call, no network. At `MOCK_LLM=0` (optional
   extension), it instead formats `PROMPT_TEMPLATE` (role/context/task/
   format/length + a negative constraint + a few-shot example) and calls a
   real LLM via Groq.
6. **Routing**: `classify_intent` decides `policy_question` vs.
   `general_question` (keyword heuristic in mock mode, LLM call in the
   extension); `route_by_intent` is a conditional edge that sends the state to
   `retrieve_and_answer` or `direct_answer` — this routing logic itself never
   depends on `MOCK_LLM`, only what happens *inside* the chosen node does.
7. **Structured output**: the FastAPI `/ask` endpoint validates the graph's
   final state against the `AskResponse` Pydantic model
   (`answer: str, sources: list[str], confidence: float`). In mock mode this
   is populated deterministically by the node code itself, so there's no LLM
   output to fail validation. The optional real-LLM path's retry-with-
   correction logic on schema-validation failure would live around the
   `call_real_llm` invocation in `retrieve_and_answer`/`direct_answer`.

```
docs/*.txt --chunk--> _chunk_document --embed--> all-MiniLM-L6-v2
                                                        |
                                                        v
                                              ChromaDB "zepto_policies"
                                                        |
query --classify_intent--(routes)--> retrieve_and_answer --embed+query--> top-3 chunks
                              \                              |
                               \--> direct_answer      MOCK_LLM ? canned : real LLM
                                          |                   |
                                          v                   v
                                    canned string      AskResponse (validated)
```
## Testing

The FastAPI `/ask` endpoint was tested locally using Swagger UI at `http://127.0.0.1:7860/docs`.

### Test 1 — Policy Question

**Request:**

```json
{
  "query": "How can I track my delivery?"
}
```

**Result:** HTTP `200 OK`

**Response:**

```json
{
  "answer": "Based on the retrieved context: Every Zepto order shows a live rider-tracking map from the moment it is packed until delivery, accessible from the 'Track Order' screen. Estimated delivery time updates automatically as the rider move",
  "sources": [
    "doc_04_chunk0",
    "doc_06_chunk0",
    "doc_04_chunk1"
  ],
  "confidence": 1
}
```

This confirms that a policy question is routed through the retrieval flow and returns retrieved document sources.

### Test 2 — General Question

**Request:**

```json
{
  "query": "What is the capital of India?"
}
```

**Result:** HTTP `200 OK`

**Response:**

```json
{
  "answer": "I can only answer questions about Zepto policies right now.",
  "sources": [],
  "confidence": 1
}
```

This confirms that a non-policy question is routed to the direct-answer flow without retrieving policy documents.

### Test Summary

| Test                    | Expected Behavior                                   | Result |
| ----------------------- | --------------------------------------------------- | ------ |
| Policy question         | Retrieve relevant policy context and return sources | PASS   |
| General question        | Do not retrieve policy documents                    | PASS   |
| FastAPI `/ask` endpoint | Return structured JSON response                     | PASS   |
| Confidence field        | Return confidence value                             | PASS   |
| Source field            | Return retrieved source IDs for policy queries      | PASS   |

The Support Assistant was successfully tested in the required deterministic mock mode with the FastAPI service running locally.
