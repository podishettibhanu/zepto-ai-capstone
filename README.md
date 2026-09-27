# Zepto Data & AI Platform — Capstone

One repository, three connected modules:

| Module | Path | Marks | What it does |
|---|---|---|---|
| Data Pipeline | [`/data_pipeline`](data_pipeline/README.md) | 25 | Scrapes books.toscrape.com, cleans it, converts GBP→INR at a fixed rate, loads into a normalized SQLite DB, runs SQL + pandas queries. |
| Analytics | [`/analytics`](analytics/README.md) | 50 | Profiles, cleans and visualizes the Titanic dataset, then builds/evaluates/tunes a classification pipeline plus a regression side-task. |
| Support Assistant | [`/support_assistant`](support_assistant/README.md) | 25 | A LangGraph-orchestrated, ChromaDB-grounded RAG service over Zepto's own policy docs, served via FastAPI, fully runnable offline in mock mode. |

## Setup

Each module has its own `requirements.txt` (chosen over one consolidated file, since the three modules have non-overlapping, occasionally conflicting dependency sets — e.g. module 3's `chromadb`/`langgraph` stack has no reason to be installed just to run module 1's scraper).

```bash
# per module, from inside that module's folder
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Running each module end to end

**1. Data Pipeline**
```bash
cd data_pipeline
python scrape_and_load.py
```
This scrapes live data, writes `books.db`, and prints all 5+ SQL query results plus the `pd.read_sql` vs `pd.merge` equivalence check. Needs internet (it hits books.toscrape.com).

**2. Analytics**
```bash
cd analytics
python 01_eda.py        # loads titanic via sns.load_dataset (needs internet once), profiles, cleans, saves titanic.csv, produces charts + EDA writeup to stdout
python 02_modeling.py   # reads the committed titanic.csv, runs the full modeling pipeline, saves models/best_pipeline.joblib
```

## **3. Support Assistant**

```bash
cd support_assistant

uvicorn main:app --reload --port 7860
```

Open the API documentation in your browser:

```text
http://127.0.0.1:7860/docs
```

Use `POST /ask` in Swagger to test the service.

Example questions:

```json
{
  "query": "How can I track my delivery?"
}
```

```json
{
  "query": "What is the capital of India?"
}
```

The first question is treated as a Zepto policy question and uses the ChromaDB retrieval pipeline. The second is treated as a general question and is rejected by the deterministic mock mode.

Runs fully offline in `MOCK_LLM=1` (default) mode — no API key is needed.

It can also be built and run with Docker:

```bash
docker build -t zepto-assistant .
docker run -p 7860:7860 zepto-assistant
```

## Design decisions summary

- **Data Pipeline**: normalized 2-table schema (`categories` ⟷ `books`), fixed FX rate of 1 GBP = 105.50 INR (an assignment constant, not a live rate), rows that fail parsing are dropped and logged rather than silently imputed with a fabricated rating.
- **Analytics**: one dataset load (`sns.load_dataset('titanic')`), cached to `titanic.csv` so grading never needs network twice; missing-value strategy follows the <5%-drop / 5–30%-impute / >30%-decide-explicitly rule; all preprocessing is fit on the training split only, enforced via a `ColumnTransformer`+`Pipeline`.
- **Support Assistant**: retrieval (embeddings + ChromaDB) always runs for real; only the *generation* step is gated behind `MOCK_LLM`, so the graded baseline needs no LLM API key at all.

## Git workflow

This repo's history contains a feature branch (`feature/data-pipeline-queries`) with 2+ commits, merged back into `main` — see `git log --graph --all`.
