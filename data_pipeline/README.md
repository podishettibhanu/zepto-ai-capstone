# Data Pipeline (`/data_pipeline`)

## Install & run

```bash
pip install -r requirements.txt
python scrape_and_load.py
```

Requires internet access (it scrapes `books.toscrape.com` live). Running it
regenerates `books.db` from scratch every time, so the database file itself
does not need to be committed — the script *is* the recreation script.

## What it does

1. Reads the category sidebar on the site's homepage, then scrapes every
   paginated listing page **within each category** (rather than the flat
   "All products" pages), so `category` is captured directly per book instead
   of being inferred later. It stops once it has collected rows from at least
   3 categories and at least 60 books total.
2. Cleans each field:
   - `price_gbp`: strip the `£` symbol, parse to `float`.
   - `rating`: map the CSS class word (`One`…`Five`) to an `int` 1–5.
   - `in_stock`: parse the availability text into a `bool`.
   - **Failure handling**: a book's rating/price come from fixed markup on a
     purpose-built scraping-practice site, so a parse failure means the row
     is genuinely anomalous, not a case where "the median rating" would mean
     anything sensible for a *book*. We therefore **drop** any row that fails
     to parse (logged with a count) rather than impute — imputing a numeric
     median rating onto a specific unrelated book would misrepresent that
     book, whereas dropping a handful of anomalous rows out of 60+ has no
     material effect on the dataset.
3. Converts `price_gbp` → `price_inr` using the fixed project rate
   **1 GBP = 105.50 INR** (an assignment-defined constant, not a live rate —
   no API call, no date reference needed).
4. Loads into a normalized 2-table SQLite schema:
   - `categories(category_id PK, category_name UNIQUE)`
   - `books(book_id PK, title, price_gbp, price_inr, rating, in_stock, category_id FK -> categories)`
5. Runs 6 SQL queries covering `SELECT/WHERE`, `ORDER BY` + `LIMIT`,
   `DISTINCT`, `BETWEEN`, `IN`, and a `JOIN` (top-rated books per category),
   printing each query's output.
6. Reads two of those results back with `pd.read_sql`, and separately
   reproduces the join query with `pd.merge` directly on in-memory
   DataFrames, then asserts the two outputs are equal.

## Git workflow note

The feature-branch-then-merge requirement is demonstrated once against the
whole repository's history (see the root README / `git log --graph --all`),
using this module's query work as the feature being developed.

## Query coverage checklist

- `1_select_where` -> SELECT + WHERE
- `2_order_by_limit` -> ORDER BY + LIMIT
- `3_distinct` -> DISTINCT
- `4_between` -> BETWEEN
- `5_in` -> IN
- `6_join_top_rated_per_category` -> JOIN (books ⋈ categories)
## Verified Run

The data pipeline was executed successfully with 69 rows across 3 categories. SQLite loading, SQL queries, pandas `read_sql`, and the equivalent `pd.merge` operation were verified successfully.