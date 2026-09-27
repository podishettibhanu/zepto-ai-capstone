"""
Zepto Data Pipeline capstone module.

Scrapes books.toscrape.com by category, cleans the fields, converts GBP -> INR
at a fixed baseline rate, loads everything into a normalized SQLite database,
and runs the required SQL + pandas queries against it.

Run:
    python scrape_and_load.py
"""

import re
import sqlite3
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://books.toscrape.com/"
DB_PATH = Path(__file__).parent / "books.db"

# Assignment-defined fixed FX rate. Not a live/historical market rate -- an
# artificial constant used only so price_inr is deterministic and reproducible.
GBP_TO_INR = 105.50

RATING_WORDS = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}

MIN_ROWS = 60  # per acceptance criteria: >= 60 rows across >= 3 categories
MIN_CATEGORIES = 3


def get_soup(url: str) -> BeautifulSoup:
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "html.parser")


def list_categories() -> list[tuple[str, str]]:
    """Returns [(category_name, category_index_url), ...] from the sidebar."""
    soup = get_soup(BASE_URL + "index.html")
    links = soup.select("div.side_categories ul li ul li a")
    cats = []
    for a in links:
        name = a.get_text(strip=True)
        href = BASE_URL + "catalogue/" + a["href"].split("catalogue/")[-1]
        cats.append((name, href))
    return cats


def scrape_category(name: str, index_url: str) -> list[dict]:
    """Scrapes every paginated page of a single category listing."""
    rows = []
    url = index_url
    while url:
        soup = get_soup(url)
        for article in soup.select("article.product_pod"):
            title = article.h3.a["title"]
            price_text = article.select_one("p.price_color").get_text(strip=True)
            rating_word = article.select_one("p.star-rating")["class"][1]
            availability_text = article.select_one(
                "p.instock.availability"
            ).get_text(strip=True)
            rows.append(
                {
                    "title": title,
                    "price_raw": price_text,
                    "star_rating_raw": rating_word,
                    "availability_raw": availability_text,
                    "category": name,
                }
            )
        next_link = soup.select_one("li.next a")
        url = (url.rsplit("/", 1)[0] + "/" + next_link["href"]) if next_link else None
    return rows


def scrape_until_enough() -> pd.DataFrame:
    all_rows: list[dict] = []
    categories_used = 0
    for name, url in list_categories():
        if name.lower() == "books":  # root pseudo-category, skip
            continue
        rows = scrape_category(name, url)
        if not rows:
            continue
        all_rows.extend(rows)
        categories_used += 1
        print(f"  scraped category '{name}': {len(rows)} books "
              f"(running total {len(all_rows)})")
        if len(all_rows) >= MIN_ROWS and categories_used >= MIN_CATEGORIES:
            break
    df = pd.DataFrame(all_rows)
    assert len(df) >= MIN_ROWS, f"only got {len(df)} rows, need >= {MIN_ROWS}"
    assert df["category"].nunique() >= MIN_CATEGORIES
    return df


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def parse_price(raw: str):
        m = re.search(r"[\d.]+", raw.replace(",", ""))
        return float(m.group()) if m else None

    def parse_rating(raw: str):
        return RATING_WORDS.get(raw)

    def parse_stock(raw: str):
        return "in stock" in raw.lower()

    df["price_gbp"] = df["price_raw"].apply(parse_price)
    df["rating"] = df["star_rating_raw"].apply(parse_rating)
    df["in_stock"] = df["availability_raw"].apply(parse_stock)

    # Rows where price or rating failed to parse are unrecoverable for this
    # dataset (there's no sensible "median rating" for a book) -- we drop them
    # rather than impute, and log how many were dropped and why.
    before = len(df)
    bad_mask = df["price_gbp"].isna() | df["rating"].isna()
    if bad_mask.any():
        print(f"  dropping {bad_mask.sum()} unparseable rows out of {before}")
    df = df[~bad_mask].reset_index(drop=True)

    df["price_inr"] = (df["price_gbp"] * GBP_TO_INR).round(2)
    return df[["title", "category", "price_gbp", "price_inr", "rating", "in_stock"]]


# ---------------------------------------------------------------------------
# Load into normalized SQLite schema
# ---------------------------------------------------------------------------

def load_to_sqlite(df: pd.DataFrame, db_path: Path) -> None:
    if db_path.exists():
        db_path.unlink()
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.executescript(
        """
        CREATE TABLE categories (
            category_id INTEGER PRIMARY KEY AUTOINCREMENT,
            category_name TEXT UNIQUE NOT NULL
        );
        CREATE TABLE books (
            book_id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            price_gbp REAL NOT NULL,
            price_inr REAL NOT NULL,
            rating INTEGER NOT NULL,
            in_stock INTEGER NOT NULL,
            category_id INTEGER NOT NULL REFERENCES categories(category_id)
        );
        """
    )
    for cat in sorted(df["category"].unique()):
        cur.execute(
            "INSERT INTO categories (category_name) VALUES (?)", (cat,)
        )
    cat_ids = dict(
        cur.execute("SELECT category_name, category_id FROM categories").fetchall()
    )
    cur.executemany(
        """INSERT INTO books
           (title, price_gbp, price_inr, rating, in_stock, category_id)
           VALUES (?, ?, ?, ?, ?, ?)""",
        [
            (
                r.title,
                r.price_gbp,
                r.price_inr,
                int(r.rating),
                int(r.in_stock),
                cat_ids[r.category],
            )
            for r in df.itertuples()
        ],
    )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------

QUERIES = {
    "1_select_where": """
        SELECT title, price_inr, rating FROM books
        WHERE in_stock = 1 AND rating >= 4
        LIMIT 10;
    """,
    "2_order_by_limit": """
        SELECT title, price_inr FROM books
        ORDER BY price_inr DESC
        LIMIT 10;
    """,
    "3_distinct": """
        SELECT DISTINCT category_name FROM categories;
    """,
    "4_between": """
        SELECT title, price_inr FROM books
        WHERE price_inr BETWEEN 500 AND 1500
        ORDER BY price_inr;
    """,
    "5_in": """
        SELECT title, rating FROM books
        WHERE rating IN (4, 5)
        LIMIT 10;
    """,
    "6_join_top_rated_per_category": """
        SELECT c.category_name, b.title, b.rating, b.price_inr
        FROM books b
        JOIN categories c ON b.category_id = c.category_id
        ORDER BY c.category_name, b.rating DESC
        LIMIT 10;
    """,
}


def run_queries(db_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    for name, sql in QUERIES.items():
        print(f"\n--- Query [{name}] ---\n{sql.strip()}")
        result = conn.execute(sql).fetchall()
        for row in result:
            print(row)

    # Read two query results into pandas via pd.read_sql
    df_sql_join = pd.read_sql(QUERIES["6_join_top_rated_per_category"], conn)
    df_sql_instock = pd.read_sql(QUERIES["1_select_where"], conn)
    print("\n--- pd.read_sql: join query ---")
    print(df_sql_join)

    # Reproduce the join purely with pandas.merge on in-memory DataFrames
    books_df = pd.read_sql("SELECT * FROM books", conn)
    categories_df = pd.read_sql("SELECT * FROM categories", conn)
    df_merge_join = (
        books_df.merge(categories_df, on="category_id")
        .sort_values(["category_name", "rating"], ascending=[True, False])
        [["category_name", "title", "rating", "price_inr"]]
        .head(10)
        .reset_index(drop=True)
    )
    print("\n--- pd.merge equivalent of the join query ---")
    print(df_merge_join)

    same = df_sql_join.reset_index(drop=True).equals(df_merge_join)
    print(f"\nSQL JOIN and pandas merge produce equivalent output: {same}")

    conn.close()


def main():
    print("Discovering categories and scraping books...")
    raw_df = scrape_until_enough()
    print(f"\nScraped {len(raw_df)} raw rows across "
          f"{raw_df['category'].nunique()} categories.")

    print("\nCleaning fields...")
    clean_df = clean(raw_df)
    print(f"{len(clean_df)} rows remain after cleaning.")
    print(clean_df.head())

    print(f"\nLoading into {DB_PATH} ...")
    load_to_sqlite(clean_df, DB_PATH)

    print("\nRunning required SQL + pandas queries...")
    run_queries(DB_PATH)


if __name__ == "__main__":
    main()
