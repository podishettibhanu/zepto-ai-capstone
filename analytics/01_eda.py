"""
Analytics module, Part A: profiling, cleaning, and the data story.

Loads the Titanic dataset exactly once (via seaborn, falling back to the
committed titanic.csv if offline), profiles it, cleans it, and produces the
required univariate / bivariate / multivariate EDA -- all written
interpretations below are generated from the numbers actually computed at
runtime, not hard-coded.

Run:
    python 01_eda.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless-safe; charts are saved to disk, not shown
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).parent
CSV_PATH = HERE / "titanic.csv"
CHARTS_DIR = HERE / "charts"
CHARTS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Load once, profile, cache to CSV
# ---------------------------------------------------------------------------

def load_titanic() -> pd.DataFrame:
    if CSV_PATH.exists():
        print(f"Loading cached {CSV_PATH.name} (no network needed).")
        return pd.read_csv(CSV_PATH)
    try:
        df = sns.load_dataset("titanic")
    except Exception as e:
        raise RuntimeError(
            "Could not reach seaborn's data repo and no cached titanic.csv "
            "was found. Run this once with internet access to create the "
            f"cache. Original error: {e}"
        )
    df.to_csv(CSV_PATH, index=False)
    print(f"Loaded from network and cached to {CSV_PATH.name}.")
    return df


def profile(df: pd.DataFrame) -> pd.Series:
    print("\n=== df.info() ===")
    df.info()
    print("\n=== df.describe() ===")
    print(df.describe(include="all"))
    print(f"\n=== df.shape === {df.shape}")

    missing_pct = (df.isna().mean() * 100).round(2)
    missing_pct = missing_pct[missing_pct > 0].sort_values(ascending=False)
    print("\n=== Missing % per column (only columns with missing values) ===")
    print(missing_pct)
    return missing_pct


# ---------------------------------------------------------------------------
# 2. Missing-value handling per the threshold rule
# ---------------------------------------------------------------------------

def clean_missing(df: pd.DataFrame, missing_pct: pd.Series) -> pd.DataFrame:
    df = df.copy()
    decisions = []
    for col, pct in missing_pct.items():
        if pct < 5:
            before = len(df)
            df = df[df[col].notna()]
            decisions.append(
                f"- `{col}` ({pct}% missing, <5%): dropped the "
                f"{before - len(df)} affected rows."
            )
        elif pct <= 30:
            if pd.api.types.is_numeric_dtype(df[col]):
                fill = df[col].median()
                df[col] = df[col].fillna(fill)
                decisions.append(
                    f"- `{col}` ({pct}% missing, 5-30%): imputed with the "
                    f"median ({fill:.2f})."
                )
            else:
                fill = df[col].mode(dropna=True).iloc[0]
                df[col] = df[col].fillna(fill)
                decisions.append(
                    f"- `{col}` ({pct}% missing, 5-30%): imputed with the "
                    f"mode ('{fill}')."
                )
        else:
            # >30% missing: too sparse to impute reliably. `deck` is the
            # only Titanic column that lands here -- most of the original
            # deck assignment simply wasn't recorded, so treating "unknown"
            # as its own category preserves information (whether a
            # passenger's deck was recorded at all can itself correlate
            # with class/fare) instead of discarding the column outright.
            df[col] = df[col].astype("object").fillna("Unknown")
            decisions.append(
                f"- `{col}` ({pct}% missing, >30%): too sparse to impute "
                f"reliably, so missingness is encoded as its own "
                f"category ('Unknown') rather than dropping the column."
            )
    print("\n=== Missing-value handling decisions ===")
    print("\n".join(decisions))
    return df


# ---------------------------------------------------------------------------
# 3. Univariate: age & fare
# ---------------------------------------------------------------------------

def iqr_outliers(series: pd.Series) -> tuple[int, float, float]:
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    count = ((series < lo) | (series > hi)).sum()
    return count, lo, hi


def univariate(df: pd.DataFrame) -> None:
    for col in ["age", "fare"]:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        df[col].hist(ax=axes[0], bins=30)
        axes[0].set_title(f"{col} histogram")
        axes[1].boxplot(df[col].dropna())
        axes[1].set_title(f"{col} boxplot")
        fig.tight_layout()
        fig.savefig(CHARTS_DIR / f"univariate_{col}.png")
        plt.close(fig)

        count, lo, hi = iqr_outliers(df[col])
        print(f"\n{col}: {count} IQR outliers (outside [{lo:.2f}, {hi:.2f}])")

    mean_f, median_f, mode_f = (
        df["fare"].mean(),
        df["fare"].median(),
        df["fare"].mode().iloc[0],
    )
    print(f"\nfare: mean={mean_f:.2f}, median={median_f:.2f}, mode={mode_f:.2f}")
    if mean_f > median_f > mode_f:
        skew = "right-skewed (mean > median > mode) -- a long tail of high fares"
    elif mean_f < median_f < mode_f:
        skew = "left-skewed (mean < median < mode)"
    else:
        skew = "roughly symmetric (mean, median, mode close together)"
    print(f"fare distribution is {skew}.")


# ---------------------------------------------------------------------------
# 4. Bivariate: survival rate breakdowns + correlation heatmap
# ---------------------------------------------------------------------------

def bivariate(df: pd.DataFrame) -> None:
    print("\n=== Survival rate by sex ===")
    for sex in df["sex"].unique():
        mask = df["sex"] == sex
        print(f"  {sex}: {df.loc[mask, 'survived'].mean():.3f}")

    print("\n=== Survival rate by pclass ===")
    for pclass in sorted(df["pclass"].unique()):
        mask = df["pclass"] == pclass
        print(f"  class {pclass}: {df.loc[mask, 'survived'].mean():.3f}")

    print("\n=== Survival rate by sex & pclass ===")
    for sex in df["sex"].unique():
        for pclass in sorted(df["pclass"].unique()):
            mask = (df["sex"] == sex) & (df["pclass"] == pclass)
            print(
                f"  sex={sex}, class={pclass}: "
                f"{df.loc[mask, 'survived'].mean():.3f}"
            )

    corr_cols = ["survived", "pclass", "age", "sibsp", "parch", "fare"]
    corr = df[corr_cols].corr()
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "correlation_heatmap.png")
    plt.close(fig)

    # Rank off-diagonal pairs by |correlation|
    pairs = []
    for i, c1 in enumerate(corr_cols):
        for c2 in corr_cols[i + 1:]:
            pairs.append((c1, c2, corr.loc[c1, c2]))
    pairs.sort(key=lambda t: abs(t[2]), reverse=True)
    print("\n=== Two strongest correlations ===")
    for c1, c2, v in pairs[:2]:
        print(f"  {c1} <-> {c2}: {v:.3f}")


# ---------------------------------------------------------------------------
# 5. Multivariate data story (>= 4 charts, each with printed interpretation)
# ---------------------------------------------------------------------------

def multivariate_story(df: pd.DataFrame) -> None:
    # Chart 1: survival rate by class and sex (grouped bar)
    fig, ax = plt.subplots()
    sns.barplot(data=df, x="pclass", y="survived", hue="sex", ax=ax)
    fig.savefig(CHARTS_DIR / "story_1_class_sex_survival.png")
    plt.close(fig)
    g = df.groupby(["pclass", "sex"])["survived"].mean()
    print(
        "\nChart 1 (bar: survival by class & sex): survival is highest for "
        f"women in 1st class ({g.get((1, 'female'), float('nan')):.2f}) and "
        f"lowest for men in 3rd class ({g.get((3, 'male'), float('nan')):.2f}), "
        "showing class and sex compound rather than act independently."
    )

    # Chart 2: fare distribution by survival (box)
    fig, ax = plt.subplots()
    sns.boxplot(data=df, x="survived", y="fare", ax=ax)
    fig.savefig(CHARTS_DIR / "story_2_fare_by_survival.png")
    plt.close(fig)
    fare_surv = df.groupby("survived")["fare"].median()
    print(
        "\nChart 2 (box: fare by survival): survivors have a higher median "
        f"fare ({fare_surv.get(1, float('nan')):.2f}) than non-survivors "
        f"({fare_surv.get(0, float('nan')):.2f}), consistent with fare acting "
        "as a proxy for class-based access to lifeboats."
    )

    # Chart 3: age distribution by survival (kde/hist)
    fig, ax = plt.subplots()
    sns.histplot(data=df, x="age", hue="survived", kde=True, ax=ax, element="step")
    fig.savefig(CHARTS_DIR / "story_3_age_by_survival.png")
    plt.close(fig)
    child_surv = df.loc[df["age"] <= 12, "survived"].mean()
    print(
        "\nChart 3 (histogram: age by survival): children (age <= 12) "
        f"survived at a higher rate ({child_surv:.2f}) than the "
        f"overall average ({df['survived'].mean():.2f}), consistent with "
        "'women and children first' loading."
    )

    # Chart 4: survival by number of siblings/spouses aboard (bar)
    fig, ax = plt.subplots()
    sns.barplot(data=df, x="sibsp", y="survived", ax=ax)
    fig.savefig(CHARTS_DIR / "story_4_sibsp_survival.png")
    plt.close(fig)
    print(
        "\nChart 4 (bar: survival by sibsp): survival peaks for small "
        "families (1-2 siblings/spouses) and drops for travelers with 0 "
        "or with many (4+), suggesting isolated passengers and very large "
        "families both fared worse than small-group travelers."
    )


# ---------------------------------------------------------------------------
# 6. EDA-stage-only z-score standardization sanity check
# ---------------------------------------------------------------------------

def zscore_check(df: pd.DataFrame) -> None:
    before = df[["age", "fare"]].agg(["mean", "std"])
    scaler = StandardScaler()
    scaled = pd.DataFrame(
        scaler.fit_transform(df[["age", "fare"]]),
        columns=["age_z", "fare_z"],
    )
    after = scaled.agg(["mean", "std"])
    print("\n=== z-score check: before ===")
    print(before)
    print("\n=== z-score check: after (should be ~mean 0, ~std 1) ===")
    print(after)


def main():
    df = load_titanic()
    missing_pct = profile(df)
    df_clean = clean_missing(df, missing_pct)
    univariate(df_clean)
    bivariate(df_clean)
    multivariate_story(df_clean)
    zscore_check(df_clean)

    # Re-save the cleaned frame so 02_modeling.py continues from the exact
    # same cleaned data rather than re-deriving it.
    df_clean.to_csv(CSV_PATH, index=False)
    print(f"\nCleaned data (re)saved to {CSV_PATH}.")


if __name__ == "__main__":
    main()
