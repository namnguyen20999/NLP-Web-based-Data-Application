# %% [markdown]
# # Cosmetics Reviews — Dataset Cleaning Pipeline
#
# This notebook documents and executes the full cleaning pipeline for
# `cosmetics_beauty_products_reviews.csv`. Each section includes an exploratory
# analysis that **motivates** the cleaning decision before any rows are removed.
#
# ## Two label-inconsistency patterns we fix
#
# | Pattern | Rating | `is_a_buyer` | Problem |
# |---|---|---|---|
# | **A — Negative buyer** | ≤ 2 ★ | `True` | Negative signal, yet label says "bought" |
# | **B — Positive non-buyer** | ≥ 4 ★ | `False` | Positive signal, yet label says "didn't buy" |
#
# ## Resolution strategies
#
# ### `"drop"`
# Remove all conflicting rows unconditionally. Fast, zero NLP cost.
#
# ### `"context_aware"` *(default — recommended)*
# For each conflict, compute the **product's rolling 6-month average rating** ending
# just before that review date. Use it as ground truth:
#
# | Pattern | Context avg | Decision | Rationale |
# |---|---|---|---|
# | A | ≤ 2.5 — product is genuinely bad | **relabel** `is_a_buyer → False` | Rating matches product trend; buyer label was wrong |
# | A | ≥ 3.5 — product is generally good | **drop** | Reviewer is an outlier; row is noise |
# | B | ≥ 3.5 — product is genuinely good | **relabel** `is_a_buyer → True` | Rating matches product trend; non-buyer label was wrong |
# | B | ≤ 2.5 — product is generally bad | **drop** | High rating on bad product is suspicious; row is noise |
# | Either | 2.5–3.5 neutral or < 3 prior reviews | **drop** | Insufficient signal to relabel safely |

# %% [markdown]
# ## Imports & logging setup

# %%
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Literal

import nltk
import numpy as np
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

BASE = Path(".")

# %% [markdown]
# ## 1. Load raw data

# %%
df_raw = pd.read_csv(BASE / "cosmetics_beauty_products_reviews.csv")
print(f"Shape: {df_raw.shape}")
df_raw.head(3)

# %% [markdown]
# ### Column types

# %%
df_raw.dtypes

# %% [markdown]
# ## 2. Missing value analysis
#
# Before any cleaning, identify which columns have nulls and whether they are critical
# for the NLP task.

# %%
null_counts = df_raw.isnull().sum()
null_pct    = (null_counts / len(df_raw) * 100).round(2)
null_summary = pd.DataFrame({"null_count": null_counts, "null_%": null_pct})
null_summary[null_summary["null_count"] > 0]

# %% [markdown]
# **Findings:**
# - `review_text` (9 nulls) and `review_rating` (1 null) — **critical**: without these
#   two fields the row cannot be used for any NLP task. → **drop**.
# - `product_tags` (~78% null) — sparse but not critical; left as-is.
#
# All other fields are complete.

# %% [markdown]
# ## 3. Rating & label distribution
#
# Understanding the overall distribution before touching any data.

# %%
print("=== review_rating distribution ===")
rating_dist = df_raw["review_rating"].value_counts().sort_index()
rating_pct  = (rating_dist / len(df_raw) * 100).round(1)
pd.DataFrame({"count": rating_dist, "%": rating_pct})

# %%
print("=== is_a_buyer distribution ===")
buyer_dist = df_raw["is_a_buyer"].value_counts()
buyer_pct  = (buyer_dist / len(df_raw) * 100).round(1)
pd.DataFrame({"count": buyer_dist, "%": buyer_pct})

# %% [markdown]
# ### 3a. Cross-tabulation: rating × is_a_buyer
#
# This reveals how the two signals relate. In a clean dataset we would expect:
# - Low ratings → `is_a_buyer=False` (unhappy reviewers who wouldn't recommend buying)
# - High ratings → `is_a_buyer=True` (happy reviewers who endorse the product)

# %%
ct = pd.crosstab(df_raw["review_rating"], df_raw["is_a_buyer"])
ct["total"] = ct.sum(axis=1)
ct["conflict_%"] = (ct[True] / ct["total"] * 100).round(1).where(ct.index <= 2, other=(ct[False] / ct["total"] * 100).round(1))
ct

# %% [markdown]
# **Key observations:**
# - **1★ reviews**: 2,451 of 3,077 (79.7%) still have `is_a_buyer=True` — the label
#   directly contradicts the rating signal.
# - **2★ reviews**: 1,334 of 1,718 (77.6%) same problem.
# - **4★ reviews**: 3,106 of 11,322 (27.4%) have `is_a_buyer=False` despite a positive rating.
# - **5★ reviews**: 8,117 of 41,626 (19.5%) same.
#
# These are the two conflict patterns we will resolve.

# %% [markdown]
# ### 3b. Pattern A deep-dive: low rating + `is_a_buyer=True`

# %%
mask_a_raw = (df_raw["review_rating"] <= 2) & (df_raw["is_a_buyer"] == True)
print(f"Pattern A conflicts: {mask_a_raw.sum()} rows ({mask_a_raw.mean()*100:.1f}% of dataset)")
print()
print("Most common review titles:")
df_raw.loc[mask_a_raw, "review_title"].value_counts().head(10)

# %%
print("Sample Pattern A rows:")
df_raw.loc[mask_a_raw, ["review_title", "review_text", "review_rating", "is_a_buyer"]].sample(5, random_state=42)

# %% [markdown]
# **Why this is label noise:** `is_a_buyer` is used as the target label for purchase-intent
# NLP models. A reviewer writing *"Waste of money, will never buy again"* with 1★ but
# `is_a_buyer=True` teaches the model that negative language → buy. This corrupts the
# decision boundary.

# %% [markdown]
# ### 3c. Pattern B deep-dive: high rating + `is_a_buyer=False`

# %%
mask_b_raw = (df_raw["review_rating"] >= 4) & (df_raw["is_a_buyer"] == False)
print(f"Pattern B conflicts: {mask_b_raw.sum()} rows ({mask_b_raw.mean()*100:.1f}% of dataset)")
print()
print("Most common review titles:")
df_raw.loc[mask_b_raw, "review_title"].value_counts().head(10)

# %%
print("Sample Pattern B rows:")
df_raw.loc[mask_b_raw, ["review_title", "review_text", "review_rating", "is_a_buyer"]].sample(5, random_state=42)

# %% [markdown]
# **Why this is label noise:** A reviewer writing glowing praise with 5★ but
# `is_a_buyer=False` teaches the model that positive language → don't buy. Pattern B
# is larger (11,223 rows, 18.3%) than Pattern A, making it the more damaging issue.

# %% [markdown]
# ## 4. Why not just drop everything?
#
# Dropping all 15,008 conflict rows (24.5% of data) is safe but wasteful. Some of
# those rows have a **recoverable** label — the `is_a_buyer` field is simply wrong,
# but the text and rating are valid training signal.
#
# We use each product's **rolling 6-month average rating** (computed from all reviews
# posted in the 6 months before each conflict review) as an independent ground-truth
# signal to decide whether to drop or relabel.

# %% [markdown]
# ### 4a. Compute product rolling 6-month context average

# %%
_df_ctx = df_raw.dropna(subset=["review_text", "review_rating"]).copy()
_df_ctx["review_date"] = pd.to_datetime(
    _df_ctx["review_date"], format="%d/%m/%Y %H:%M", errors="coerce"
)
_df_ctx_s = _df_ctx.sort_values(["product_id", "review_date"]).set_index("review_date")

_roll        = _df_ctx_s.groupby("product_id")["review_rating"].rolling("180D", min_periods=1)
_roll_sum    = _roll.sum().reset_index(level=0, drop=True)
_roll_count  = _roll.count().reset_index(level=0, drop=True)

_ctx_count   = _roll_count - 1
_ctx_avg     = (_roll_sum - _df_ctx_s["review_rating"]) / _ctx_count.replace(0, np.nan)

_df_ctx_s["context_avg"]   = _ctx_avg.values
_df_ctx_s["context_count"] = _ctx_count.values
_df_ctx_s = _df_ctx_s.reset_index()
_df_ctx_s = _df_ctx_s.set_index(_df_ctx.sort_values(["product_id", "review_date"]).index).reindex(_df_ctx.index)

# %% [markdown]
# ### 4b. Context avg distribution for Pattern A conflicts

# %%
_mask_a = (_df_ctx_s["review_rating"] <= 2) & (_df_ctx_s["is_a_buyer"] == True)
ctx_a = _df_ctx_s.loc[_mask_a, "context_avg"]

print("Pattern A — context_avg (product 6-month avg before the conflict review):")
print(ctx_a.describe().round(3))
print(f"\nRows with no prior reviews (NaN): {ctx_a.isna().sum()}")
print(f"Rows with context_avg <= 2.5 (genuinely bad product) → relabel: {(ctx_a <= 2.5).sum()}")
print(f"Rows with context_avg >= 3.5 (generally good product) → drop:   {(ctx_a >= 3.5).sum()}")
print(f"Rows in neutral zone 2.5–3.5                          → drop:   {((ctx_a > 2.5) & (ctx_a < 3.5)).sum()}")

# %% [markdown]
# **Interpretation:** The vast majority of Pattern A conflicts (≈96%) have a high
# context avg (≥ 3.5), meaning the product is generally well-regarded — the 1–2★
# reviewer is an outlier noise point, not representative. Only ≈4% are on genuinely
# poor products, where relabeling `is_a_buyer → False` is the right correction.

# %% [markdown]
# ### 4c. Context avg distribution for Pattern B conflicts

# %%
_mask_b = (_df_ctx_s["review_rating"] >= 4) & (_df_ctx_s["is_a_buyer"] == False)
ctx_b = _df_ctx_s.loc[_mask_b, "context_avg"]

print("Pattern B — context_avg (product 6-month avg before the conflict review):")
print(ctx_b.describe().round(3))
print(f"\nRows with no prior reviews (NaN): {ctx_b.isna().sum()}")
print(f"Rows with context_avg >= 3.5 (genuinely good product) → relabel: {(ctx_b >= 3.5).sum()}")
print(f"Rows with context_avg <= 2.5 (generally bad product)  → drop:    {(ctx_b <= 2.5).sum()}")
print(f"Rows in neutral zone 2.5–3.5                          → drop:    {((ctx_b > 2.5) & (ctx_b < 3.5)).sum()}")

# %% [markdown]
# **Interpretation:** ≈94% of Pattern B conflicts have a high context avg (≥ 3.5),
# confirming the product is genuinely good — the `is_a_buyer=False` label is wrong
# and can be safely corrected to `True`. Only ≈6% are on bad products, where a
# 4–5★ review is a suspicious outlier → drop.

# %% [markdown]
# ### 4d. Threshold justification
#
# | Threshold | Value | Rationale |
# |---|---|---|
# | `context_low` | **2.5** | Midpoint between 1★ and 4★; products below this are consistently panned |
# | `context_high` | **3.5** | Standard "above average" cutoff; products here are reliably well-received |
# | `min_context_reviews` | **3** | Fewer than 3 prior reviews in the window gives insufficient signal → fall back to drop |
#
# Rows whose context avg falls in the neutral band (2.5–3.5) are ambiguous — we
# cannot confidently relabel them, so we drop them as a safe default.

# %% [markdown]
# ### 4e. Final decision preview (before running the pipeline)

# %%
def _preview_decisions(ctx_series: pd.Series, pattern: str, low: float = 2.5, high: float = 3.5) -> None:
    relabel = (ctx_series <= low).sum() if pattern == "A" else (ctx_series >= high).sum()
    drop    = len(ctx_series) - relabel
    print(f"Pattern {pattern}: {relabel} relabeled | {drop} dropped  (of {len(ctx_series)} conflicts)")

_preview_decisions(ctx_a, "A")
_preview_decisions(ctx_b, "B")
print(f"\nTotal rows kept (relabeled): {(ctx_a <= 2.5).sum() + (ctx_b >= 3.5).sum()}")
print(f"Total rows removed (dropped): {(ctx_a > 2.5).sum() + (ctx_b < 3.5).sum()}")

# %% [markdown]
# ---
# ## Pipeline functions

# %% [markdown]
# ### VADER helpers
# Used only when `sentiment_mode='rating_and_sentiment'`. VADER provides a
# text-level confirmation of the rating signal — useful to avoid flagging rows
# where the reviewer gave a low rating but their text was actually neutral or
# mixed (e.g. "average product, not for my skin type").

# %%
def _ensure_vader() -> SentimentIntensityAnalyzer:
    try:
        return SentimentIntensityAnalyzer()
    except LookupError:
        logger.info("Downloading VADER lexicon...")
        nltk.download("vader_lexicon", quiet=True)
        return SentimentIntensityAnalyzer()


def _combine_text(row: pd.Series) -> str:
    """Concatenate title + body; title carries denser signal so it leads."""
    title = str(row.get("review_title", "") or "")
    body  = str(row.get("review_text",  "") or "")
    return f"{title}. {body}".strip()


def compute_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """Add `vader_compound` column (float in [-1, 1])."""
    logger.info("Computing VADER sentiment scores on %d rows...", len(df))
    sia      = _ensure_vader()
    combined = df.apply(_combine_text, axis=1)
    df["vader_compound"] = combined.map(lambda t: sia.polarity_scores(t)["compound"])
    logger.info("Sentiment scoring complete.")
    return df

# %% [markdown]
# ### Step 1 — Drop rows with missing critical fields

# %%
def drop_missing_critical(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows where `review_text` or `review_rating` is null."""
    mask    = df["review_text"].isna() | df["review_rating"].isna()
    dropped = int(mask.sum())
    if dropped:
        logger.info("Dropped %d rows with null review_text or review_rating.", dropped)
    return df[~mask].copy()

# %% [markdown]
# ### Step 2 — Conflict flagging
#
# #### Pattern A: low rating + `is_a_buyer=True`

# %%
def flag_negative_buyers(
    df: pd.DataFrame,
    *,
    low_rating_threshold: int = 2,
    neg_sentiment_threshold: float = 0.0,
    mode: Literal["rating_only", "rating_and_sentiment"] = "rating_only",
) -> pd.Series:
    """
    Return boolean mask — True = Pattern A conflict (low-rating buyer).

    Parameters
    ----------
    low_rating_threshold    Ratings at or below this are negative (default 2).
    neg_sentiment_threshold VADER compound below this = negative text (default 0.0).
                            Only used when mode='rating_and_sentiment'.
    mode                    'rating_only'          — flag on rating ≤ threshold & is_a_buyer=True.
                            'rating_and_sentiment' — additionally require negative VADER score;
                                                     keeps rows where text is ambiguous despite
                                                     a low rating.
    """
    if mode == "rating_and_sentiment" and "vader_compound" not in df.columns:
        raise ValueError("Call compute_sentiment(df) before using mode='rating_and_sentiment'.")
    low_rating = df["review_rating"] <= low_rating_threshold
    is_buyer   = df["is_a_buyer"] == True
    if mode == "rating_only":
        return low_rating & is_buyer
    return low_rating & is_buyer & (df["vader_compound"] < neg_sentiment_threshold)

# %% [markdown]
# #### Pattern B: high rating + `is_a_buyer=False`

# %%
def flag_positive_non_buyers(
    df: pd.DataFrame,
    *,
    high_rating_threshold: int = 4,
    pos_sentiment_threshold: float = 0.0,
    mode: Literal["rating_only", "rating_and_sentiment"] = "rating_only",
) -> pd.Series:
    """
    Return boolean mask — True = Pattern B conflict (high-rating non-buyer).

    Parameters
    ----------
    high_rating_threshold   Ratings at or above this are positive (default 4).
    pos_sentiment_threshold VADER compound above this = positive text (default 0.0).
                            Only used when mode='rating_and_sentiment'.
    mode                    'rating_only'          — flag on rating ≥ threshold & is_a_buyer=False.
                            'rating_and_sentiment' — additionally require positive VADER score.
    """
    if mode == "rating_and_sentiment" and "vader_compound" not in df.columns:
        raise ValueError("Call compute_sentiment(df) before using mode='rating_and_sentiment'.")
    high_rating = df["review_rating"] >= high_rating_threshold
    not_buyer   = df["is_a_buyer"] == False
    if mode == "rating_only":
        return high_rating & not_buyer
    return high_rating & not_buyer & (df["vader_compound"] > pos_sentiment_threshold)

# %% [markdown]
# ### Step 3a — Strategy: `"drop"`
# Remove all flagged rows with no further analysis. Use when speed matters or when
# the dataset is large enough that losing ~25% of rows is acceptable.

# %%
def _drop_pattern(df: pd.DataFrame, mask: pd.Series, label: str) -> pd.DataFrame:
    before  = len(df)
    dropped = int(mask.sum())
    logger.info(
        "[%s] drop: removed %d / %d rows (%.1f%%)",
        label, dropped, before, dropped / before * 100,
    )
    clean = df[~mask].copy()
    logger.info("  ↳ %d → %d rows.", before, len(clean))
    return clean

# %% [markdown]
# ### Step 3b — Strategy: `"context_aware"`
#
# #### Compute product rolling 6-month average
#
# For each review, we want the product's mean rating from the **180 days before** that
# review (exclusive of the review itself). This gives an independent quality signal
# not contaminated by the review we are trying to classify.
#
# Implementation uses pandas time-based `rolling("180D")` on a `DatetimeIndex`,
# which is O(n log n) — safe for datasets in the millions.

# %%
def compute_product_context_avg(
    df: pd.DataFrame,
    window_days: int = 180,
    date_col: str = "review_date",
    date_format: str = "%d/%m/%Y %H:%M",
) -> pd.DataFrame:
    """
    Add `context_avg` and `context_count` to *df*.

    `context_avg`   — product's mean rating in the `window_days` days before each review
                      (exclusive of the review itself). NaN when no prior reviews exist.
    `context_count` — number of reviews that contributed to `context_avg`.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], format=date_format, errors="coerce")
    nat_count = df[date_col].isna().sum()
    if nat_count:
        logger.warning("%d rows have unparseable review_date — context_avg will be NaN.", nat_count)

    df_sorted = df.sort_values(["product_id", date_col]).copy()
    df_sorted = df_sorted.set_index(date_col)

    roll       = df_sorted.groupby("product_id")["review_rating"].rolling(f"{window_days}D", min_periods=1)
    roll_sum   = roll.sum().reset_index(level=0, drop=True)
    roll_count = roll.count().reset_index(level=0, drop=True)

    # Subtract current row → "prior-only" window
    ctx_count = roll_count - 1
    ctx_avg   = (roll_sum - df_sorted["review_rating"]) / ctx_count.replace(0, np.nan)

    df_sorted["context_avg"]   = ctx_avg.values
    df_sorted["context_count"] = ctx_count.values
    df_sorted = df_sorted.reset_index()

    # Restore original row order
    df_sorted = df_sorted.set_index(df.sort_values(["product_id", date_col]).index)
    return df_sorted.reindex(df.index)

# %% [markdown]
# #### Resolve conflicts using context
#
# Decision logic per row:
# - **Pattern A** (low rating + is_a_buyer=True):
#   - `context_avg ≤ context_low` → product is genuinely bad → `is_a_buyer` was wrong → **relabel False**
#   - otherwise → reviewer is an outlier on a good product → **drop**
# - **Pattern B** (high rating + is_a_buyer=False):
#   - `context_avg ≥ context_high` → product is genuinely good → `is_a_buyer` was wrong → **relabel True**
#   - otherwise → suspicious high rating on a bad product → **drop**
# - Either pattern with `context_count < min_context_reviews` → not enough history → **drop**

# %%
def resolve_conflicts_with_context(
    df: pd.DataFrame,
    mask_a: pd.Series,
    mask_b: pd.Series,
    *,
    context_low: float = 2.5,
    context_high: float = 3.5,
    min_context_reviews: int = 3,
) -> pd.DataFrame:
    """
    Apply context-aware resolution to Pattern A and B conflicts.

    Parameters
    ----------
    mask_a              Boolean mask of Pattern A rows.
    mask_b              Boolean mask of Pattern B rows.
    context_low         Products ≤ this avg are "genuinely bad" (default 2.5).
    context_high        Products ≥ this avg are "genuinely good" (default 3.5).
    min_context_reviews Min prior reviews in window to trust context_avg (default 3).
    """
    if "context_avg" not in df.columns:
        raise ValueError("Call compute_product_context_avg(df) first.")

    df            = df.copy()
    drop_idx:         list[int] = []
    relabel_false:    list[int] = []
    relabel_true:     list[int] = []

    def _trustworthy(r: pd.Series) -> bool:
        return not pd.isna(r["context_avg"]) and r.get("context_count", 0) >= min_context_reviews

    for idx, row in df[mask_a].iterrows():
        if _trustworthy(row) and row["context_avg"] <= context_low:
            relabel_false.append(idx)
        else:
            drop_idx.append(idx)

    for idx, row in df[mask_b].iterrows():
        if _trustworthy(row) and row["context_avg"] >= context_high:
            relabel_true.append(idx)
        else:
            drop_idx.append(idx)

    df.loc[relabel_false, "is_a_buyer"] = False
    df.loc[relabel_true,  "is_a_buyer"] = True
    df = df.drop(index=drop_idx)

    drop_a = sum(1 for i in drop_idx if mask_a.get(i, False))
    drop_b = sum(1 for i in drop_idx if mask_b.get(i, False))
    logger.info("[Pattern A] relabeled → False: %d | dropped: %d  (of %d)", len(relabel_false), drop_a, int(mask_a.sum()))
    logger.info("[Pattern B] relabeled → True:  %d | dropped: %d  (of %d)", len(relabel_true),  drop_b, int(mask_b.sum()))
    logger.info("  ↳ Total dropped: %d | Total relabeled: %d", len(drop_idx), len(relabel_false) + len(relabel_true))
    return df

# %% [markdown]
# ## Full pipeline

# %%
def clean_dataset(
    input_path: str | Path,
    output_path: str | Path | None = None,
    *,
    low_rating_threshold: int = 2,
    high_rating_threshold: int = 4,
    neg_sentiment_threshold: float = 0.0,
    pos_sentiment_threshold: float = 0.0,
    sentiment_mode: Literal["rating_only", "rating_and_sentiment"] = "rating_only",
    strategy: Literal["drop", "context_aware"] = "context_aware",
    context_window_days: int = 180,
    context_low: float = 2.5,
    context_high: float = 3.5,
    min_context_reviews: int = 3,
    drop_helper_columns: bool = True,
) -> pd.DataFrame:
    """
    End-to-end cleaning pipeline.

    Parameters
    ----------
    input_path              Path to raw CSV.
    output_path             Write cleaned CSV here; None = skip saving.
    low_rating_threshold    Ratings ≤ this mark a negative review (default 2).
    high_rating_threshold   Ratings ≥ this mark a positive review (default 4).
    neg_sentiment_threshold VADER cutoff for negative text — Pattern A (default 0.0).
    pos_sentiment_threshold VADER cutoff for positive text — Pattern B (default 0.0).
    sentiment_mode          'rating_only' (fast) or 'rating_and_sentiment' (conservative).
    strategy                'drop'          — remove all conflicts unconditionally.
                            'context_aware' — use product 6-month rolling avg to decide.
    context_window_days     Rolling window size in days (default 180 = 6 months).
    context_low             Context avg ≤ this → genuinely bad product (default 2.5).
    context_high            Context avg ≥ this → genuinely good product (default 3.5).
    min_context_reviews     Min prior reviews required to trust context_avg (default 3).
    drop_helper_columns     Drop vader_compound / context_avg / context_count (default True).
    """
    input_path = Path(input_path)
    logger.info("=" * 60)
    logger.info("Loading dataset from %s", input_path)
    df = pd.read_csv(input_path)
    logger.info("Loaded %d rows, %d columns.", len(df), df.shape[1])
    logger.info("Strategy: %s | Sentiment mode: %s", strategy, sentiment_mode)
    logger.info("=" * 60)

    df = drop_missing_critical(df)

    if sentiment_mode == "rating_and_sentiment":
        df = compute_sentiment(df)

    mask_a = flag_negative_buyers(
        df,
        low_rating_threshold=low_rating_threshold,
        neg_sentiment_threshold=neg_sentiment_threshold,
        mode=sentiment_mode,
    )
    mask_b = flag_positive_non_buyers(
        df,
        high_rating_threshold=high_rating_threshold,
        pos_sentiment_threshold=pos_sentiment_threshold,
        mode=sentiment_mode,
    )
    logger.info(
        "Conflicts flagged — Pattern A: %d | Pattern B: %d | Total: %d",
        int(mask_a.sum()), int(mask_b.sum()), int((mask_a | mask_b).sum()),
    )

    if strategy == "drop":
        df = _drop_pattern(df, mask_a, "Pattern A")
        mask_b = flag_positive_non_buyers(
            df,
            high_rating_threshold=high_rating_threshold,
            pos_sentiment_threshold=pos_sentiment_threshold,
            mode=sentiment_mode,
        )
        df = _drop_pattern(df, mask_b, "Pattern B")

    else:
        logger.info(
            "Computing product context averages (window=%dd, low=%.1f, high=%.1f)...",
            context_window_days, context_low, context_high,
        )
        df = compute_product_context_avg(df, window_days=context_window_days)
        df = resolve_conflicts_with_context(
            df, mask_a, mask_b,
            context_low=context_low,
            context_high=context_high,
            min_context_reviews=min_context_reviews,
        )

    helper_cols  = ["vader_compound", "context_avg", "context_count"]
    if drop_helper_columns:
        cols_to_drop = [c for c in helper_cols if c in df.columns]
        if cols_to_drop:
            df.drop(columns=cols_to_drop, inplace=True)
    df.reset_index(drop=True, inplace=True)

    if output_path is not None:
        out = Path(output_path)
        df.to_csv(out, index=False)
        logger.info("Cleaned dataset saved to %s", out)

    logger.info("=" * 60)
    logger.info("Pipeline complete. Final shape: %d rows, %d columns.", len(df), df.shape[1])
    logger.info("=" * 60)
    return df

# %% [markdown]
# ## Run

# %%
df_clean = clean_dataset(
    input_path=BASE / "cosmetics_beauty_products_reviews.csv",
    output_path=BASE / "cosmetics_beauty_products_reviews_clean.csv",
    low_rating_threshold=2,
    high_rating_threshold=4,
    strategy="context_aware",       # swap to "drop" for unconditional removal
    sentiment_mode="rating_only",   # swap to "rating_and_sentiment" for conservative flagging
    context_window_days=180,
    context_low=2.5,
    context_high=3.5,
    min_context_reviews=3,
)

# %% [markdown]
# ### Final dataset summary

# %%
print(f"Raw shape:   {df_raw.shape}")
print(f"Clean shape: {df_clean.shape}")
print(f"Rows removed: {len(df_raw) - len(df_clean)} ({(1 - len(df_clean)/len(df_raw))*100:.1f}%)")
print()
print("=== is_a_buyer after cleaning ===")
print(df_clean["is_a_buyer"].value_counts())
print()
print("=== Rating distribution after cleaning ===")
print(df_clean["review_rating"].value_counts().sort_index())
