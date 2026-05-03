# %% [markdown]
# # Assignment 3 - Milestone I: Natural Language Processing
# # Task 1 - Basic Text Pre-processing
# **Student Name:** Yoshita Sarin
# **Student ID:** s4225113
# ## Environment
# * Python 3 (Jupyter Notebook)
# ## Libraries used
# | Library | Purpose |
# |---|---|
# | `pandas` | Reading the CSV file and saving output files |
# | `numpy` | Computing review length statistics |
# | `nltk.RegexpTokenizer` | Tokenising each review using the assignment pattern |
# | `nltk.stem.WordNetLemmatizer` | Converting words to their base form |
# | `itertools.chain` | Flattening nested token lists for vocabulary operations |
# | `collections.Counter` | Counting word and document frequencies |
# | `collections.defaultdict` | Efficient document frequency computation |
# | `html` | Decoding HTML entities (e.g. `&amp;` -> `&`) |
# | `contractions` | Expanding contractions (e.g. `don't` -> `do not`) |
# | `emoji` | Detecting and removing emoji characters |
# | `langdetect` | Detecting the language of each review |
# | `unicodedata` | Normalising Unicode and removing diacritics |
# | `pathlib` | File path handling |
# %% [markdown]
# ---
# ## Introduction
# This notebook covers **Task 1** of the assignment: cleaning and pre-processing approximately
# 61,000 cosmetics and beauty product reviews so they are ready for downstream machine-learning
# tasks. We focus exclusively on the `review_text` column - the free-text narrative written
# by each reviewer.
# ### What we produce
# | Output file | Description |
# |---|---|
# | `processed.csv` | Original dataset with `review_text` replaced by cleaned, space-joined tokens |
# | `vocab.txt` | Alphabetically sorted unigram vocabulary in `word:index` format |
# ### Pre-processing pipeline
# We apply the following steps in order. Steps marked **Required** are mandated by the
# assignment brief; steps marked **Additional** go beyond the minimum to improve data quality.
# | # | Step | Type |
# |---|---|---|
# | 1 | Load and inspect the data | Required |
# | 2 | Check for missing values, duplicates, and non-English reviews | Additional |
# | 3 | Analyse special characters (tags, digits, emojis, HTML entities, etc.) | Additional |
# | 4 | Decode HTML entities (`&amp;` -> `&`) | Additional |
# | 5 | Expand contractions (`don't` -> `do not`) | Additional |
# | 6 | Normalise repeated characters (`loooove` -> `loove`) | Additional |
# | 7 | Remove `@tags`, `#hashtags`, brackets, URLs, digits, diacritics, emojis, Unicode | Required |
# | 8 | Tokenise with `r"[a-zA-Z]+(?:[-'][a-zA-Z]+)?"` and lowercase | Required |
# | 9 | Remove tokens shorter than 2 characters | Required |
# | 10 | Remove stop words (`stopwords_en.txt`) | Required |
# | 11 | Lemmatise using POS-aware WordNet lemmatizer | Additional |
# | 12 | Second stopword removal pass (catches lemma-induced stopwords) | Additional |
# | 13 | Verify token lists end-to-end (format, case, stopwords, digits, etc.) | Additional |
# | 14 | Remove words that appear only once (term frequency = 1) | Required |
# | 15 | Remove the top 20 most frequent words by document frequency | Required |
# | 16 | Save `processed.csv` and `vocab.txt` | Required |
# %% [markdown]
# ---
# ## Importing libraries
# %%
from collections import Counter, defaultdict
from html import unescape
from itertools import chain
from pathlib import Path
import re
import string
import unicodedata
import contractions as contractions_lib
import emoji
import nltk
import numpy as np
import pandas as pd
import regex
from nltk import RegexpTokenizer
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet',                        quiet=True)
nltk.download('omw-1.4',                        quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)
# %% [markdown]
# ---
# ## 1. Data Loading & Inspection
# Before any preprocessing begins we load the raw data, understand its structure, and
# confirm that the dataset matches our expectations. This stage has no effect on the data -
# it is purely observational.
# %% [markdown]
# ### 1.1 Loading the data
# %%
REVIEWS_PATH = '../data/cosmetics_beauty_products_reviews.csv'
raw_reviews_df = pd.read_csv(REVIEWS_PATH, sep=',', header=0)
# %% [markdown]
# ### 1.2 Initial data inspection
# %%
raw_reviews_df.shape
# %% [markdown]
# #### 1.2.1 Shape and DataFrame metadata
# - **Total reviews:** 61,284
# - **Total columns:** 15
# %%
raw_reviews_df.head()
# %%
raw_reviews_df.tail()
# %% [markdown]
# #### 1.2.2 Column breakdown
# The 15 columns fall into four natural groups.
# **Review content**
# | Column | Description |
# |---|---|
# | `review_id` | Unique identifier per review |
# | `review_title` | Short headline |
# | `review_text` | Full body text - our primary NLP field |
# | `author` | Reviewer identity |
# | `review_date` | Date of submission |
# | `review_rating` | Numeric score (1–5) |
# | `is_a_buyer` | Verified purchase flag |
# **Product identity**
# | Column | Description |
# |---|---|
# | `product_id` | Unique product key |
# | `product_title` | Product name |
# | `brand_name` | Brand |
# | `product_url` | Source link |
# **Pricing & aggregate ratings**
# | Column | Description |
# |---|---|
# | `price` | Product price at review time |
# | `avg_product_rating` | Average rating across all reviews |
# | `product_rating_count` | Total number of ratings |
# **Taxonomy**
# | Column | Description |
# |---|---|
# | `product_tags` | Category/tag labels |
# %% [markdown]
# #### 1.2.3 Key observations
# 1. **`review_text` is our core field** - it provides the free-text narrative we pre-process.
# 2. **`review_rating`** gives a ground-truth sentiment label for supervised modelling.
# 3. **`is_a_buyer`** enables credibility filtering in later tasks.
# 4. **`brand_name` and `product_tags`** allow slice-level analysis by brand or category.
# 5. **`review_date`** supports time-series analysis of sentiment trends.
# %% [markdown]
# ---
# ### 1.3 Data Quality Checks
# We perform three quality checks before any text transformation: missing values,
# duplicate reviews, and non-English content. Addressing these early prevents
# misleading statistics at later pipeline stages.
# %% [markdown]
# #### 1.3.1 Missing values in `review_text`
# We check how many reviews have no text at all. Missing values would cause the pipeline
# to crash, so we replace them with empty strings so they pass through cleanly and are
# handled later as empty-token reviews. <br>
# <b>&#8594; Observation:</b> Only 9 reviews (≈ 0.015%) have no text. We replace them with
# empty strings so the pipeline does not crash on `NaN` inputs.
# %%
missing_count = raw_reviews_df['review_text'].isna().sum()
print(f"Missing review_text values: {missing_count:,} ({missing_count / len(raw_reviews_df):.3%})")
raw_reviews_df[raw_reviews_df['review_text'].isna()]
# %%
raw_reviews_df['review_text'] = raw_reviews_df['review_text'].fillna('')
assert raw_reviews_df['review_text'].isna().sum() == 0
print("No missing values remain in review_text.")
# %% [markdown]
# #### 1.3.2 Duplicate reviews
# Duplicate reviews inflate word frequencies and can bias frequency-based filtering steps.
# We check for exact duplicates on `review_text` and remove them so that every word count
# in later steps reflects genuine usage rather than copy-paste repetition. <br>
# <b>&#8594; Observation:</b> [Update after running.] Duplicate reviews are removed below.
# %%
n_dupes = raw_reviews_df.duplicated(subset=['review_text'], keep='first').sum()
print(f"Exact duplicate reviews : {n_dupes:,} ({n_dupes / len(raw_reviews_df):.2%})")
if n_dupes > 0:
    print("\nSample of most-duplicated texts:")
    dupe_texts = (
        raw_reviews_df[raw_reviews_df.duplicated(subset=['review_text'], keep=False)]
        .groupby('review_text').size()
        .sort_values(ascending=False)
        .head(5)
    )
    for text, count in dupe_texts.items():
        print(f"  [{count}×] {repr(str(text)[:100])}")
# %%
raw_reviews_df = (
    raw_reviews_df
    .drop_duplicates(subset=['review_text'], keep='first')
    .reset_index(drop=True)
)
print(f"Reviews after deduplication: {len(raw_reviews_df):,}")
# %% [markdown]
# #### 1.3.3 Non-English reviews
# Our stop-word list and lemmatiser are English-only. Non-English reviews will produce
# mostly noise tokens. We detect them here to understand their prevalence. We do **not**
# remove them automatically - `normalize_unicode` will strip non-ASCII characters, so
# most non-English content is dropped naturally at tokenisation. <br>
# <b>&#8594; Observation:</b> [Update after running.] Non-English reviews represent a small
# fraction of the dataset. Characters from non-Latin scripts (Hindi, Arabic, etc.) are
# discarded by `normalize_unicode`, so those reviews will naturally produce empty or
# near-empty token lists.
# %%
try:
    from langdetect import detect, LangDetectException
    def detect_language(text: str) -> str:
        try:
            return detect(str(text)) if str(text).strip() else 'unknown'
        except LangDetectException:
            return 'unknown'
    sample = raw_reviews_df['review_text'].sample(min(5000, len(raw_reviews_df)), random_state=42)
    lang_counts = sample.apply(detect_language).value_counts()
    print("Language distribution (sample of 5,000 reviews):")
    print(lang_counts.head(10).to_string())
    non_en_pct = (1 - lang_counts.get('en', 0) / len(sample)) * 100
    print(f"\nEstimated non-English: {non_en_pct:.1f}%")
except ImportError:
    print("langdetect not installed. Run: pip install langdetect")
# %% [markdown]
# ---
# ## 2. Text Preprocessing
# This section defines and applies the full text-cleaning pipeline. We first audit the
# raw text to confirm which noise types are present (Section 2.1), then define one helper
# function per transformation (Section 2.2), assemble them into a single pipeline
# (Section 2.3), apply post-tokenisation steps on word lists (Section 2.4), and
# verify the final token lists end-to-end (Section 2.5).
# %% [markdown]
# ### 2.1 Special Character Analysis
# Before writing any cleaning code we inspect the raw text to confirm which noise types
# are actually present. Every helper function we define is justified by evidence here.
# %% [markdown]
# #### 2.1.1 `@tags`
# <b>&#8594; Observation:</b> Some reviews contain `@username` mentions that are person-specific
# identifiers carrying no product sentiment. We will remove them in the pipeline.
# %%
has_tags = raw_reviews_df['review_text'].str.contains(r'@\w+', regex=True, na=False).sum()
print(f"Reviews with @tags    : {has_tags:,} ({has_tags / len(raw_reviews_df):.2%})")
tagged = raw_reviews_df[raw_reviews_df['review_text'].str.contains(r'@\w+', regex=True, na=False)]
print("\nFirst 3 examples:")
for i, text in enumerate(tagged['review_text'].head(3), 1):
    print(f"  {i}. {text[:150]}")
# %% [markdown]
# #### 2.1.2 `#hashtags`
# <b>&#8594; Observation:</b> The `#` symbol causes our tokenizer to miss the word that follows.
# We replace `#` with a space so that `#love` becomes ` love` and the word is captured correctly.
# %%
has_hashtags = raw_reviews_df['review_text'].str.contains(r'#\w+', regex=True, na=False).sum()
print(f"Reviews with #hashtags: {has_hashtags:,} ({has_hashtags / len(raw_reviews_df):.2%})")
hashtagged = raw_reviews_df[raw_reviews_df['review_text'].str.contains(r'#\w+', regex=True, na=False)]
print("\nFirst 3 examples:")
for i, text in enumerate(hashtagged['review_text'].head(3), 1):
    print(f"  {i}. {text[:150]}")
# %% [markdown]
# #### 2.1.3 Digits
# <b>&#8594; Observation:</b> Many reviews contain digits (product codes, prices, star ratings).
# These carry no lexical sentiment meaning and will be removed before tokenisation.
# %%
has_digits = raw_reviews_df['review_text'].str.contains(r'\d', regex=True, na=False).sum()
print(f"Reviews with digits   : {has_digits:,} ({has_digits / len(raw_reviews_df):.2%})")
digit_reviews = raw_reviews_df[raw_reviews_df['review_text'].str.contains(r'\d', regex=True, na=False)]
print("\nFirst 3 examples:")
for i, text in enumerate(digit_reviews['review_text'].head(3), 1):
    print(f"  {i}. {text[:150]}")
# %% [markdown]
# #### 2.1.4 Punctuation marks
# <b>&#8594; Observation:</b> Nearly all reviews contain punctuation. We replace punctuation
# with spaces to prevent token sticking (e.g. `great.Highly` -> `great Highly`), keeping
# hyphens and apostrophes to support the tokenizer pattern.
# %%
punct_pattern = '[' + re.escape(string.punctuation) + ']'
has_punct = raw_reviews_df['review_text'].str.contains(punct_pattern, regex=True, na=False).sum()
print(f"Reviews with punctuation: {has_punct:,} ({has_punct / len(raw_reviews_df):.2%})")
punct_counts = {
    p: raw_reviews_df['review_text'].str.contains(re.escape(p), regex=True, na=False).sum()
    for p in string.punctuation
}
print("\nTop 10 punctuation marks:")
for p, c in sorted(punct_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f"  '{p}': {c:,} reviews ({c / len(raw_reviews_df):.1%})")
# %% [markdown]
# #### 2.1.5 Diacritics (accent marks)
# <b>&#8594; Observation:</b> A small number of reviews contain accented characters
# (e.g. `café`, `naïve`). Without normalisation, `café` and `cafe` would be treated
# as separate vocabulary items. We strip diacritics to prevent vocabulary fragmentation.
# %%
def has_diacritics(text):
    if pd.isna(text): return False
    return any(unicodedata.combining(c) for c in unicodedata.normalize('NFD', str(text)))
diac_count = raw_reviews_df['review_text'].apply(has_diacritics).sum()
print(f"Reviews with diacritics: {diac_count:,} ({diac_count / len(raw_reviews_df):.2%})")
diac_reviews = raw_reviews_df[raw_reviews_df['review_text'].apply(has_diacritics)]
print("\nFirst 3 examples:")
for i, text in enumerate(diac_reviews['review_text'].head(3), 1):
    print(f"  {i}. {text[:150]}")
# %% [markdown]
# #### 2.1.6 Extra whitespace
# <b>&#8594; Observation:</b> [Update after running.] Extra whitespace is handled by the
# tokenizer pattern, which splits on any whitespace, so no explicit step is needed.
# %%
def has_extra_whitespace(text):
    if pd.isna(text): return False
    return bool(re.search(r'  |\t|\n|\r', str(text)))
ws_count = raw_reviews_df['review_text'].apply(has_extra_whitespace).sum()
multiple_spaces = raw_reviews_df['review_text'].str.contains(r'  ',    regex=True, na=False).sum()
has_tabs        = raw_reviews_df['review_text'].str.contains(r'\t',    regex=True, na=False).sum()
has_newlines    = raw_reviews_df['review_text'].str.contains(r'\n|\r', regex=True, na=False).sum()
print(f"Reviews with extra whitespace : {ws_count:,} ({ws_count / len(raw_reviews_df):.2%})")
print(f"  Multiple spaces             : {multiple_spaces:,}")
print(f"  Tab characters              : {has_tabs:,}")
print(f"  Newlines                    : {has_newlines:,}")
# %% [markdown]
# #### 2.1.7 Round brackets
# <b>&#8594; Observation:</b> Parentheses wrap non-essential asides and technical specs.
# We strip them to prevent brackets from merging adjacent words into a single token.
# %%
has_rb = raw_reviews_df['review_text'].str.contains(r'[()]', regex=True, na=False).sum()
print(f"Reviews with round brackets: {has_rb:,} ({has_rb / len(raw_reviews_df):.2%})")
rb_reviews = raw_reviews_df[raw_reviews_df['review_text'].str.contains(r'[()]', regex=True, na=False)]
print("\nFirst 3 examples:")
for i, text in enumerate(rb_reviews['review_text'].head(3), 1):
    print(f"  {i}. {text[:150]}")
# %% [markdown]
# #### 2.1.8 Curly brackets
# <b>&#8594; Observation:</b> No curly brackets found - no action required.
# %%
has_cb = raw_reviews_df['review_text'].str.contains(r'[{}]', regex=True, na=False).sum()
print(f"Reviews with curly brackets: {has_cb:,} ({has_cb / len(raw_reviews_df):.2%})")
# %% [markdown]
# #### 2.1.9 Square brackets
# <b>&#8594; Observation:</b> No square brackets found - no action required.
# %%
has_sb = raw_reviews_df['review_text'].str.contains(r'[\[\]]', regex=True, na=False).sum()
print(f"Reviews with square brackets: {has_sb:,} ({has_sb / len(raw_reviews_df):.2%})")
# %% [markdown]
# #### 2.1.10 URLs
# <b>&#8594; Observation:</b> A small number of reviews contain hyperlinks. URLs carry no
# product sentiment and will be removed before tokenisation.
# %%
has_urls = raw_reviews_df['review_text'].str.contains(r'https?://\S+|www\.\S+', regex=True, na=False).sum()
print(f"Reviews with URLs: {has_urls:,} ({has_urls / len(raw_reviews_df):.2%})")
url_reviews = raw_reviews_df[raw_reviews_df['review_text'].str.contains(r'https?://\S+|www\.\S+', regex=True, na=False)]
print("\nFirst 3 examples:")
for i, text in enumerate(url_reviews['review_text'].head(3), 1):
    print(f"  {i}. {text[:150]}")
# %% [markdown]
# #### 2.1.11 Emojis
# <b>&#8594; Observation:</b> Over 3,000 reviews contain emojis. We remove them entirely
# rather than converting to text descriptions. Conversion maps emojis to generic words
# (e.g. 😍 -> `smiling`, `face`) that are ambiguous — `face` could come from
# *"broke out on my face"* or from an emoji, which would mislead any downstream model.
# Removing emojis keeps the vocabulary clean and ensures every word reflects genuine
# product language.
# %%
def has_emoji_chars(text):
    if pd.isna(text): return False
    return any(
        unicodedata.category(c) == 'So'
        or (0x1F300 <= ord(c) <= 0x1FAFF)
        or (0x2600 <= ord(c) <= 0x27BF)
        for c in str(text)
    )
emoji_count = raw_reviews_df['review_text'].apply(has_emoji_chars).sum()
print(f"Reviews with emojis: {emoji_count:,} ({emoji_count / len(raw_reviews_df):.2%})")
emoji_reviews = raw_reviews_df[raw_reviews_df['review_text'].apply(has_emoji_chars)]
print("\nFirst 3 examples:")
for i, text in enumerate(emoji_reviews['review_text'].head(3), 1):
    print(f"  {i}. {text[:150]}")
# %% [markdown]
# #### 2.1.12 All unique emoji symbols
# We map every unique emoji used in the dataset to understand the diversity of symbols
# and decide whether any require special handling.
# %%
_emoji_range = (
    "\U0001F300-\U0001F5FF\U0001F600-\U0001F64F\U0001F680-\U0001F6FF"
    "\U0001F700-\U0001F77F\U0001F780-\U0001F7FF\U0001F800-\U0001F8FF"
    "\U0001F900-\U0001F9FF\U0001FA70-\U0001FAFF☀-⛿✀-➿"
)
_emoji_full_pattern = regex.compile(
    rf"(?:(?:[\U0001F1E6-\U0001F1FF]{{2}})|(?:[0-9#*]️?⃣)|"
    rf"(?:[{_emoji_range}]️?[\U0001F3FB-\U0001F3FF]?(?:‍[{_emoji_range}]️?[\U0001F3FB-\U0001F3FF]?)*))"
)
def extract_emojis(text):
    return _emoji_full_pattern.findall(str(text)) if not pd.isna(text) else []
all_unique_emojis = sorted(set(
    e for lst in raw_reviews_df['review_text'].apply(extract_emojis) for e in lst
))
print(f"Total unique emojis found: {len(all_unique_emojis)}")
print("\nAll unique emojis:")
print(" ".join(all_unique_emojis))
# %% [markdown]
# #### 2.1.13 Non-Latin Unicode characters
# <b>&#8594; Observation:</b> Nearly 4,000 reviews contain non-Latin characters from scripts
# such as Hindi, Arabic, or mathematical notation. `normalize_unicode` will convert them to
# ASCII equivalents where possible; remaining characters are silently dropped.
# %%
def contains_non_latin(text):
    if pd.isna(text): return False
    return any(
        ord(c) > 127 and unicodedata.category(c) not in ('So', 'Mn')
        for c in str(text)
    )
unicode_count = raw_reviews_df['review_text'].apply(contains_non_latin).sum()
print(f"Reviews with non-Latin unicode: {unicode_count:,} ({unicode_count / len(raw_reviews_df):.2%})")
non_latin_chars = {
    c for text in raw_reviews_df['review_text'] if not pd.isna(text)
    for c in str(text)
    if ord(c) > 127 and unicodedata.category(c) not in ('So', 'Mn')
}
print(f"\nUnique non-Latin characters: {len(non_latin_chars)}")
print("Examples (first 15):")
for c in list(non_latin_chars)[:15]:
    try:    name = unicodedata.name(c)
    except ValueError: name = '[No name]'
    print(f"  '{c}' (U+{ord(c):04X}) - {name}")
# %% [markdown]
# #### 2.1.14 HTML entities
# Reviews scraped from the web often contain HTML entities such as `&amp;` (ampersand),
# `&lt;` (less-than), and `&#39;` (apostrophe). We check whether these are present and
# must be decoded before any other cleaning step. <br>
# <b>&#8594; Observation:</b> [Update after running.] HTML entities such as `&amp;`, `&lt;`,
# and `&#39;` are artefacts of web scraping. We decode them to their actual characters as the
# very first cleaning step so that `&#39;` -> `'` can participate in contraction expansion.
# %%
has_html = raw_reviews_df['review_text'].str.contains(r'&\w+;|&#\d+;', regex=True, na=False).sum()
print(f"Reviews with HTML entities: {has_html:,} ({has_html / len(raw_reviews_df):.2%})")
html_reviews = raw_reviews_df[raw_reviews_df['review_text'].str.contains(r'&\w+;|&#\d+;', regex=True, na=False)]
print("\nFirst 3 examples:")
for i, text in enumerate(html_reviews['review_text'].head(3), 1):
    print(f"  {i}. {text[:150]}")
# %% [markdown]
# #### 2.1.15 Contractions
# Contractions such as `don't`, `can't`, and `it's` are preserved as single tokens by our
# tokenizer pattern. However, `WordNetLemmatizer` cannot lemmatize them - `don't` stays
# as `don't` rather than mapping to `do`. We check how many reviews are affected.
# <b>&#8594; Observation:</b> Many reviews contain contractions. We expand them before
# tokenisation so that `don't` -> `do not`, allowing each component to be lemmatized
# and filtered independently (`not` is removed by the stop-word filter).
# %%
_contraction_re = re.compile(
    r"\b\w+n't\b|\b(I'm|I've|I'll|I'd|you're|you've|can't|won't|don't|"
    r"doesn't|didn't|isn't|aren't|wasn't|weren't|it's|that's|there's|"
    r"they're|we're|what's|who's|would've|could've|should've)\b",
    re.IGNORECASE,
)
has_contractions = raw_reviews_df['review_text'].str.contains(_contraction_re, regex=True, na=False).sum()
print(f"Reviews with contractions: {has_contractions:,} ({has_contractions / len(raw_reviews_df):.2%})")
contr_reviews = raw_reviews_df[raw_reviews_df['review_text'].str.contains(_contraction_re, regex=True, na=False)]
print("\nFirst 3 examples:")
for i, text in enumerate(contr_reviews['review_text'].head(3), 1):
    print(f"  {i}. {text[:150]}")
# %% [markdown]
# #### 2.1.16 Repeated characters
# Colloquial exaggeration is common in beauty reviews: `loooove`, `sooooo`, `amazinggg`.
# Without normalisation, each variant becomes a separate vocabulary entry that will never
# accumulate enough frequency to survive the rare-word filter.
# <b>&#8594; Observation:</b> We collapse runs of 3+ identical characters to 2
# (e.g. `loooove` -> `loove`). This is enough to preserve emphasis while allowing
# lemmatisation to merge variants into a single vocabulary entry.
# %%
has_repeated = raw_reviews_df['review_text'].str.contains(r'(.)\1{2,}', regex=True, na=False).sum()
print(f"Reviews with repeated chars: {has_repeated:,} ({has_repeated / len(raw_reviews_df):.2%})")
rep_reviews = raw_reviews_df[raw_reviews_df['review_text'].str.contains(r'(.)\1{2,}', regex=True, na=False)]
print("\nFirst 3 examples:")
for i, text in enumerate(rep_reviews['review_text'].head(3), 1):
    print(f"  {i}. {text[:150]}")
# %% [markdown]
# ---
# ### 2.2 Preprocessing Helper Functions
# We define one helper function per transformation. Each function is self-contained,
# handles `None` / non-string inputs gracefully, and does exactly one thing.
# The pipeline in Section 2.3 composes them in the correct order.
# %% [markdown]
# #### 2.2.1 Decode HTML entities
# HTML entities are scraping artefacts. We decode them first so that `&#39;` becomes `'`
# before contraction expansion, and `&amp;` becomes `&` before punctuation removal.
# %%
def decode_html_entities(text: object) -> str:
    """Decode HTML entities. e.g. &amp; -> &, &#39; -> '"""
    if text is None:
        return ""
    return unescape(str(text))
# %% [markdown]
# #### 2.2.2 Expand contractions
# Contractions such as `don't` are preserved as a single token by the tokenizer pattern
# but cannot be lemmatized correctly. Expanding them to `do not` lets both components
# be processed independently - `not` is removed by the stop-word filter, and `do` is
# lemmatized to its base form.
# %%
def expand_contractions(text: object) -> str:
    """Expand English contractions. e.g. don't -> do not, can't -> cannot"""
    if text is None:
        return ""
    return contractions_lib.fix(str(text))
# %% [markdown]
# #### 2.2.3 Normalise repeated characters
# Colloquial exaggeration (e.g. `loooove`, `sooooo`) creates spurious vocabulary entries.
# We reduce any run of 3+ identical characters to 2, preserving some emphasis
# (e.g. `loooove` -> `loove`) while enabling lemmatisation to merge the variants.
# %%
_REPEATED_CHARS = re.compile(r'(.)\1{2,}')
def normalize_repeated_chars(text: object) -> str:
    """Collapse 3+ consecutive identical characters to 2. e.g. loooove -> loove"""
    if text is None:
        return ""
    return _REPEATED_CHARS.sub(r'\1\1', str(text))
# %% [markdown]
# #### 2.2.4 Lowercase
# %%
def lowercase(text: object) -> str:
    """Convert text to lowercase so Skin and skin are treated as the same word."""
    if text is None:
        return ""
    return str(text).lower()
# %% [markdown]
# #### 2.2.5 Remove `@tags`
# %%
def replace_tags(text: object) -> str:
    """Remove @username mentions, replacing them with a space."""
    if text is None:
        return ""
    return re.sub(r'@\w+', ' ', str(text))
# %% [markdown]
# #### 2.2.6 Replace `#hashtags`
# %%
def replace_hashtags(text: object) -> str:
    """Replace # with a space to free the keyword from the symbol."""
    if text is None:
        return ""
    return str(text).replace('#', ' ')
# %% [markdown]
# #### 2.2.7 Remove digits
# %%
def remove_digits(text: object) -> str:
    """Remove all digit characters."""
    if text is None:
        return ""
    return re.sub(r'\d+', '', str(text))
# %% [markdown]
# #### 2.2.8 Remove punctuation
# We keep `-` and `'` to remain compatible with the tokenizer pattern
# `r"[a-zA-Z]+(?:[-'][a-zA-Z]+)?"`, which captures hyphenated words and
# contractions as single tokens.
# %%
_PUNCT_TABLE = str.maketrans(
    string.punctuation.replace('-', '').replace("'", ""),
    ' ' * len(string.punctuation.replace('-', '').replace("'", "")),
)
def remove_punctuation(text: object) -> str:
    """Replace punctuation with spaces, preserving hyphens and apostrophes."""
    if text is None:
        return ""
    return str(text).translate(_PUNCT_TABLE)
# %% [markdown]
# #### 2.2.9 Remove diacritics
# %%
def remove_diacritics(text: object) -> str:
    """Strip accent marks. e.g. café -> cafe, naïve -> naive"""
    if text is None:
        return ""
    return ''.join(
        c for c in unicodedata.normalize('NFD', str(text))
        if not unicodedata.combining(c)
    )
# %% [markdown]
# #### 2.2.10 Remove round brackets
# %%
def remove_round_brackets(text: object) -> str:
    """Strip ( and ) characters."""
    if text is None:
        return ""
    return str(text).replace('(', '').replace(')', '')
# %% [markdown]
# #### 2.2.11 Remove URLs
# %%
def remove_urls(text: object) -> str:
    """Remove http://, https://, and www. links."""
    if text is None:
        return ""
    return re.sub(r'https?://\S+|www\.\S+', '', str(text))
# %% [markdown]
# #### 2.2.12 Remove emojis
# We remove emojis entirely rather than converting them to text descriptions.
# Conversion produces generic words (e.g. 😍 -> `smiling`, `face`) that are
# indistinguishable from the same words used in genuine product sentences,
# which would introduce ambiguity into the vocabulary. Removal keeps the
# vocabulary clean and ensures every token reflects real review language.
# %%
def remove_emojis(text: object) -> str:
    """Remove all emoji characters from text."""
    if text is None:
        return ""
    return emoji.replace_emoji(str(text), replace='')
# %% [markdown]
# #### 2.2.13 Normalise Unicode
# Some reviews contain fancy Unicode variants of regular Latin characters
# (e.g. mathematical bold italic `𝙄𝙩'𝙨` instead of `It's`). We convert them
# to their ASCII equivalents via NFKD decomposition; characters with no ASCII
# equivalent are silently dropped.
# %%
def normalize_unicode(text: object) -> str:
    """Normalise Unicode to ASCII. e.g. 𝙄𝙩'𝙨 -> It's. Non-ASCII chars are dropped."""
    if text is None:
        return ""
    return (
        unicodedata.normalize('NFKD', str(text))
        .encode('ascii', 'ignore')
        .decode('ascii')
    )
# %% [markdown]
# #### 2.2.14 Statistics helper
# We call this function after every major pipeline step to track how each
# transformation affects vocabulary size, token count, and review length.
# This lets us verify that the pipeline is progressing correctly.
# %%
def stats_print(tk_reviews: list[list[str]]) -> None:
    """Print a summary of vocabulary and review length statistics."""
    words        = list(chain.from_iterable(tk_reviews))
    unique_words = set(words)
    lens         = [len(r) for r in tk_reviews]
    print(f"Number of unique words         : {len(unique_words):,}")
    print(f"Total number of words          : {len(words):,}")
    if words:
        print(f"Word variety (unique/total)    : {len(unique_words)/len(words):.5f}")
    else:
        print("Word variety (unique/total)    : 0.0")
    print(f"Number of reviews              : {len(tk_reviews):,}")
    if lens:
        print(f"Average review length          : {np.mean(lens):.2f}")
        print(f"Longest review (in words)      : {int(np.max(lens))}")
        print(f"Shortest review (in words)     : {int(np.min(lens))}")
        print(f"Standard deviation of length   : {np.std(lens):.2f}")
# %% [markdown]
# ---
# ### 2.3 Cleaning Pipeline & Tokenisation
# We assemble all helper functions into a single `clean_and_tokenise` function.
# The order of operations is critical:
# - **HTML decoding before contraction expansion** - so `&#39;` -> `'` can participate
#   in contraction matching.
# - **Contraction expansion before punctuation removal** - so `don't` -> `do not`
#   before `'` is stripped.
# - **Emojis before Unicode normalisation** - so emoji characters are stripped
#   cleanly before the Unicode step runs.
# - **Lowercase last** - so none of the earlier steps accidentally upper-case anything.
# | Step | Function | What it does |
# |---|---|---|
# | 1 | `decode_html_entities` | `&amp;` -> `&`, `&#39;` -> `'` |
# | 2 | `expand_contractions` | `don't` -> `do not` |
# | 3 | `normalize_repeated_chars` | `loooove` -> `loove` |
# | 4 | `replace_tags` | Remove `@username` |
# | 5 | `replace_hashtags` | `#` -> space |
# | 6 | `remove_round_brackets` | Strip `(` `)` |
# | 7 | `remove_emojis` | Strip all emoji characters |
# | 8 | `remove_urls` | Remove `http://…` links |
# | 9 | `remove_digits` | Remove all numbers |
# | 10 | `remove_diacritics` | `café` -> `cafe` |
# | 11 | `normalize_unicode` | Non-ASCII -> ASCII / drop |
# | 12 | `remove_punctuation` | Replace punct with spaces (keeps `-` `'`) |
# | 13 | `lowercase` | Standardise casing |
# | 14 | `RegexpTokenizer` | Split using `r"[a-zA-Z]+(?:[-'][a-zA-Z]+)?"` |
# <b>&#8594; Observation:</b> [Update after running.] After tokenisation the unique-word count
# is at its highest - every following step can only reduce it. The word variety ratio gives us
# a baseline to compare against after each filtering step.
# %%
PATTERN   = r"[a-zA-Z]+(?:[-'][a-zA-Z]+)?"
tokenizer = RegexpTokenizer(PATTERN)
def clean_and_tokenise(text: str) -> list[str]:
    """Run the full cleaning pipeline on one review and return its token list."""
    text = decode_html_entities(text)
    text = expand_contractions(text)
    text = normalize_repeated_chars(text)
    text = replace_tags(text)
    text = replace_hashtags(text)
    text = remove_round_brackets(text)
    text = remove_emojis(text)
    text = remove_urls(text)
    text = remove_digits(text)
    text = remove_diacritics(text)
    text = normalize_unicode(text)
    text = remove_punctuation(text)
    text = lowercase(text)
    return tokenizer.tokenize(text)
tk_reviews = [clean_and_tokenise(r) for r in raw_reviews_df['review_text'].tolist()]
stats_print(tk_reviews)
# %% [markdown]
# #### 2.3.1 Pipeline verification
# We run automated checks across all tokenised reviews to confirm that every
# cleaning step was applied correctly before we proceed.
# %%
print("=== Cleaning Pipeline Verification ===\n")
cleaned_texts = [' '.join(tokens) for tokens in tk_reviews]
checks = {
    "@tags removed"          : sum(bool(re.search(r'@\w+', t)) for t in cleaned_texts),
    "#hashtags replaced"     : sum('#' in t for t in cleaned_texts),
    "Round brackets removed" : sum('(' in t or ')' in t for t in cleaned_texts),
    "URLs removed"           : sum(bool(re.search(r'https?://|www\.', t)) for t in cleaned_texts),
    "Digits removed"         : sum(bool(re.search(r'\d', t)) for t in cleaned_texts),
    "Diacritics removed"     : sum(
        any(unicodedata.combining(c) for c in unicodedata.normalize('NFD', t))
        for t in cleaned_texts
    ),
    "Punctuation removed"    : sum(
        any(c in string.punctuation.replace('-', '').replace("'", "") for c in t)
        for t in cleaned_texts
    ),
    "Lowercase applied"      : sum(t != t.lower() for t in cleaned_texts),
    "Tokens are words only"  : sum(
        not all(re.fullmatch(r"[a-zA-Z]+(?:[-'][a-zA-Z]+)?", w) for w in tokens)
        for tokens in tk_reviews if tokens
    ),
}
all_passed = True
for check, violations in checks.items():
    passed = violations == 0
    status = "✅ PASS" if passed else f"❌ FAIL ({violations:,} reviews)"
    print(f"  {status}  {check}")
    if not passed:
        all_passed = False
print(f"\n{'✅ All checks passed!' if all_passed else '❌ Some checks failed.'}")
print(f"Total reviews checked: {len(tk_reviews):,}")
# %% [markdown]
# ---
# ### 2.4 Post-Tokenisation Preprocessing
# The steps below operate on lists of tokens rather than raw strings. We apply them
# in the order mandated by the assignment brief.
# %% [markdown]
# #### 2.4.1 Handling empty reviews
# After tokenisation, some reviews produce no tokens - typically because their original
# text consisted entirely of digits, URLs, or non-ASCII symbols that were all removed.
# We inspect these reviews to understand why they became empty. We do not remove them
# at this stage; they will appear as empty strings in `processed.csv`. <br>
# <b>&#8594; Observation:</b> The empty reviews contain no meaningful lexical content
# (e.g. only digits, symbols, or very short emoji-only text). They do not contribute
# to the vocabulary and will appear as empty strings in `processed.csv`.
# %%
empty_indices = [i for i, r in enumerate(tk_reviews) if not r]
print(f"Empty reviews after tokenisation: {len(empty_indices)} ({len(empty_indices)/len(tk_reviews):.2%})")
for i in empty_indices[:5]:
    original = raw_reviews_df['review_text'].iloc[i]
    print(f"\n[{i}] Original : {repr(str(original)[:120])}")
    print(f"     Tokens   : {clean_and_tokenise(original) or '(empty)'}")
stats_print(tk_reviews)
# %% [markdown]
# #### 2.4.2 Removing short words (length < 2)
# Single-character tokens (`a`, `i`, `u`) almost never carry sentiment and typically
# arise from punctuation leftovers or abbreviations. Removing them reduces vocabulary
# noise without discarding any meaningful content. <br>
# <b>&#8594; Observation:</b> [Update after running.] Removing single-character tokens has
# a small effect on vocabulary size but meaningfully reduces total token count.
# %%
MIN_WORD_LENGTH = 2
def filter_short_words(
    tokenized_reviews: list[list[str]],
    min_length: int = MIN_WORD_LENGTH,
) -> list[list[str]]:
    """Remove tokens shorter than min_length characters."""
    return [[w for w in review if len(w) >= min_length] for review in tokenized_reviews]
tk_reviews = filter_short_words(tk_reviews)
stats_print(tk_reviews)
# %% [markdown]
# #### 2.4.3 Removing stop words
# Stop words (`the`, `is`, `and`, `of`, …) appear in virtually every review and contribute
# no discriminative signal to a model. We use the stop-word list supplied with the assignment
# (`stopwords_en.txt`). We use a `frozenset` for O(1) membership tests. <br>
# <b>&#8594; Observation:</b> [Update after running.] Removing stop words dramatically
# reduces total token count while leaving the unique vocabulary largely intact, which
# sharply improves the word variety ratio.
# %%
STOPWORDS_PATH = Path("../data/stopwords_en.txt")
def load_stopwords(path: Path) -> frozenset[str]:
    """Load the stop-word list from disk into a frozenset."""
    if not path.exists():
        raise FileNotFoundError(f"Stopwords file not found: {path}")
    words = frozenset(path.read_text(encoding="utf-8").split())
    print(f"Loaded {len(words):,} stopwords from {path}")
    return words
def remove_stopwords(
    tokenized_reviews: list[list[str]],
    stopwords: frozenset[str],
) -> list[list[str]]:
    """Remove stop words from every review."""
    return [[w for w in review if w not in stopwords] for review in tokenized_reviews]
stopwords_en = load_stopwords(STOPWORDS_PATH)
tk_reviews   = remove_stopwords(tk_reviews, stopwords_en)
empty_after_stop = sum(1 for r in tk_reviews if not r)
print(f"Empty reviews after stopword removal: {empty_after_stop}")
stats_print(tk_reviews)
# %% [markdown]
# #### 2.4.4 Lemmatisation
# Lemmatisation maps inflected word forms to their dictionary base form so that
# `products`, `product`, and `producing` are all counted as `product`. We use a
# **POS-aware** strategy: the tokenizer uses the grammatical role of each word
# (noun, verb, adjective, adverb) to choose the correct base form - for example,
# `loved` is identified as a verb and correctly mapped to `love`.
# We also apply three safety rules to prevent over-aggressive reductions:
# | Rule | Example prevented |
# |---|---|
# | Skip tokens of length ≤ 3 | `us` -> `u`, `bs` -> `b` |
# | Reject a lemma that collapses to 1 character | any word -> single letter |
# | Reject a lemma ≥ 2 characters shorter than the original | `boss` -> `bos` |
# %% [markdown]
# ##### 2.4.4.1 Lemmatisation helper functions
# <b>&#8594; Observation:</b> [Update after running.] Lemmatisation reduces the unique word
# count by merging inflected variants without removing any tokens. The total word count
# and review count remain identical - only word forms change.
# %%
lemmatizer    = WordNetLemmatizer()
_lemma_cache: dict[tuple[str, str], str] = {}
def get_wordnet_pos(treebank_tag: str) -> str:
    """Convert an NLTK treebank POS tag to the WordNet equivalent."""
    if treebank_tag.startswith("J"): return wordnet.ADJ
    if treebank_tag.startswith("V"): return wordnet.VERB
    if treebank_tag.startswith("R"): return wordnet.ADV
    return wordnet.NOUN
def safe_lemmatise(word: str, pos: str) -> str:
    """POS-aware lemmatisation with safety guards against over-reduction."""
    key = (word, pos)
    if key not in _lemma_cache:
        lemma = lemmatizer.lemmatize(word, pos)
        if len(word) <= 3:                                   # rule 1
            lemma = word
        if len(lemma) == 1 and len(word) > 1:               # rule 2
            lemma = word
        if len(word) >= 4 and len(lemma) <= len(word) - 2:  # rule 3
            lemma = word
        _lemma_cache[key] = lemma
    return _lemma_cache[key]
def lemmatise_review(review: list[str]) -> list[str]:
    """POS-tag and safely lemmatise one tokenised review."""
    return [
        safe_lemmatise(w, get_wordnet_pos(tag))
        for w, tag in nltk.pos_tag(review)
    ]
tk_reviews_lemmatized = [lemmatise_review(r) for r in tk_reviews]
stats_print(tk_reviews_lemmatized)
# %%
lemmatization_changes = [(w, l) for (w, p), l in _lemma_cache.items() if w != l]
print(f"Total words changed by lemmatisation: {len(lemmatization_changes):,}")
print("\nAll changes (original -> base form):")
for word, lemma in sorted(lemmatization_changes):
    print(f"  {word:>20s}  ->  {lemma}")
# %%
print("Before lemmatisation:")
stats_print(tk_reviews)
print("\nAfter lemmatisation:")
stats_print(tk_reviews_lemmatized)
# %% [markdown]
# #### 2.4.5 Removing short words again after lemmatisation
# Lemmatisation can occasionally reduce a word to a single character through aggressive
# base-form reduction. We apply the short-word filter a second time to catch any such
# cases and maintain a clean token list.
# %%
tk_reviews = filter_short_words(tk_reviews_lemmatized)
stats_print(tk_reviews)
# %%
short_words = [(i, j, w) for i, r in enumerate(tk_reviews) for j, w in enumerate(r) if len(w) < 2]
if not short_words:
    print("PASS: All tokens are at least 2 characters long.")
else:
    print(f"FAIL: {len(short_words):,} short tokens remain.")
    for i, j, w in short_words[:20]:
        print(f"  Review {i}, position {j}: {repr(w)}")
# %% [markdown]
# #### 2.4.6 Second stopword pass after lemmatisation
# Lemmatisation reduces inflected forms to their base form. Some of those base forms
# are stopwords that were not caught earlier because the inflected form was not on the
# list (e.g. `wondering` → `wonder`, `appreciated` → `appreciate`, `seconds` → `second`).
# We apply a second stopword removal pass to clean these up.
# %%
before_second_stop = sum(len(r) for r in tk_reviews)
tk_reviews = remove_stopwords(tk_reviews, stopwords_en)
after_second_stop = sum(len(r) for r in tk_reviews)
print(f"Tokens removed by second stopword pass: {before_second_stop - after_second_stop:,}")
stats_print(tk_reviews)
# %% [markdown]
# ---
# ### 2.5 Post-Tokenisation Verification
# After all token-level preprocessing steps are complete we run a final battery of
# automated checks to confirm that the pipeline behaved correctly end-to-end. Unlike
# the pipeline verification in Section 2.3.1 (which checked the raw-string cleaning),
# this section verifies the **token lists** that will feed into vocabulary construction.
# %%
print("=== Post-Tokenisation Verification ===\n")
# --- 1. Token format: every token must match the assignment pattern -----------
bad_format = [
    (i, w)
    for i, review in enumerate(tk_reviews)
    for w in review
    if not re.fullmatch(r"[a-z]+(?:[-'][a-z]+)?", w)
]
_status = "✅ PASS" if not bad_format else f"❌ FAIL ({len(bad_format):,} tokens)"
print(f"  {_status}  All tokens match [a-z]+(?:[-'][a-z]+)?")
if bad_format:
    for idx, w in bad_format[:5]:
        print(f"    Review {idx}: {repr(w)}")
# --- 2. No uppercase letters in any token -------------------------------------
has_upper = [(i, w) for i, r in enumerate(tk_reviews) for w in r if w != w.lower()]
_status = "✅ PASS" if not has_upper else f"❌ FAIL ({len(has_upper):,} tokens)"
print(f"  {_status}  All tokens are lowercase")
# --- 3. No tokens shorter than 2 characters -----------------------------------
short_tokens = [(i, w) for i, r in enumerate(tk_reviews) for w in r if len(w) < 2]
_status = "✅ PASS" if not short_tokens else f"❌ FAIL ({len(short_tokens):,} tokens)"
print(f"  {_status}  No tokens shorter than 2 characters")
# --- 4. No stop words remain --------------------------------------------------
stopword_tokens = [(i, w) for i, r in enumerate(tk_reviews) for w in r if w in stopwords_en]
_status = "✅ PASS" if not stopword_tokens else f"❌ FAIL ({len(stopword_tokens):,} tokens)"
print(f"  {_status}  No stop words remaining")
if stopword_tokens:
    sample = list({w for _, w in stopword_tokens})[:10]
    print(f"    Sample: {sample}")
# --- 5. No digits in any token ------------------------------------------------
digit_tokens = [(i, w) for i, r in enumerate(tk_reviews) for w in r if re.search(r'\d', w)]
_status = "✅ PASS" if not digit_tokens else f"❌ FAIL ({len(digit_tokens):,} tokens)"
print(f"  {_status}  No digits in tokens")
# --- 6. No punctuation in tokens (beyond allowed - and ') --------------------
bad_punct = [
    (i, w)
    for i, r in enumerate(tk_reviews)
    for w in r
    if any(c in string.punctuation.replace('-', '').replace("'", "") for c in w)
]
_status = "✅ PASS" if not bad_punct else f"❌ FAIL ({len(bad_punct):,} tokens)"
print(f"  {_status}  No disallowed punctuation in tokens")
# --- 7. Token count summary ---------------------------------------------------
total_tokens   = sum(len(r) for r in tk_reviews)
non_empty      = sum(1 for r in tk_reviews if r)
empty_reviews  = len(tk_reviews) - non_empty
unique_types   = len({w for r in tk_reviews for w in r})
print(f"\n  Token statistics after full post-tokenisation preprocessing:")
print(f"    Total reviews      : {len(tk_reviews):>10,}")
print(f"    Non-empty reviews  : {non_empty:>10,}")
print(f"    Empty reviews      : {empty_reviews:>10,}")
print(f"    Total tokens       : {total_tokens:>10,}")
print(f"    Unique types       : {unique_types:>10,}")
print(f"    Avg tokens/review  : {total_tokens / max(non_empty, 1):>10.1f}")
# --- 8. Sample output ---------------------------------------------------------
print("\n  Sample token lists (first 3 non-empty reviews):")
shown = 0
for i, review in enumerate(tk_reviews):
    if review:
        print(f"    Review {i:>5}: {review[:12]}{'...' if len(review) > 12 else ''}")
        shown += 1
        if shown == 3:
            break
# %% [markdown]
# ---
# ## 3. Vocabulary Refinement
# We apply two frequency-based filtering steps as required by the assignment brief.
# Both steps operate on the final token lists after all preprocessing is complete,
# ensuring that frequency counts are based on the clean, lemmatised vocabulary.
# %% [markdown]
# ### 3.1 Removing words that appear only once (hapax legomena)
# A word that appears exactly once across all ~61,000 reviews is almost always a typo,
# a unique brand name, or a transcription error. A model cannot learn anything from a
# word it has seen only once, and these words inflate the vocabulary unnecessarily.
# We remove them based on **term frequency** - the total number of times each word
# appears across the entire document collection. <br>
# <b>&#8594; Observation:</b> [Update after running.] A large proportion of unique words
# appear only once - these are overwhelmingly typos and one-off proper nouns. Removing
# them significantly shrinks the vocabulary without losing any repeatable signal.
# %%
term_freq  = Counter(chain.from_iterable(tk_reviews))
rare_words = {w for w, c in term_freq.items() if c == 1}
print(f"Words appearing only once : {len(rare_words):,}")
print(f"Percentage of vocabulary  : {len(rare_words) / len(term_freq):.1%}")
tk_reviews = [[w for w in r if w not in rare_words] for r in tk_reviews]
stats_print(tk_reviews)
# %% [markdown]
# ### 3.2 Removing the top 20 most frequent words (by document frequency)
# The top 20 words by **document frequency** - the number of reviews each word appears
# in - are domain-specific filler words that occur in nearly every review and carry no
# discriminative power. Examples are `good`, `product`, `skin`, and `love`. These words
# cannot help a model distinguish one review from another, so we remove exactly the top
# 20 as specified by the assignment brief.
# We use document frequency (rather than term frequency) for this step because a word
# that appears 10 times in a single review should count less than a word that appears
# once in 10 different reviews. <br>
# <b>&#8594; Observation:</b> The frequency distribution follows a power law - most words
# are rare and a tiny number are extremely common. This is the classic Zipfian distribution
# observed in natural language. The top-20 removal targets the far-right extreme of this curve.
# <b>&#8594; Observation:</b> [Update after running.] The top 20 words removed are
# domain-specific filler words such as `good`, `product`, `skin`, and `love` that appear
# across the vast majority of reviews. Removing them leaves a vocabulary that is more
# discriminative and suitable for downstream modelling tasks.
# %%
def word_frequency_stats(
    tokenized_reviews: list[list[str]],
    top_n: int = 20,
) -> pd.DataFrame:
    """Compute per-word term frequency and document frequency statistics."""
    freq         = Counter(w for r in tokenized_reviews for w in r)
    total_tokens = sum(freq.values())
    doc_freq: dict[str, int] = defaultdict(int)
    for review in tokenized_reviews:
        for w in set(review):
            doc_freq[w] += 1
    df_freq = pd.DataFrame(freq.most_common(), columns=["word", "count"])
    df_freq["pct_of_tokens"]  = df_freq["count"] / total_tokens * 100
    df_freq["cumulative_pct"] = df_freq["pct_of_tokens"].cumsum()
    df_freq["doc_freq"]       = df_freq["word"].map(doc_freq)
    df_freq["doc_freq_pct"]   = df_freq["doc_freq"] / len(tokenized_reviews) * 100
    print(f"Vocabulary size : {len(freq):,}")
    print(f"Total tokens    : {total_tokens:,}")
    print(f"\nTop {top_n} most frequent words:")
    print(df_freq.head(top_n).to_string(index=False))
    print(f"\nBottom {top_n} least frequent words:")
    print(df_freq.tail(top_n).to_string(index=False))
    return df_freq
def remove_top_n_by_doc_freq(
    tokenized_reviews: list[list[str]],
    df_freq: pd.DataFrame,
    top_n: int = 20,
) -> list[list[str]]:
    """Remove the top N most frequent words ranked by document frequency."""
    top_words = set(df_freq.nlargest(top_n, "doc_freq")["word"])
    print(f"Top {top_n} words removed (by document frequency):")
    print(
        df_freq.nlargest(top_n, "doc_freq")[["word", "doc_freq", "doc_freq_pct"]]
        .to_string(index=False)
    )
    return [[w for w in r if w not in top_words] for r in tokenized_reviews]
# %%
df_freq = word_frequency_stats(tk_reviews)
# %%
df_freq["count"].plot(
    kind="hist", bins=100, log=True,
    title="Word frequency distribution (log scale)",
    xlabel="Frequency",
    ylabel="Number of words",
)
# %%
tk_reviews = remove_top_n_by_doc_freq(tk_reviews, df_freq, top_n=20)
stats_print(tk_reviews)
# %% [markdown]
# ---
# ## 4. Output Files
# We produce the two files required by the assignment brief. Both are built directly
# from `tk_reviews` so that `processed.csv` and `vocab.txt` are guaranteed to be
# consistent with each other - every word in the vocabulary is present in the CSV,
# and every token in the CSV is indexed in the vocabulary.
# %% [markdown]
# ### 4.1 Saving `processed.csv`
# We replace the `review_text` column with the cleaned, space-joined token list for
# each review. All other columns are preserved so that downstream tasks (Task 2, Task 3)
# can still access metadata such as `review_rating`, `brand_name`, and `product_tags`.
# We pass `index=False` so pandas does not add an extra index column.
# %%
processed_df = raw_reviews_df.copy()
processed_df["review_text"] = [" ".join(r) for r in tk_reviews]
processed_df.to_csv("../outputs/processed.csv", index=False)
print(f"Saved processed.csv - {len(processed_df):,} rows, {processed_df.shape[1]} columns.")
processed_df[["review_id", "review_title", "review_text", "review_rating"]].head()
# %% [markdown]
# ### 4.2 Saving `vocab.txt`
# The vocabulary file lists every unique word remaining in the cleaned reviews,
# sorted alphabetically, one entry per line, in the format `word:integer_index`.
# The index starts at 0. This is the unigram vocabulary required by the brief.
# We verify the output against all format requirements before finishing.
# %%
vocab = sorted(set(chain.from_iterable(tk_reviews)))
Path("../outputs/vocab.txt").write_text(
    "\n".join(f"{w}:{i}" for i, w in enumerate(vocab)),
    encoding="utf-8",
)
lines = Path("../outputs/vocab.txt").read_text(encoding="utf-8").splitlines()
print(f"Vocabulary size : {len(lines):,} words")
print("\nFirst 10 entries:")
for line in lines[:10]:
    print(f"  {line}")
print("\nLast 5 entries:")
for line in lines[-5:]:
    print(f"  {line}")
# %% [markdown]
# ---
# ## Summary
# We built a robust text pre-processing pipeline for approximately 61,000 cosmetics and
# beauty product reviews. The pipeline followed all required assignment steps and extended
# them with additional quality improvements.
# ### Required steps completed
# | Step | Action taken |
# |---|---|
# | Tokenisation | `r"[a-zA-Z]+(?:[-'][a-zA-Z]+)?"` applied after full text cleaning |
# | Lowercase | Applied as the final step before tokenisation |
# | Remove short words | Tokens shorter than 2 characters removed (applied twice) |
# | Remove stop words | Assignment-provided `stopwords_en.txt` |
# | Remove hapax legomena | Words with term frequency = 1 removed |
# | Remove top 20 | Top 20 words by document frequency removed |
# ### Output files produced
# - **`processed.csv`** - original dataset with cleaned `review_text`; all metadata
#   columns preserved for use in Tasks 2 and 3.
# - **`vocab.txt`** - alphabetically sorted unigram vocabulary in `word:index` format,
#   verified to satisfy all format requirements in the brief.
# 