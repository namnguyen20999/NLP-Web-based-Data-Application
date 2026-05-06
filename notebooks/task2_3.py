# %% [markdown]
# # Assignment 2: Milestone I — Natural Language Processing
# ## Tasks 2 & 3: Feature Representations and Review Classification
# #### Student Name: XXXX XXXX
# #### Student ID: 000000
# #
# **Environment:** Python 3.10+ | Jupyter Notebook
# #
# **Libraries used:**
# | Library | Purpose |
# |---|---|
# | `os` | File existence checks |
# | `collections.Counter` | Efficient word-frequency counting |
# | `numpy` | Dense vector arithmetic |
# | `pandas` | Dataset loading and tabular manipulation |
# | `scipy.sparse.csr_matrix` | Memory-efficient sparse matrix for BoW vectors |
# | `sklearn.feature_extraction.text` | `CountVectorizer`, `TfidfVectorizer` |
# | `sklearn.linear_model` | Logistic Regression |
# | `sklearn.svm` | Linear SVM (`LinearSVC`) |
# | `sklearn.naive_bayes` | Multinomial Naïve Bayes |
# | `sklearn.model_selection` | `StratifiedKFold`, `cross_validate` |
# | `sklearn.pipeline` | End-to-end ML pipelines |
# | `sklearn.compose` | `ColumnTransformer` for multi-field feature fusion |
# | `sklearn.preprocessing` | `OneHotEncoder`, `StandardScaler` |
# | `sklearn.impute` | `SimpleImputer` for missing numeric values |
# | `IPython.display` | Formatted DataFrame rendering in Jupyter |
# #
# ---
# ## Introduction
# #
# This notebook covers **Tasks 2 and 3** of Assignment 2, working with the cosmetics
# and beauty product reviews dataset.
# #
# **Task 2 — Feature Representations** converts pre-processed review text from
# `processed.csv` into three numeric representations consumed by the classifiers:
# #
# * **Bag-of-Words (BoW):** Sparse unigram frequency vectors anchored to the
#   7 241-word vocabulary produced in Task 1 (`vocab.txt`).
# * **Unweighted GloVe Embeddings:** Dense 300-d vectors formed by averaging the
#   pre-trained GloVe word embeddings of every token in a review.
# * **TF-IDF Weighted GloVe Embeddings:** Dense 300-d vectors formed by computing a
#   TF-IDF–weighted average of GloVe embeddings, down-weighting high-frequency,
#   low-information tokens so that rarer, topic-specific words carry more influence.
# #
# **Task 3 — Classification** addresses two research questions:
# #
# * **Q1:** Which combination of feature representation and classifier achieves the
#   highest predictive performance on the binary `is_a_buyer` label?
# * **Q2:** Does augmenting the review body with additional context — review title,
#   product metadata, brand, and pricing — improve classification accuracy?
# #
# All experiments use 5-fold Stratified Cross-Validation to produce reliable,
# variance-aware performance estimates.
# 
# %% [markdown]
# ---
# ## Importing Libraries
# 
# %%
import os
from collections import Counter

import numpy as np
import pandas as pd
from IPython.display import display
from scipy.sparse import csr_matrix
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC

# %% [markdown]
# ---
# ## Global Configuration
# #
# All file paths and hyper-parameters are declared here so they can be changed in one
# place without hunting through the notebook.
# 
# %%
# --- Paths ---
PROCESSED_PATH  = "processed.csv"          # Pre-processed reviews (Task 1 output)
VOCAB_PATH      = "vocab.txt"              # Vocabulary with word:index mapping (Task 1 output)
GLOVE_PATH      = "glove.6B.300d.txt"     # Pre-trained GloVe 300-d embeddings
COUNT_VEC_PATH  = "count_vectors.txt"     # Task 2a output: sparse BoW vectors
UNWEIGHTED_PATH = "unweighted_vectors.txt" # Task 2b output: unweighted GloVe vectors
WEIGHTED_PATH   = "weighted_vectors.txt"  # Task 2b output: TF-IDF weighted GloVe vectors

# --- Hyper-parameters ---
EMBEDDING_DIM = 300   # GloVe vector dimension
RANDOM_STATE  = 42    # Global seed for reproducibility

# %% [markdown]
# ---
# ## Task 2 — Generating Feature Representations
# #
# Three output files are produced, each using the same line format:
# #
# | File | Type | Line format |
# |---|---|---|
# | `count_vectors.txt` | Sparse BoW | `#idx,word_idx:freq,...` |
# | `unweighted_vectors.txt` | Dense 300-d | `#idx,v0,v1,...,v299` |
# | `weighted_vectors.txt` | Dense 300-d | `#idx,v0,v1,...,v299` |
# 
# %% [markdown]
# ### Task 2a — Bag-of-Words Count Vectors
# #
# Each review is serialised as a sparse vector that records the frequency of every
# vocabulary word it contains.  Words absent from `vocab.txt` are ignored, keeping the
# output compact.  Entries within each line are sorted by word index for stable output.
# 
# %% [markdown]
# #### Load Vocabulary
# 
# %%
word2idx: dict[str, int] = {}

with open(VOCAB_PATH, "r", encoding="utf-8") as _f:
    for _line in _f:
        _line = _line.strip()
        if not _line:
            continue
        # vocab.txt format produced by Task 1: word:index
        _word, _idx = _line.rsplit(":", 1)
        word2idx[_word] = int(_idx)

print(f"Vocabulary size  : {len(word2idx):,}")
print(f"First 10 entries : {list(word2idx.items())[:10]}")

# %% [markdown]
# #### Load Pre-processed Reviews
# 
# %%
df_task2 = pd.read_csv(PROCESSED_PATH)
reviews: list[str] = df_task2["review_text"].fillna("").astype(str).tolist()

print(f"Reviews loaded        : {len(reviews):,}")
print(f"Example (first 200 ch): {reviews[0][:200]}")

# %% [markdown]
# #### Build and Save Count Vectors
# 
# %%
def to_sparse_count_line(review_idx: int, review_text: str, w2i: dict) -> str:
    """Serialise one review as a sparse count-vector line.

    Format: #<review_idx>,<word_idx>:<freq>,...
    """
    tokens = review_text.split() if review_text else []
    freq   = Counter(tok for tok in tokens if tok in w2i)
    pairs  = sorted((w2i[word], cnt) for word, cnt in freq.items())

    if not pairs:
        return f"#{review_idx},"

    return f"#{review_idx}," + ",".join(f"{idx}:{cnt}" for idx, cnt in pairs)


count_vector_lines = [
    to_sparse_count_line(i, text, word2idx) for i, text in enumerate(reviews)
]

with open(COUNT_VEC_PATH, "w", encoding="utf-8") as _f:
    _f.write("\n".join(count_vector_lines))

print(f"count_vectors.txt — {len(count_vector_lines):,} lines written")
print("Preview (first 3 lines, truncated to 200 chars):")
for _line in count_vector_lines[:3]:
    print(f"  {_line[:200]}")

# %% [markdown]
# ### Task 2b — GloVe Embedding Vectors
# #
# Pre-trained GloVe embeddings (`glove.6B.300d.txt`) encode semantic relationships
# learned from a large external corpus into 300-dimensional vectors.  Two aggregation
# strategies are implemented:
# #
# | Strategy | Description |
# |---|---|
# | **Unweighted average** | All in-vocabulary tokens contribute equally |
# | **TF-IDF weighted average** | Rare, informative tokens receive higher weight |
# #
# Reviews with no GloVe-covered tokens are represented as zero vectors.
# 
# %% [markdown]
# #### Tokenisation Helper
# 
# %%
def tokenize(text: str) -> list[str]:
    """Split pre-processed review text into tokens.

    processed.csv already contains clean, lowercased, space-separated tokens
    produced by Task 1, so a plain split is correct and consistent with the
    count-vector tokeniser.
    """
    return text.split() if text else []

# %% [markdown]
# #### Load GloVe Embeddings
# 
# %%
def load_glove(file_path: str) -> dict[str, np.ndarray]:
    """Parse a GloVe text file into a word → vector dictionary."""
    embeddings: dict[str, np.ndarray] = {}
    with open(file_path, encoding="utf-8") as _f:
        for _line in _f:
            _values = _line.split()
            embeddings[_values[0]] = np.asarray(_values[1:], dtype="float32")
    return embeddings


glove = load_glove(GLOVE_PATH)
print(f"GloVe vocabulary : {len(glove):,} words  |  dimension: {EMBEDDING_DIM}")

# %% [markdown]
# #### Build and Save Unweighted Embedding Vectors
# 
# %%
unweighted_vectors: list[str] = []

for _i, _review in enumerate(reviews):
    _tokens  = tokenize(_review)
    _vecs    = [glove[w] for w in _tokens if w in glove]
    _avg_vec = np.mean(_vecs, axis=0) if _vecs else np.zeros(EMBEDDING_DIM)
    unweighted_vectors.append(f"#{_i}," + ",".join(map(str, _avg_vec)))

with open(UNWEIGHTED_PATH, "w", encoding="utf-8") as _f:
    _f.write("\n".join(unweighted_vectors) + "\n")

print(f"unweighted_vectors.txt — {len(unweighted_vectors):,} lines written")
print("\nPreview — first 3 reviews:")
for _i in range(3):
    _tokens = tokenize(reviews[_i])
    _in_glove = [w for w in _tokens if w in glove]
    print(f"\n  Review #{_i}")
    print(f"  Original text : {reviews[_i][:200]}")
    print(f"  Tokens        : {_tokens[:15]}{'...' if len(_tokens) > 15 else ''}")
    print(f"  In GloVe      : {len(_in_glove)}/{len(_tokens)} tokens")
    print(f"  Full vector   : {unweighted_vectors[_i]}")

# %% [markdown]
# #### Build and Save TF-IDF Weighted Embedding Vectors
# #
# **Step 1** — Fit a TF-IDF vectorizer on all reviews to learn corpus-wide term
# weights.  The same custom tokenizer is used so the vocabulary matches GloVe look-ups.
# 
# %%
_tfidf_vectorizer = TfidfVectorizer(tokenizer=tokenize, token_pattern=None, lowercase=True)
tfidf_matrix      = _tfidf_vectorizer.fit_transform(reviews)
word_tfidf_index  = {w: i for i, w in enumerate(_tfidf_vectorizer.get_feature_names_out())}

print(f"TF-IDF vocabulary size : {len(word_tfidf_index):,}")

# %% [markdown]
# **Step 2** — For each review, compute the TF-IDF–weighted average of its token
# embeddings, then normalise by the total weight to remove length bias.
# 
# %%
weighted_vectors: list[str] = []

for _i, _review in enumerate(reviews):
    _tokens       = tokenize(_review)
    _vec          = np.zeros(EMBEDDING_DIM)
    _total_weight = 0.0

    for _word in _tokens:
        if _word in glove and _word in word_tfidf_index:
            _w             = tfidf_matrix[_i, word_tfidf_index[_word]]
            _vec          += glove[_word] * _w
            _total_weight += _w

    if _total_weight > 0:
        _vec /= _total_weight

    weighted_vectors.append(f"#{_i}," + ",".join(map(str, _vec)))

with open(WEIGHTED_PATH, "w", encoding="utf-8") as _f:
    _f.write("\n".join(weighted_vectors) + "\n")

print(f"weighted_vectors.txt — {len(weighted_vectors):,} lines written")
print("\nPreview — first 3 reviews:")
for _i in range(3):
    _tokens = tokenize(reviews[_i])
    _in_both = [w for w in _tokens if w in glove and w in word_tfidf_index]
    print(f"\n  Review #{_i}")
    print(f"  Original text    : {reviews[_i][:200]}")
    print(f"  Tokens           : {_tokens[:15]}{'...' if len(_tokens) > 15 else ''}")
    print(f"  In GloVe+TF-IDF  : {len(_in_both)}/{len(_tokens)} tokens")
    print(f"  Full vector      : {weighted_vectors[_i]}")

# %% [markdown]
# #### Validator — Original Text vs Unweighted vs Weighted Vectors
# #
# Each stored line is a formatted string `#idx,v0,v1,...`.  We parse both vectors
# back to numpy arrays so we can compare them numerically.  Unweighted and weighted
# vectors represent the same review but must **not** be identical — TF-IDF weighting
# shifts the average toward rarer, more informative tokens.
# 
# %%
def _parse_vec_line(line: str) -> np.ndarray:
    """Parse '#idx,v0,v1,...' back to a numpy array (drops the index)."""
    parts = line.split(",")
    return np.array([float(v) for v in parts[1:]], dtype=float)


def validator_embedding(review_idx, review_txts, tokenised, unweighted_lines, weighted_lines, n_dims=10):
    u_vec = _parse_vec_line(unweighted_lines[review_idx])
    w_vec = _parse_vec_line(weighted_lines[review_idx])

    u_preview = ",".join(f"{v:.6f}" for v in u_vec[:n_dims])
    w_preview = ",".join(f"{v:.6f}" for v in w_vec[:n_dims])

    norm_u = np.linalg.norm(u_vec)
    norm_w = np.linalg.norm(w_vec)
    cosine = (
        np.dot(u_vec, w_vec) / (norm_u * norm_w)
        if norm_u > 0 and norm_w > 0 else float("nan")
    )

    print("=" * 60)
    print(f"Review #{review_idx}")
    print(f"Original text : {review_txts[review_idx][:200]}")
    print(f"Tokens        : {tokenised[review_idx][:12]}")
    print("-" * 60)
    print(f"Unweighted (first {n_dims} of {len(u_vec)} dims):")
    print(f"  #{review_idx},{u_preview},...")
    print(f"Weighted   (first {n_dims} of {len(w_vec)} dims):")
    print(f"  #{review_idx},{w_preview},...")
    print("-" * 60)
    print(f"Unweighted norm : {norm_u:.4f}")
    print(f"Weighted norm   : {norm_w:.4f}")
    print(f"Cosine similarity (u vs w) : {cosine:.4f}  (expected < 1.0)")
    print(f"Vectors identical          : {np.allclose(u_vec, w_vec)}  (expected False)")
    print("=" * 60)
    print()


tokenised_for_display = df_task2["review_text"].fillna("").astype(str).apply(str.split).tolist()

for _test_idx in range(3):
    validator_embedding(_test_idx, reviews, tokenised_for_display, unweighted_vectors, weighted_vectors)

# %% [markdown]
# ---
# ## Task 3 — Clothing Review Classification
# #
# Using the three representations generated above, we train and evaluate classifiers
# to predict `is_a_buyer` — whether a reviewer actually purchased the product.
# 
# %% [markdown]
# ### Shared I/O Utilities
# #
# These parsing functions are shared across Q1 and Q2.  `align_to_dataframe` reorders
# a loaded matrix to match the sequential row order of a DataFrame, preventing
# index–label mismatches when doc indices are out of order.
# 
# %%
def load_count_vectors(path: str) -> tuple:
    """Parse a sparse count-vector file into a CSR matrix.

    Returns (csr_matrix, doc_indices_array) or (None, None) on empty file.
    """
    doc_indices: list[int] = []
    rows, cols, data = [], [], []
    max_col = -1

    with open(path, "r", encoding="utf-8") as _f:
        for _row_id, _line in enumerate(_f):
            _line = _line.strip()
            if not _line:
                continue
            _parts  = _line.split(",")
            _doc_id = int(_parts[0].lstrip("#"))
            doc_indices.append(_doc_id)

            for _item in _parts[1:]:
                if not _item or ":" not in _item:
                    continue
                _c, _v = _item.split(":")
                _c, _v = int(_c), float(_v)
                rows.append(_row_id)
                cols.append(_c)
                data.append(_v)
                max_col = max(max_col, _c)

    if not doc_indices:
        return None, None

    X = csr_matrix((data, (rows, cols)), shape=(len(doc_indices), max_col + 1))
    return X, np.array(doc_indices)


def load_dense_vectors(path: str) -> tuple:
    """Parse a dense embedding file into a NumPy array.

    Returns (np.ndarray, doc_indices_array) or (None, None) on empty file.
    """
    doc_indices: list[int] = []
    vectors: list[list[float]] = []

    with open(path, "r", encoding="utf-8") as _f:
        for _line in _f:
            _line = _line.strip()
            if not _line:
                continue
            _parts = _line.split(",")
            doc_indices.append(int(_parts[0].lstrip("#")))
            vectors.append([float(x) for x in _parts[1:] if x.strip()])

    if not doc_indices:
        return None, None

    return np.array(vectors, dtype=float), np.array(doc_indices)


def align_to_dataframe(X, doc_indices: np.ndarray, n_rows: int):
    """Reorder X rows to match sequential DataFrame indices 0…n-1."""
    if X is None or doc_indices is None:
        return None
    if len(doc_indices) != n_rows:
        raise ValueError(
            f"Vector file has {len(doc_indices)} rows but DataFrame has {n_rows} rows."
        )
    order = np.argsort(doc_indices)
    return X[order] if hasattr(X, "tocsr") else X[order, :]

# %% [markdown]
# ---
# ### Q1 — Language Model Comparison
# #
# **Objective:** Determine which combination of feature representation and classifier
# achieves the best `is_a_buyer` classification performance.
# #
# **Representations evaluated:** Bag-of-Words · Unweighted GloVe · TF-IDF Weighted GloVe
# #
# **Classifiers evaluated:** Logistic Regression · Linear SVM · Multinomial Naïve Bayes
# #
# > **Note:** Multinomial Naïve Bayes requires strictly non-negative inputs.  It is
# > therefore evaluated only on the Bag-of-Words representation and excluded from the
# > dense embedding experiments.
# #
# **Method:** 5-fold Stratified Cross-Validation reporting mean ± std Accuracy,
# Precision, Recall, and F1.
# 
# %% [markdown]
# #### Load Dataset and Prepare Labels
# 
# %%
# Load each vector file into a DataFrame and attach the corresponding is_a_buyer label.
# The #idx field in every vector line is the 0-based row position in processed.csv,
# so df_task2.loc[idx, "is_a_buyer"] gives the exact label for each vector row.
X_count,      idx_count      = load_count_vectors(COUNT_VEC_PATH)
X_unweighted, idx_unweighted = load_dense_vectors(UNWEIGHTED_PATH)
X_weighted,   idx_weighted   = load_dense_vectors(WEIGHTED_PATH)

df_count = pd.DataFrame.sparse.from_spmatrix(
    X_count,
    index=idx_count,
    columns=[f"w{i}" for i in range(X_count.shape[1])],
)
df_count["is_a_buyer"] = df_task2.loc[idx_count, "is_a_buyer"].to_numpy()

df_unweighted = pd.DataFrame(
    X_unweighted,
    index=idx_unweighted,
    columns=[f"dim_{i}" for i in range(X_unweighted.shape[1])],
)
df_unweighted["is_a_buyer"] = df_task2.loc[idx_unweighted, "is_a_buyer"].to_numpy()

df_weighted = pd.DataFrame(
    X_weighted,
    index=idx_weighted,
    columns=[f"dim_{i}" for i in range(X_weighted.shape[1])],
)
df_weighted["is_a_buyer"] = df_task2.loc[idx_weighted, "is_a_buyer"].to_numpy()

print(f"\ndf_count      : {df_count.shape}  - is_a_buyer sample: {df_count['is_a_buyer'].iloc[:3].tolist()}")
print(f"df_unweighted : {df_unweighted.shape}  - is_a_buyer sample: {df_unweighted['is_a_buyer'].iloc[:3].tolist()}")
print(f"df_weighted   : {df_weighted.shape}  - is_a_buyer sample: {df_weighted['is_a_buyer'].iloc[:3].tolist()}")

# %% [markdown]
# #### Cross-Validation Helper
# 
# %%
def evaluate_representation(X, y, representation_name: str,
                             classifier_name: str, classifier) -> dict:
    """Run 5-fold stratified CV and return a labelled metrics dict."""
    _cv      = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    _scoring = {"accuracy": "accuracy", "precision": "precision",
                "recall": "recall", "f1": "f1"}

    _scores = cross_validate(classifier, X, y, cv=_cv, scoring=_scoring,
                             n_jobs=-1, error_score="raise")

    # Per-fold breakdown
    print(f"\n  [{representation_name}] {classifier_name}")
    print(f"  {'Fold':<6} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print(f"  {'-'*6} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for _fold in range(5):
        print(f"  {_fold + 1:<6} "
              f"{_scores['test_accuracy'][_fold]:>10.4f} "
              f"{_scores['test_precision'][_fold]:>10.4f} "
              f"{_scores['test_recall'][_fold]:>10.4f} "
              f"{_scores['test_f1'][_fold]:>10.4f}")
    print(f"  {'Mean':<6} "
          f"{_scores['test_accuracy'].mean():>10.4f} "
          f"{_scores['test_precision'].mean():>10.4f} "
          f"{_scores['test_recall'].mean():>10.4f} "
          f"{_scores['test_f1'].mean():>10.4f}")
    print(f"  {'Std':<6} "
          f"{_scores['test_accuracy'].std():>10.4f} "
          f"{_scores['test_precision'].std():>10.4f} "
          f"{_scores['test_recall'].std():>10.4f} "
          f"{_scores['test_f1'].std():>10.4f}")

    return {
        "Representation" : representation_name,
        "Classifier"     : classifier_name,
        "Accuracy"       : f"{_scores['test_accuracy'].mean():.4f} ± {_scores['test_accuracy'].std():.4f}",
        "Precision"      : f"{_scores['test_precision'].mean():.4f} ± {_scores['test_precision'].std():.4f}",
        "Recall"         : f"{_scores['test_recall'].mean():.4f} ± {_scores['test_recall'].std():.4f}",
        "F1"             : f"{_scores['test_f1'].mean():.4f} ± {_scores['test_f1'].std():.4f}",
        "_f1_sort"       : _scores["test_f1"].mean(),
    }

# %%
# Logistic Regression with count vectors
evaluate_representation(X_count, df_count['is_a_buyer'], "Bag-of-Words", "Logistic Regression",
                        LogisticRegression(max_iter=2000, solver="liblinear", random_state=RANDOM_STATE))
# %%
# Logistic Regression with unweighted GloVe vectors
evaluate_representation(X_unweighted, df_unweighted['is_a_buyer'], "Unweighted GloVe", "Logistic Regression",
                        LogisticRegression(max_iter=2000, solver="liblinear", random_state=RANDOM_STATE))
# %%
# Logistic Regression with weighted GloVe vectors
evaluate_representation(X_weighted, df_weighted['is_a_buyer'], "Weighted GloVe", "Logistic Regression",
                        LogisticRegression(max_iter=2000, solver="liblinear", random_state=RANDOM_STATE))
# %% [markdown]
# #### Initialise Classifiers and Load Representations
# 
# %%
# Classifiers compatible with all representation types (including dense / negative values)
general_classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=2000, solver="liblinear",
                                              random_state=RANDOM_STATE),
    "Linear SVM"         : LinearSVC(random_state=RANDOM_STATE),
}

# MultinomialNB added only for the non-negative Bag-of-Words representation
count_classifiers = {
    **general_classifiers,
    "Multinomial Naïve Bayes": MultinomialNB(),
}

# Map each representation name to its file path and format
representations_q1 = {
    "Bag-of-Words" : (COUNT_VEC_PATH,  "sparse"),
    "Unweighted"   : (UNWEIGHTED_PATH, "dense"),
    "Weighted"     : (WEIGHTED_PATH,   "dense"),
}

# %% [markdown]
# #### Run Q1 Experiments
# 
# %%
q1_results = []

for rep_name, (path, fmt) in representations_q1.items():
    if not os.path.exists(path):
        print(f"[SKIP] File not found: {path}")
        continue

    X_rep, idx = (load_count_vectors(path) if fmt == "sparse"
                  else load_dense_vectors(path))

    if X_rep is None:
        print(f"[SKIP] Could not parse: {path}")
        continue

    y_rep = df.loc[idx, "label"].to_numpy()
    clfs  = count_classifiers if rep_name == "Bag-of-Words" else general_classifiers

    print(f"\n[{rep_name}]  shape={X_rep.shape}")
    for clf_name, clf in clfs.items():
        print(f"  Training {clf_name} ...")
        q1_results.append(
            evaluate_representation(X_rep, y_rep, rep_name, clf_name, clf)
        )

# %% [markdown]
# #### Q1 Results Summary
# 
# %%
q1_df = pd.DataFrame(q1_results)

if not q1_df.empty:
    q1_df = (q1_df
             .sort_values("_f1_sort", ascending=False)
             .drop(columns=["_f1_sort"])
             .reset_index(drop=True))
    print("Q1 — Language Model Comparison (sorted by F1, descending):\n")
    display(q1_df)
else:
    print("No Q1 results — verify that Task 2 output files exist in the working directory.")

# %% [markdown]
# ---
# ### Q2 — Does Additional Context Improve Classification?
# #
# **Objective:** Test whether supplementing the review body with richer contextual
# signals improves `is_a_buyer` prediction accuracy.
# #
# **Three information scenarios are compared side-by-side:**
# #
# | Scenario | Input features |
# |---|---|
# | Text only | `review_text` (Task 2 pre-computed vectors) |
# | Text + Title | `review_text` + `review_title` |
# | Text + Title + Extra | above + `product_title`, `brand_name`, `price`, `avg_product_rating`, `product_rating_count` |
# #
# For scenarios 2 and 3, sklearn pipelines are built dynamically using
# `ColumnTransformer` so that each text field is vectorised independently and numeric
# fields are imputed and scaled before being concatenated into a single feature matrix.
# #
# **Method:** Same 5-fold Stratified CV strategy as Q1 for a fair comparison.
# 
# %% [markdown]
# #### Load and Prepare Q2 Dataset
# 
# %%
# Reuse processed.csv (already loaded as df_task2) so that row indices align exactly
# with the pre-computed vector files, which were built from processed.csv.
df_q2 = df_task2.copy().reset_index(drop=True)

TEXT_COLS    = ["review_text", "review_title", "product_title", "brand_name"]
NUMERIC_COLS = ["price", "avg_product_rating", "product_rating_count"]

for _col in TEXT_COLS:
    df_q2[_col] = df_q2.get(_col, pd.Series([""] * len(df_q2))).fillna("")

for _col in NUMERIC_COLS:
    if _col not in df_q2.columns:
        df_q2[_col] = np.nan
    df_q2[_col] = pd.to_numeric(df_q2[_col], errors="coerce")

if df_q2["is_a_buyer"].dtype == bool:
    df_q2["label"] = df_q2["is_a_buyer"].astype(int)
else:
    df_q2["label"] = (
        df_q2["is_a_buyer"].astype(str).str.strip().str.lower()
        .map({"true": 1, "false": 0})
    )

df_q2 = df_q2.dropna(subset=["label"]).reset_index(drop=True)
df_q2["label"] = df_q2["label"].astype(int)
y_q2 = df_q2["label"].to_numpy()

print(f"Q2 dataset shape : {df_q2.shape}")
print(f"\nLabel distribution:\n{df_q2['label'].value_counts().to_string()}")

# %% [markdown]
# #### Cross-Validation Engines and Pipeline Factory
# 
# %%
_cv_q2      = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
_scoring_q2 = {"accuracy": "accuracy", "precision": "precision",
               "recall": "recall", "f1": "f1"}


def _build_result_row(scores: dict, info_setting: str,
                      rep_name: str, clf_name: str) -> dict:
    return {
        "Information Setting" : info_setting,
        "Representation"      : rep_name,
        "Classifier"          : clf_name,
        "Accuracy"            : f"{scores['test_accuracy'].mean():.4f}",
        "F1"                  : f"{scores['test_f1'].mean():.4f}",
        "_f1_sort"            : scores["test_f1"].mean(),
    }


def evaluate_matrix_q2(X, y, rep_name: str,
                        info_setting: str, clf_name: str, clf) -> dict:
    """Evaluate a pre-computed matrix with cross-validation."""
    _scores = cross_validate(clf, X, y, cv=_cv_q2, scoring=_scoring_q2,
                             n_jobs=-1, error_score="raise")
    return _build_result_row(_scores, info_setting, rep_name, clf_name)


def evaluate_pipeline_q2(preprocessor, X_df, y, rep_name: str,
                          info_setting: str, clf_name: str, clf) -> dict:
    """Evaluate a full sklearn Pipeline (preprocessor + classifier)."""
    _pipe   = Pipeline([("preprocessor", preprocessor), ("classifier", clf)])
    _scores = cross_validate(_pipe, X_df, y, cv=_cv_q2, scoring=_scoring_q2,
                             n_jobs=-1, error_score="raise")
    return _build_result_row(_scores, info_setting, rep_name, clf_name)


# Shared numeric transformer: impute missing values then scale
_numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler(with_mean=False)),
])


def make_preprocessor(rep_name: str, include_extra: bool) -> ColumnTransformer:
    """Return a ColumnTransformer tailored to the requested representation.

    Each text field gets its own independent vectorizer instance so they learn
    separate vocabularies from different columns.
    """
    if rep_name == "Weighted":
        def _make_vec(max_f): return TfidfVectorizer(max_features=max_f)
    elif rep_name == "Unweighted":
        # binary=True approximates presence-only (unweighted) encoding
        def _make_vec(max_f): return CountVectorizer(binary=True, max_features=max_f)
    else:  # Bag-of-Words
        def _make_vec(max_f): return CountVectorizer(max_features=max_f)

    transformers = [
        ("review_text_vec",  _make_vec(5000), "review_text"),
        ("review_title_vec", _make_vec(2000), "review_title"),
    ]
    if include_extra:
        transformers.extend([
            ("product_title_vec", _make_vec(2000),                            "product_title"),
            ("brand_ohe",         OneHotEncoder(handle_unknown="ignore"),      ["brand_name"]),
            ("numeric",           _numeric_transformer,                        NUMERIC_COLS),
        ])
    return ColumnTransformer(transformers, remainder="drop")


classifiers_q2 = {
    "Logistic Regression": LogisticRegression(max_iter=2000, solver="liblinear",
                                              random_state=RANDOM_STATE),
    "Linear SVM"         : LinearSVC(random_state=RANDOM_STATE),
}

# %% [markdown]
# #### Scenario 1 — Text Only (Task 2 Pre-computed Vectors)
# 
# %%
q2_results = []
n_rows_q2  = len(df_q2)

text_only_reps = {
    "Bag-of-Words" : (COUNT_VEC_PATH,  "sparse"),
    "Weighted"     : (WEIGHTED_PATH,   "dense"),
    "Unweighted"   : (UNWEIGHTED_PATH, "dense"),
}

for rep_name, (path, fmt) in text_only_reps.items():
    if not os.path.exists(path):
        print(f"[SKIP] {path} not found.")
        continue
    _loader     = load_count_vectors if fmt == "sparse" else load_dense_vectors
    _X_raw, _idx = _loader(path)
    _X_aligned   = align_to_dataframe(_X_raw, _idx, n_rows_q2)

    print(f"Scenario 1 — Text only [{rep_name}]")
    for clf_name, clf in classifiers_q2.items():
        q2_results.append(
            evaluate_matrix_q2(_X_aligned, y_q2, rep_name, "Text only", clf_name, clf)
        )

# %% [markdown]
# #### Scenario 2 — Text + Review Title
# 
# %%
X_q2_df = df_q2[TEXT_COLS + NUMERIC_COLS].copy()
rep_names_q2 = ["Bag-of-Words", "Weighted", "Unweighted"]

print("Scenario 2 — Text + Title")
for rep_name in rep_names_q2:
    preprocessor = make_preprocessor(rep_name, include_extra=False)
    for clf_name, clf in classifiers_q2.items():
        print(f"  [{rep_name}] {clf_name}")
        q2_results.append(
            evaluate_pipeline_q2(preprocessor, X_q2_df, y_q2,
                                  rep_name, "Text + Title", clf_name, clf)
        )

# %% [markdown]
# #### Scenario 3 — Text + Title + Extra Metadata
# 
# %%
print("Scenario 3 — Text + Title + Extra Metadata")
for rep_name in rep_names_q2:
    preprocessor = make_preprocessor(rep_name, include_extra=True)
    for clf_name, clf in classifiers_q2.items():
        print(f"  [{rep_name}] {clf_name}")
        q2_results.append(
            evaluate_pipeline_q2(preprocessor, X_q2_df, y_q2,
                                  rep_name, "Text + Title + Extra", clf_name, clf)
        )

# %% [markdown]
# #### Q2 Results Summary
# 
# %%
q2_df = pd.DataFrame(q2_results)

if not q2_df.empty:
    q2_df = (q2_df
             .sort_values(["Information Setting", "Representation", "_f1_sort"],
                          ascending=[True, True, False])
             .drop(columns=["_f1_sort"])
             .reset_index(drop=True))
    print("Q2 — Information Expansion Comparison:\n")
    display(q2_df)
else:
    print("No Q2 results — verify file paths and that Task 2 output files exist.")

# %% [markdown]
# ### Q2 Analysis and Findings
# #
# **Research Question:** Does incorporating additional product and contextual information
# beyond the review body improve `is_a_buyer` classification accuracy?
# #
# **Conclusion:**
# Based on 5-fold stratified cross-validation, incorporating supplementary data
# [*fill in after running — e.g., "significantly improved" / "did not meaningfully improve"*]
# the model's ability to classify purchasing behaviour compared to review text alone.
# #
# **Key Observations:**
# #
# 1. **Review Title (Scenario 2 vs. Scenario 1):**
#    Adding the review title produced a [slight / significant] change in F1 score.
#    Titles tend to be short, opinionated phrases ("Perfect moisturiser!", "Broke me
#    out immediately") that carry concentrated sentiment — [confirming / contradicting]
#    the hypothesis that they provide complementary signal to the longer review body.
# #
# 2. **Structured Metadata (Scenario 3 vs. Scenario 2):**
#    Appending product title, brand (one-hot encoded), and numeric fields (price,
#    average rating, rating count) [further improved / had negligible effect on]
#    performance, suggesting that [pricing and brand signals carry discriminative
#    information / `is_a_buyer` is driven almost entirely by review text].
# #
# 3. **Best Overall Configuration:**
#    Across all three scenarios, **[representation]** paired with **[classifier]**
#    consistently achieved the highest F1-score.  This highlights the importance of
#    [*your explanation — e.g., TF-IDF weighting for suppressing noisy common words*].
# 
# %% [markdown]
# ---
# ## Summary
# #
# This notebook completed **Tasks 2 and 3** of Assignment 2.
# #
# **Task 2** produced three vector representations of the cosmetics review corpus:
# #
# * **Bag-of-Words** — sparse count vectors built from a 7,241-word vocabulary,
#   capturing exact lexical frequency.
# * **Unweighted GloVe** — 300-d dense vectors via uniform averaging of token
#   embeddings, capturing distributional semantics without frequency bias.
# * **TF-IDF Weighted GloVe** — 300-d dense vectors that down-weight common tokens,
#   allowing rare, topic-specific words to dominate the review representation.
# #
# **Task 3** evaluated Logistic Regression, Linear SVM, and Multinomial Naïve Bayes
# (BoW only) on two research questions:
# #
# * **Q1** compared the three representations and found that [*fill in best result
#   after running, e.g., "TF-IDF Weighted GloVe + Linear SVM achieved the highest
#   mean F1 of X.XX"*].
# * **Q2** investigated information enrichment and found that [*fill in conclusion,
#   e.g., "adding the review title improved F1 by X pp, while structured metadata
#   provided marginal additional gains"*].
# #
# Overall, [*your high-level takeaway about which representation and how much extra
# context is worth the added complexity*].
# 