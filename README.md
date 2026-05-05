# Cosmetics/Beauty Review Analysis & Classification

## 📌 Overview

This project focuses on analyzing cosmetics/beauty product reviews and building machine learning models to classify purchasing behaviour based on review data.

The project is divided into two main tasks:

* **Task 2**: Generate feature representations from review text
* **Task 3**: Build and evaluate classification models using those features

---

## 📂 Project Structure

```
NLP-Web-based-Data-Application/
│── notebooks/
│   └── task2_3.ipynb        # Main notebook for Task 2 & 3
│
│── outputs/
│   ├── count_vectors.txt
│   ├── unweighted_vectors.txt
│   └── weighted_vectors.txt
│
│── vocab.txt                # Vocabulary from Task 1
│── README.md
```

---

## ⚙️ Requirements

Install the required Python libraries before running:

```bash
pip install numpy pandas scikit-learn gensim
```

---

## 📊 Task 2: Feature Representation

We generated three types of feature representations from review text:

1. **Bag-of-Words (Count Vectors)**

   * Based on the provided `vocab.txt`
   * Stored in `count_vectors.txt`

2. **Unweighted Embedding Representation**

   * Average of word embeddings (GloVe)
   * Stored in `unweighted_vectors.txt`

3. **TF-IDF Weighted Embedding Representation**

   * Weighted average using TF-IDF scores
   * Stored in `weighted_vectors.txt`

---

## 🤖 Task 3: Classification Models

We trained and compared multiple models:

* Logistic Regression (baseline)
* Linear SVM
* Multinomial Naive Bayes

### Evaluation Method

* **5-fold cross-validation**
* Metrics:

  * Accuracy
  * Precision
  * Recall
  * F1-score

---

## 🔍 Experiments

### Q1: Language Model Comparison

We compared:

* Bag-of-Words
* Unweighted embeddings
* Weighted embeddings

👉 Goal: Identify which representation performs best.

---

### Q2: Effect of Additional Information

We evaluated models using:

1. Review text only
2. Review text + title
3. Review text + additional features (e.g., price, rating, brand)

👉 Goal: Determine if extra features improve performance.

---

## 📁 Important Note on Embeddings

The GloVe embedding file is **NOT included** in this repository due to size limitations.

### To run the project:

1. Download GloVe embeddings from:
   https://nlp.stanford.edu/projects/glove/

2. Use:

```
glove.6B.300d.txt
```

3. Place it in:

```
notebooks/
```

---

## ▶️ How to Run

1. Open the notebook:

```
notebooks/task2_3.ipynb
```

2. Run all cells sequentially:

* Feature generation (Task 2)
* Model training & evaluation (Task 3)

---

## 📈 Results Summary

| Model               | Representation  | Accuracy |
| ------------------- | --------------- | -------- |
| Logistic Regression | TF-IDF Weighted | XX%      |
| Linear SVM          | TF-IDF Weighted | XX%      |
| Naive Bayes         | Bag-of-Words    | XX%      |

*(Replace XX with your actual results)*

---

## 🧠 Key Insights

* Weighted embeddings generally outperform unweighted ones
* SVM and Logistic Regression perform better than Naive Bayes
* Adding additional features improves classification accuracy

---

## 👤 Author



---

## 📌 Notes

* Large files (e.g., embeddings) are excluded using `.gitignore`
* Ensure correct file paths when running the notebook
