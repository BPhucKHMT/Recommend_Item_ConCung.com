# 🎯 Recommend_Item_ConCung.com

![RecSys](https://img.shields.io/badge/RecSys-Personalized-blue?style=flat-square) ![Python](https://img.shields.io/badge/Python-3.8%2B-green?style=flat-square) ![Status](https://img.shields.io/badge/status-active-brightgreen?style=flat-square)

<img width="1912" height="878" alt="image" src="https://github.com/user-attachments/assets/aac3531e-d522-4fd0-879b-29a94190e421" />

A personalized **recommendation system** for ConCung.com customers. The system recommends products from transaction history, product metadata, and behavioral features.

Due to data privacy constraints, please contact the project owner directly to access the dataset.

---

## 📁 Project Structure

```
.
├── app.py
├── candidate_version3.py
├── candidate_version4.py
├── config.py
├── eval.py
├── features.py
├── final_submission.json
├── get_candidates.py
├── item_data.csv
├── main.py
├── params.json
├── reranking_lasso.py
├── reranking_logistic.py
├── reranking.py
├── test_step_1.py
├── test.py
├── data/
│   └── table/
│       ├── item_data.parquet
│       └── ...
├── notebook/
│   ├── check_date_item.ipynb
│   ├── check_groundtruth_availability.ipynb
│   ├── evaluate.ipynb
│   ├── test.ipynb
│   └── ...
└── ...
```

---

## 🚦 Pipeline and Strategy

### 1️⃣ Data Preprocessing

- Load transaction and product data from `.parquet` files.
- Normalize date fields and remove columns with insufficient information.
- Filter unavailable products listed in [`unavailable_items.txt`](unavailable_items.txt).

---

### 2️⃣ Candidate Generation

**Main strategies:**

- **ALS Matrix Factorization (Collaborative Filtering)**
  - Uses Alternating Least Squares (ALS) to learn latent factors from the user-item matrix.
  - Exploits purchase history to recommend products that match user behavior.
  - Supports GPU batch processing when available for better speed.

- **BPR Ranking (Bayesian Personalized Ranking)**
  - Learns product preference order for each user from historical interactions.
  - Optimized for implicit feedback recommendation tasks.

- **Item2Vec Embedding**
  - Uses Word2Vec to learn product embeddings from purchase sequences.
  - Finds products that often appear in similar shopping contexts.

- **Content-Based Filtering**
  - Uses product attributes such as brand, category, and description.
  - Computes similarity with cosine similarity, TF-IDF, and FAISS GPU when available.
  - Recommends content-similar products based on each user's past purchases.

- **Trending Items**
  - Recommends best-selling products in recent time windows.
  - Can filter by year or month to avoid outdated products.

- **Segment-Based Fallback (Cold Start)**
  - Handles users with limited purchase history by grouping users into segments.
  - Segments can use gender, region, membership level, or similar attributes.
  - Falls back to global trending products when segment-level data is insufficient.

- **Purchased and Unavailable Product Exclusion**
  - Excludes products already purchased by the user.
  - Excludes unavailable products listed in [`unavailable_items.txt`](unavailable_items.txt).

- **Diversity with Maximal Marginal Relevance (MMR)**
  - Improves recommendation diversity across categories and brands.
  - Reduces repeated or overly similar products in the final list.

**Reference code:**

- [`candidate_version4.py`](candidate_version4.py): `get_candidates`, `get_advanced_fallback`, `maximal_marginal_relevance`, and related candidate-generation utilities.

---

### 3️⃣ Feature Engineering

- [`features.py`](features.py) builds features for each customer-product pair:
  - Purchase-history similarity: co-occurrence, brand similarity, and category similarity.
  - Price, price trend, and product popularity.
  - Repurchase frequency, velocity, and repurchase rate.
  - Embedding-based features when available.
  - Contextual features such as date and seasonality.

---

### 4️⃣ Reranking

- [`reranking.py`](reranking.py), [`reranking_lasso.py`](reranking_lasso.py), and [`reranking_logistic.py`](reranking_logistic.py) rerank recommendation candidates using multiple feature groups.
- The reranking stage applies penalties for price mismatch, increases diversity, prioritizes newer products, and rewards products that match the user's preferred brands or categories.

---

### 5️⃣ Evaluation

- [`eval.py`](eval.py) evaluates recommendation quality using Precision@10.
- Evaluation excludes products already purchased in the user's history.
- Predictions are compared against the actual ground truth.

---

### 6️⃣ Output Generation

- Final recommendations are saved to [`final_submission.json`](final_submission.json) or [`result.json`](result.json).

---

## 🛠️ How to Run

1. **Install dependencies:**

   ```sh
   pip install -r requirements.txt
   ```

   If `requirements.txt` is unavailable, install the core packages manually: `polars`, `pandas`, `numpy`, `tqdm`, `scikit-learn`, `gensim`, `streamlit`, and related dependencies.

2. **Run the main pipeline:**

   ```sh
   python main.py
   ```

3. **Run the Streamlit demo app:**

   ```sh
   streamlit run app.py
   ```

4. **Tune parameters:**

   Update time windows, candidate counts, and other settings in [`params.json`](params.json).

---

## 📒 Supporting Notebooks and Analysis

| Notebook | Purpose |
|---|---|
| [notebook/check_date_item.ipynb](notebook/check_date_item.ipynb) | Checks and cleans product date fields, including time-related anomalies. |
| [notebook/check_groundtruth_availability.ipynb](notebook/check_groundtruth_availability.ipynb) | Analyzes ground-truth coverage and verifies user/item availability. |
| [notebook/EDA1.ipynb](notebook/EDA1.ipynb) | Performs high-level exploratory data analysis, including user, item, and transaction statistics. |
| [notebook/EDA2.ipynb](notebook/EDA2.ipynb) | Provides deeper analysis of user behavior, product behavior, and shopping trends. |
| [notebook/evaluate.ipynb](notebook/evaluate.ipynb) | Runs offline evaluation and compares candidate-generation and reranking strategies. |
| [notebook/recommendation-preprocess.ipynb](notebook/recommendation-preprocess.ipynb) | Preprocesses data, normalizes fields, and creates input features for the pipeline. |
| [notebook/test.ipynb](notebook/test.ipynb) | Provides quick checks for functions and modules in the recommendation pipeline. |

---

## 💡 Strategy Notes

- **Unavailable product filtering:** Prevents discontinued or unavailable products from being recommended.
- **Precision optimization for new purchases:** Scores only products that users have not purchased before.
- **Recommendation diversity:** Avoids returning only products from the same category or brand.
- **Performance optimization:** Uses Polars for large-scale data processing and batch processing during feature generation.

---

## 📂 Important Files

- [`main.py`](main.py): Main recommendation pipeline.
- [`config.py`](config.py): Configuration parameters and time-window query helpers.
- [`features.py`](features.py): Feature engineering for model input.
- [`candidate_version4.py`](candidate_version4.py): Product candidate generation.
- [`reranking.py`](reranking.py): Reranking functions that combine multiple feature signals.
- [`eval.py`](eval.py): Recommendation evaluation logic.

---

## 🤝 Result

- Achieved Precision@10 of **10.96%** on February 2025 transaction data.

---

<p align="center">
  <img src="https://img.icons8.com/color/96/000000/artificial-intelligence.png" width="80"/>
</p>
