<div align="center">

# 🔍 Sentiment Analysis & Recommender System
### Amazon Industrial & Scientific Product Reviews

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/)
[![NLTK](https://img.shields.io/badge/NLTK-NLP-4DB6AC?style=for-the-badge)](https://www.nltk.org/)

**Team #4 · COMP 262 · Centennial College · Semester 4**

*A two-phase NLP pipeline — from lexicon-based sentiment analysis to ML models, LLM summarisation, and a sentiment-enhanced recommender system.*

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Dataset](#-dataset)
- [Phase 1 — Lexicon-Based Sentiment Analysis](#-phase-1--lexicon-based-sentiment-analysis)
- [Phase 2 — ML Models, Recommender & LLM](#-phase-2--ml-models-recommender--llm)
- [Results at a Glance](#-results-at-a-glance)
- [Getting Started](#-getting-started)
- [Project Structure](#-project-structure)
- [References](#-references)

---

## 🌐 Overview

This project builds an end-to-end NLP pipeline on **77,071 Amazon product reviews** from the *Industrial and Scientific* category. It progresses through two phases:

| Phase | Focus | Key Techniques |
|-------|-------|---------------|
| **Phase 1** | Lexicon-based sentiment analysis | VADER, TextBlob, stratified sampling |
| **Phase 2** | ML models, recommender & LLM tasks | TF-IDF + LR/SVM, collaborative filtering, FLAN-T5, BART |

The core question being answered throughout:
> *Can we accurately predict the sentiment in technical product reviews — and use that sentiment signal to build better product recommendations?*

---

## 🏗️ Architecture

```
Raw Amazon Reviews (JSON)
        │
        ▼
┌───────────────────────────────────────────────────────────────────┐
│                         PHASE 1                                   │
│                                                                   │
│  ┌──────────────┐   ┌─────────────────┐   ┌──────────────────┐   │
│  │    Data      │──▶│  Pre-processing  │──▶│ Lexicon Analysis │   │
│  │ Exploration  │   │  (clean, label,  │   │ VADER vs TextBlob│   │
│  │ (77K reviews)│   │  sample 1,000)   │   │                  │   │
│  └──────────────┘   └─────────────────┘   └──────────────────┘   │
│                                                    │              │
│                               sentiment_analysis_results.csv ◀───┘
└───────────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────────┐
│                         PHASE 2                                   │
│                                                                   │
│  ┌────────────┐   ┌─────────────────┐   ┌──────────────────────┐ │
│  │ ML Models  │   │  Recommender    │   │   LLM Tasks          │ │
│  │ TF-IDF +   │   │  Sentiment-     │   │   Review Summariser  │ │
│  │ LR / SVM   │   │  Enhanced CF    │   │   CSR Response Gen   │ │
│  └────────────┘   └─────────────────┘   └──────────────────────┘ │
└───────────────────────────────────────────────────────────────────┘
```

---

## 📦 Dataset

| Attribute | Value |
|-----------|-------|
| **Source** | Amazon Product Reviews – Industrial & Scientific |
| **Total Reviews** | 77,071 |
| **Unique Products** | 5,334 |
| **Unique Users** | 11,041 |
| **Average Rating** | 4.52 / 5 ⭐ |
| **Average Review Length** | 241 characters · 44 words |

### Rating Distribution

```
⭐⭐⭐⭐⭐  56,150  ████████████████████████████████░░  72.9%
⭐⭐⭐⭐    12,061  ███████░░░░░░░░░░░░░░░░░░░░░░░░░░  15.6%
⭐⭐⭐      4,442  ██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   5.8%
⭐⭐        1,936  █░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   2.5%
⭐          2,482  █░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   3.2%
```

> ⚠️ **Class imbalance note:** The dataset is heavily skewed toward 5-star reviews (72.9%), which makes macro-averaged F1 the primary evaluation metric throughout the project — it treats each sentiment class equally regardless of frequency.

**Citation:**
> Ni, J., Li, J., & McAuley, J. (2019). *Justifying recommendations using distantly-labeled reviews and fine-grained aspects.* EMNLP-IJCNLP 2019. https://nijianmo.github.io/amazon/index.html

---

## 📘 Phase 1 — Lexicon-Based Sentiment Analysis

Phase 1 builds a full NLP pipeline from raw data to a validated lexicon comparison.

### Step-by-Step Pipeline

```
Step 1  →  Data Exploration        Understand structure, distributions, missing values
Step 2  →  Pre-processing          Label sentiment, remove outliers, clean text
Step 3  →  Lexicon Study           Deep-dive into VADER vs TextBlob design
Step 4  →  Apply Lexicons          Score all reviews with both VADER and TextBlob
Step 5  →  Stratified Sampling     Select 1,000 balanced reviews for evaluation
Step 6  →  Build Models            Threshold-based classifiers using lexicon scores
Step 7  →  Compare & Report        Accuracy, Precision, Recall, F1 comparison table
```

### Pre-processing Details

| Operation | Description |
|-----------|-------------|
| **Sentiment labelling** | 1–2 stars → Negative · 3 stars → Neutral · 4–5 stars → Positive |
| **Column selection** | `reviewText`, `summary`, `overall`, `reviewerID`, `asin` |
| **Outlier removal** | Reviews with < 3 or > 2,000 words removed |
| **Text cleaning** | Lowercase, URL removal, punctuation stripping, NLTK stopword removal |
| **Stratified sampling** | 1,000 reviews proportionally sampled across all 3 sentiment classes |

### Lexicon Comparison — VADER vs TextBlob

| Metric | VADER | TextBlob | Winner |
|--------|-------|----------|--------|
| **Accuracy** | 79.3% | 73.5% | ✅ VADER |
| **Macro F1** | 0.4748 | 0.4332 | ✅ VADER |
| **Weighted F1** | 0.8212 | 0.7854 | ✅ VADER |
| **Macro Recall** | 0.5143 | 0.4559 | ✅ VADER |
| Agreement between models | **79.2%** (792 / 1,000 reviews) | | |

**Why VADER wins on this dataset:**
- Handles exclamation marks, ALL-CAPS, and intensity modifiers — common in product reviews
- Better recall on Negative and Neutral classes
- TextBlob tends to classify more reviews as Neutral (more conservative scoring)

> *Both lexicon models are limited by their fixed vocabulary — they cannot learn domain-specific terms like "defect" or "stripped threads". This motivates Phase 2.*

---

## 📗 Phase 2 — ML Models, Recommender & LLM

Phase 2 applies supervised machine learning, builds a recommender system, and leverages large language models.

### Steps 11–13: TF-IDF + Machine Learning

The same 1,000-review sample is used to train and evaluate two ML classifiers:

```
Review Text
    │
    ▼
TF-IDF Vectorizer (unigrams + bigrams, min_df=3, max_features=10,000)
    │
    ├──▶  Logistic Regression  (C=1.0, max_iter=1000)
    │
    └──▶  Linear SVM           (LinearSVC, C=0.5, dual=False)
```

### Step 14: Cross-Model Comparison

All four models evaluated on the **identical 1,000-review test set** from Phase 1:

```
                        Accuracy    F1-Macro   F1-Weighted
                        ────────    ────────   ───────────
  🥇 Linear SVM          83.9%      0.6363      0.8649
  🥈 Logistic Regression  83.2%      0.6285      0.8597
  🥉 VADER               79.3%      0.4748      0.8212
     TextBlob            73.5%      0.4332      0.7854
```

**Key findings:**
- ML models outperform lexicon-based approaches by **+13–15% in Macro F1**
- The Neutral class (3-star reviews) is the hardest to classify for all models — mixed language makes it ambiguous
- Linear SVM learns domain-specific negative terms (`broke`, `defect`, `return`) that lexicon tools miss
- **Recommendation:** Linear SVM with TF-IDF for production use

### Step 15: Sentiment-Enhanced Recommender System

A **sentiment-aware item-based collaborative filtering** approach that blends star ratings with VADER sentiment scores:

```
Enhanced Rating Formula:
─────────────────────────────────────────────────────────────
  sentiment_score  = (vader_compound + 1) / 2
  normalised_star  = (star_rating - 1) / 4
  enhanced_score   = 0.6 × normalised_star + 0.4 × sentiment_score
  enhanced_rating  = enhanced_score × 4 + 1
─────────────────────────────────────────────────────────────

  Item Similarity  = Cosine(user-item matrix columns)
  Predicted Rating = Σ(sim[i,j] × rating[j]) / Σ(sim[i,j])
```

**Results — Enhanced vs Original Ratings:**

| Metric | Original CF | Sentiment-Enhanced CF | Improvement |
|--------|-------------|----------------------|-------------|
| **MAE** | 0.2445 | 0.2282 | ✅ **+6.7%** |
| **RMSE** | 0.7076 | 0.5316 | ✅ **+24.9%** |

> Blending sentiment scores provides a **smoother, more nuanced signal** — especially valuable for polarising products where star ratings alone are not enough.

### Steps 16–17: LLM Tasks with HuggingFace

Two LLM-powered tasks using pre-trained transformer models:

| Step | Task | Model Used |
|------|------|------------|
| **Step 16** | Automatic review summarisation | `facebook/bart-large-cnn` |
| **Step 17** | CSR (Customer Service) response generation | `google/flan-t5-base` |

**Step 16 — Review Summariser:**
Condenses long product reviews into concise summaries, extracting the key positive and negative points.

**Step 17 — CSR Response Generator:**
Automatically generates professional customer service responses to negative reviews — reducing response time and ensuring consistent tone.

> ⚠️ First run downloads HuggingFace model weights (~250 MB – ~1.6 GB). Subsequent runs use the local cache (`~/.cache/huggingface/`).

---

## 📊 Results at a Glance

```
┌─────────────────────────────────────────────────────────────────┐
│              MODEL PERFORMANCE SUMMARY                          │
├──────────────────────┬──────────┬───────────┬───────────────────┤
│ Model                │ Accuracy │ F1-Macro  │ F1-Weighted       │
├──────────────────────┼──────────┼───────────┼───────────────────┤
│ 🥇 Linear SVM        │  83.9%   │  0.6363   │  0.8649           │
│ 🥈 Logistic Regress. │  83.2%   │  0.6285   │  0.8597           │
│ 🥉 VADER Lexicon     │  79.3%   │  0.4748   │  0.8212           │
│    TextBlob Lexicon  │  73.5%   │  0.4332   │  0.7854           │
├──────────────────────┴──────────┴───────────┴───────────────────┤
│ Recommender System (CF)                                         │
├────────────────────────────────┬────────────┬───────────────────┤
│ Approach                       │    MAE     │      RMSE         │
├────────────────────────────────┼────────────┼───────────────────┤
│ Original Ratings CF            │   0.2445   │    0.7076         │
│ Sentiment-Enhanced CF          │   0.2282   │    0.5316         │
│ Improvement                    │   +6.7%    │   +24.9% ✅       │
└────────────────────────────────┴────────────┴───────────────────┘
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or higher
- Pip package manager
- Amazon Industrial & Scientific review dataset (`.json` or `.json.gz`)

### Installation & Running

**Phase 1:**

```bash
cd Phase#1
pip install -r requirements.txt
python main.py
```

**Phase 2:**

```bash
cd Phase#2
pip install -r requirements_phase2.txt
python main_phase2.py
```

**Phase 2 selective execution:**

```bash
python main_phase2.py --skip-llm           # Skip LLM steps (Steps 16–17)
python main_phase2.py --skip-recommender   # Skip Recommender (Step 15)
python main_phase2.py --skip-comparison    # Skip Cross-comparison (Step 14)
python main_phase2.py --skip-ml            # Skip ML modeling (Steps 11–13)
```

> **Important:** Run Phase 1 before Phase 2 — `sentiment_analysis_results.csv` from Phase 1 is required as input for Step 14.

### Dependencies Overview

| Package | Purpose |
|---------|---------|
| `pandas`, `numpy` | Data manipulation |
| `matplotlib`, `seaborn` | Visualisation |
| `vaderSentiment` | VADER lexicon scorer |
| `textblob` | TextBlob lexicon scorer |
| `nltk` | Tokenisation, stopwords |
| `scikit-learn` | TF-IDF, LR, LinearSVC |
| `transformers`, `torch` | HuggingFace LLM models |
| `wordcloud` | Word cloud visualisations |

---

## 📁 Project Structure

```
sentiment-analysis-recommender-system/
│
├── README.md
├── How to run.txt
│
├── Phase#1/
│   ├── main.py                        # 🚀 Phase 1 entry point
│   ├── data_exploration.py            # Step 1  — Dataset stats & distributions
│   ├── preprocessing.py               # Step 2  — Cleaning, labelling, sampling
│   ├── lexicon_analysis.py            # Step 3  — VADER & TextBlob deep-dive
│   ├── sentiment_analysis.py          # Step 6-7 — Model building & evaluation
│   ├── requirements.txt               # Phase 1 dependencies
│   │
│   ├── preprocessed_data.csv          # Cleaned dataset
│   ├── sentiment_analysis_results.csv # Predictions (1,000 reviews)
│   ├── model_comparison_table.csv     # VADER vs TextBlob metrics
│   ├── data_exploration_summary.txt   # Dataset statistics report
│   ├── lexicon_comparison_report.txt  # Detailed lexicon comparison
│   └── model_comparison_conclusion.txt
│
└── Phase#2/
    ├── main_phase2.py                 # 🚀 Phase 2 entry point
    ├── ml_modeling.py                 # Steps 11-13 — TF-IDF + LR + SVM
    ├── model_comparison.py            # Step 14   — Cross-model comparison
    ├── recommender.py                 # Step 15   — Sentiment-enhanced CF
    ├── llm_tasks.py                   # Steps 16-17 — Summariser + CSR gen
    ├── requirements_phase2.txt        # Phase 2 dependencies
    │
    ├── ml_comparison_table.csv        # ML model metrics
    ├── step14_comparison_table.csv    # All-model comparison
    ├── step14_all_predictions.csv     # All model predictions
    ├── step14_conclusion.txt          # Cross-model conclusion
    ├── step15_cf_results.csv          # Recommender results
    ├── step15_conclusion.txt          # Recommender conclusion
    ├── step16_summaries.txt           # LLM-generated summaries
    ├── step17_csr_response.txt        # LLM-generated CSR responses
    ├── step16_17_report_results.txt   # Full LLM report
    │
    └── models/
        ├── tfidf_vectorizer.pkl       # Trained TF-IDF vectorizer
        ├── logistic_regression.pkl    # Trained LR model
        └── linear_svm.pkl             # Trained SVM model
```

---

## 📚 References

- Ni, J., Li, J., & McAuley, J. (2019). *Justifying recommendations using distantly-labeled reviews and fine-grained aspects.* EMNLP-IJCNLP 2019. https://nijianmo.github.io/amazon/index.html
- Hutto, C. J., & Gilbert, E. (2014). *VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text.* ICWSM 2014.
- Lewis, M., et al. (2020). *BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation.* ACL 2020.
- Chung, H. W., et al. (2022). *Scaling Instruction-Finetuned Language Models (FLAN-T5).* Google Research.

---


