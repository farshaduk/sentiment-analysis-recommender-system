# Phase #2 – Sentiment Analysis & Recommender Systems
**Team #4 | Industrial and Scientific Dataset (Amazon Reviews)**

## Overview

In Phase #2, we applied machine learning techniques for sentiment analysis and compared different models. We also built a recommender system and used LLMs for summarizing reviews.

---

## Files

| File | Description |
|------|-------------|
| `main_phase2.py` | **Entry point** – orchestrates all Phase #2 steps |
| `ml_modeling.py` | Steps 11-13: TF-IDF + Logistic Regression + Linear SVM |
| `model_comparison.py` | Step 14: Compare all 4 models on the Phase #1 lexicon sample |
| `recommender.py` | Step 15: Sentiment-enhanced item-based collaborative filtering |
| `llm_tasks.py` | Steps 16-17: HuggingFace LLM summarisation & CSR response |
| `requirements_phase2.txt` | Python package requirements for Phase #2 |

---

## Setup

```bash
# Install all dependencies (includes Phase #1 packages)
pip install -r requirements_phase2.txt
```

> **Note:** Steps 16-17 will download Hugging Face models (~250 MB – ~1.6 GB) on first run.  
> Subsequent runs use the local cache (`~/.cache/huggingface/`).

---

## Running

```bash
# From the Phase#2 directory:
python main_phase2.py

# Run only specific steps:
python main_phase2.py --skip-llm              # skip LLM (Steps 16-17)
python main_phase2.py --skip-recommender      # skip recommender (Step 15)
python main_phase2.py --skip-comparison       # skip comparison (Step 14)
python main_phase2.py --skip-ml               # skip ML modeling (Steps 11-13)
```

> **Important:** Run Steps 11-13 before Step 14 (comparison needs the trained models).  
> Run Phase #1 (`main.py` in Phase#1) before Step 14 (needs `sentiment_analysis_results.csv`).

---


## Output Files

All outputs are saved in the `Phase#2/` directory:

| File | Step |
|------|------|
| `ml_subset_exploration.png` | 11 |
| `ml_comparison_table.csv` | 13 |
| `ml_model_comparison.png` | 13 |
| `logistic_regression_confusion_matrix.png` | 13 |
| `linear_svm_confusion_matrix.png` | 13 |
| `step14_all_confusion_matrices.png` | 14 |
| `step14_model_comparison.png` | 14 |
| `step14_radar_chart.png` | 14 |
| `step14_comparison_table.csv` | 14 |
| `step14_conclusion.txt` | 14 |
| `step14_all_predictions.csv` | 14 |
| `step15_rating_comparison.png` | 15 |
| `step15_rating_delta.png` | 15 |
| `step15_cf_evaluation.png` | 15 |
| `step15_cf_results.csv` | 15 |
| `step15_conclusion.txt` | 15 |
| `step16_summaries.txt` | 16 |
| `step17_csr_response.txt` | 17 |
| `step16_17_report_results.txt` | 16-17 |
| `models/tfidf_vectorizer.pkl` | 11 |
| `models/logistic_regression.pkl` | 11 |
| `models/linear_svm.pkl` | 11 |

---

## Dataset Reference
Ni, J., Li, J., & McAuley, J. (2019). *Justifying recommendations using distantly-labeled reviews and fine-grained aspects*. EMNLP-IJCNLP 2019.  
https://nijianmo.github.io/amazon/index.html
