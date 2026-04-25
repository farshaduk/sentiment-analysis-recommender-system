"""
Phase #1 – Main Entry Point
Team #4: Industrial and Scientific Dataset

This script orchestrates every Phase #1 deliverable:
    Step 1 – Data exploration
    Step 2 – Text pre-processing (labelling, column selection, outliers, cleaning)
    Step 3 – Lexicon package study & comparison report
    Step 4 – Pre-process text per selected lexicon (VADER & TextBlob)
    Step 5 – Randomly select 1000 reviews (stratified sampling)
    Step 6 – Build two lexicon-based sentiment analysis models
    Step 7 – Validate results & produce comparison table
    Step 10 – All code is documented

Dataset reference:
    Ni, J., Li, J., & McAuley, J. (2019). Justifying recommendations using
    distantly-labeled reviews and fine-grained aspects. EMNLP-IJCNLP 2019.
    https://nijianmo.github.io/amazon/index.html
"""

import pandas as pd
import json
import os
import sys

from data_exploration import DataExplorer
from preprocessing import TextPreprocessor
from lexicon_analysis import LexiconAnalyzer
from sentiment_analysis import SentimentAnalyzer


# ------------------------------------------------------------------ #
#  Data loading
# ------------------------------------------------------------------ #
def load_amazon_data(file_path: str) -> pd.DataFrame:
    """
    Load Amazon review data from a line-delimited JSON file.
    Supports both ``.json`` and ``.json.gz`` formats.

    Args:
        file_path: Path to the review file.

    Returns:
        DataFrame containing all review records.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    import gzip

    print(f"Loading data from {file_path}...")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")

    data = []
    opener = gzip.open if file_path.endswith(".gz") else open

    with opener(file_path, "rt", encoding="utf-8") as fh:
        for line_num, line in enumerate(fh, 1):
            try:
                data.append(json.loads(line.strip()))
            except json.JSONDecodeError as e:
                print(f"  Warning: skipping line {line_num} – {e}")
                continue

    df = pd.DataFrame(data)
    print(f"  Loaded {len(df)} reviews")
    return df


# ------------------------------------------------------------------ #
#  Main pipeline
# ------------------------------------------------------------------ #
def main():
    """Execute the full Phase #1 pipeline."""

    print("=" * 80)
    print("PHASE #1: SENTIMENT ANALYSIS – LEXICON APPROACH")
    print("Team #4: Industrial and Scientific Dataset")
    print("=" * 80)

    # ---- Configuration ----
    DATA_FILE   = "Industrial_and_Scientific.json.gz"  # or .json.gz
    SAMPLE_SIZE = 1000      # Step 5: number of reviews to sample
    RANDOM_SEED = 42        # for reproducibility

    # Also accept .gz variant
    if not os.path.exists(DATA_FILE) and os.path.exists(DATA_FILE + ".gz"):
        DATA_FILE = DATA_FILE + ".gz"

    if not os.path.exists(DATA_FILE):
        # Try common alternative names
        for alt in ("Industrial_and_Scientific_5.json",
                    "Industrial_and_Scientific_5.json.gz"):
            if os.path.exists(alt):
                DATA_FILE = alt
                break

    if not os.path.exists(DATA_FILE):
        print(f"\n  ✗ Data file '{DATA_FILE}' not found.")
        print("  Please download the Industrial and Scientific k-core dataset from:")
        print("    https://nijianmo.github.io/amazon/index.html")
        print("  Place the .json or .json.gz file in this directory.")
        sys.exit(1)

    # ================================================================== #
    #  STEP 1 – Data Exploration
    # ================================================================== #
    print("\n" + "=" * 80)
    print("STEP 1: DATASET DATA EXPLORATION")
    print("=" * 80)

    df = load_amazon_data(DATA_FILE)
    explorer = DataExplorer(df=df)
    exploration_results = explorer.run_full_exploration(save_results=True)

    # ================================================================== #
    #  STEP 2 – Text Pre-processing
    # ================================================================== #
    print("\n" + "=" * 80)
    print("STEP 2: TEXT BASIC PRE-PROCESSING")
    print("=" * 80)

    preprocessor = TextPreprocessor(df)

    # 2-b  Column selection with justification
    column_justification = {
        "reviewText":  "Primary text content – the main input for sentiment analysis.",
        "overall":     "Star rating (1-5) used to derive the ground-truth sentiment label.",
        "summary":     "Short review title that may reinforce or contradict the body text.",
        "asin":        "Product ID – needed to track reviews per product.",
        "reviewerID":  "User ID – needed to track reviews per user.",
    }
    preprocessor.select_columns(
        list(column_justification.keys()), column_justification
    )

    # 2-a  Label sentiment
    preprocessor.label_sentiment("overall")

    # 2-c  Detect outliers
    preprocessor.detect_outliers()

    # Basic text cleaning (Step 4 justification is printed inside)
    preprocessor.clean_text_basic("reviewText")

    # Combine summary + reviewText into a single column for richer sentiment signal
    # Justification: the summary field acts as a headline that often carries
    # concentrated sentiment cues (e.g. "Terrible quality!", "Love it!").
    # Merging it with the review body gives both lexicons more context.
    preprocessor.df["full_text"] = (
        preprocessor.df["summary"].fillna("") + " " +
        preprocessor.df["reviewText"].fillna("")
    ).str.strip()
    print("\n  Created 'full_text' = summary + reviewText for stronger sentiment signal")

    # Save preprocessed data
    preprocessor.save_preprocessed_data("preprocessed_data.csv")
    preprocessed_df = preprocessor.df

    # ================================================================== #
    #  STEP 3 – Study Lexicon Packages
    # ================================================================== #
    print("\n" + "=" * 80)
    print("STEP 3: STUDY LEXICON PACKAGES")
    print("=" * 80)

    lexicon_analyzer = LexiconAnalyzer()

    # Use a mix of actual reviews and constructed examples
    sample_texts = preprocessed_df["reviewText"].dropna().head(15).tolist()
    example_texts = [
        "This product is amazing! It works perfectly and exceeded my expectations.",
        "The item arrived broken and the quality is terrible. Very disappointed.",
        "It's okay, nothing special. Does what it's supposed to do.",
        "GREAT product!!! Highly recommend to everyone!",
        "Not bad, but could be better. The price is reasonable though.",
    ]
    sample_texts = (sample_texts + example_texts)[:20]

    lexicon_analyzer.generate_comparison_report(sample_texts)

    print("\n" + "=" * 80)
    print("LEXICON SELECTION SUMMARY")
    print("=" * 80)
    print("""
  Selected lexicons for model building:
    1. VADER  – Best for product reviews with informal language,
                capitalisation, and punctuation emphasis.
    2. TextBlob – Good general-purpose tool; provides subjectivity.

  Rejected:
    ✗ SentiWordNet – Too slow, heavy preprocessing strips sentiment cues.
    """)

    # ================================================================== #
    #  STEP 4 – Pre-process for Each Lexicon
    # ================================================================== #
    print("\n" + "=" * 80)
    print("STEP 4: PRE-PROCESS TEXT FOR SELECTED LEXICONS")
    print("=" * 80)
    print("""
  VADER pre-processing:
    • Minimal – preserve capitalisation, punctuation, and modifiers.
    • VADER is specifically designed to exploit these signals.

  TextBlob pre-processing:
    • Light cleaning – remove residual HTML, normalise whitespace.
    • TextBlob handles case-folding internally.

  Justification for NOT applying heavier preprocessing:
    • Stop-word removal would remove context words VADER needs.
    • Lowercasing would hide emphasis (e.g., "GREAT" vs "great").
    • Stemming / lemmatisation changes surface forms that both lexicons
      look up in their dictionaries.
    """)

    # ================================================================== #
    #  STEP 5 – Random Sample of 1000 Reviews
    # ================================================================== #
    print("\n" + "=" * 80)
    print(f"STEP 5: RANDOMLY SELECT {SAMPLE_SIZE} REVIEWS (STRATIFIED)")
    print("=" * 80)

    # The SentimentAnalyzer.build_models() handles stratified sampling
    # internally – see sentiment_analysis.py for details.

    # ================================================================== #
    #  STEP 6 – Build Sentiment Analysis Models
    # ================================================================== #
    print("\n" + "=" * 80)
    print("STEP 6: BUILD SENTIMENT ANALYSIS MODELS (LEXICON APPROACH)")
    print("=" * 80)

    sentiment_analyzer = SentimentAnalyzer(
        preprocessed_df,
        text_col="full_text",          # summary + reviewText for richer context
        label_col="sentiment_label",
    )
    sentiment_analyzer.build_models(
        sample_size=SAMPLE_SIZE, random_state=RANDOM_SEED
    )

    # ================================================================== #
    #  STEP 7 – Validate & Compare Results
    # ================================================================== #
    print("\n" + "=" * 80)
    print("STEP 7: VALIDATE RESULTS AND PROVIDE COMPARISON TABLE")
    print("=" * 80)

    comparison_results = sentiment_analyzer.compare_models()
    sentiment_analyzer.save_results("sentiment_analysis_results.csv")

    # ================================================================== #
    #  Final summary
    # ================================================================== #
    print("\n" + "=" * 80)
    print("PHASE #1 COMPLETE – SUMMARY OF GENERATED FILES")
    print("=" * 80)
    generated = [
        ("data_exploration_summary.txt",       "Step 1 – Exploration results"),
        ("reviews_per_product_distribution.png","Step 1 – Product review distribution"),
        ("reviews_per_user_distribution.png",  "Step 1 – User review distribution"),
        ("review_length_analysis.png",         "Step 1 – Review length analysis"),
        ("rating_distribution.png",            "Step 1 – Rating distribution"),
        ("rating_vs_length.png",               "Step 1 – Rating vs review length"),
        ("word_cloud.png",                     "Step 1 – Word cloud"),
        ("sentiment_distribution.png",         "Step 2 – Sentiment label distribution"),
        ("preprocessed_data.csv",              "Step 2 – Preprocessed dataset"),
        ("lexicon_comparison_report.txt",      "Step 3 – Lexicon comparison report"),
        ("confusion_matrix_vader.png",         "Step 7 – VADER confusion matrix"),
        ("confusion_matrix_textblob.png",      "Step 7 – TextBlob confusion matrix"),
        ("model_comparison.png",               "Step 7 – Side-by-side model comparison"),
        ("model_comparison_table.csv",         "Step 7 – Metrics comparison table"),
        ("model_comparison_conclusion.txt",    "Step 7 – Written conclusion"),
        ("sentiment_analysis_results.csv",     "Step 7 – Full prediction results"),
    ]
    for fname, desc in generated:
        status = "✓" if os.path.exists(fname) else "✗"
        print(f"  {status}  {fname:45s} {desc}")

     


if __name__ == "__main__":
    main()
