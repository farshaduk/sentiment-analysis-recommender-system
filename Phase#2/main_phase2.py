"""
Phase #2 - Main Entry Point
Team #4 : Industrial and Scientific Dataset

This script orchestrates every Phase #2 deliverable:

    Step 11 - ML approach: subset selection, exploration, pre-processing,
              TF-IDF representation, stratified 70/30 split, two ML models
              (Logistic Regression + Linear SVM).
    Step 12 - Training process (cross-validation, grid search).
    Step 13 - Testing: accuracy, precision, recall, F1, confusion matrix.
    Step 14 - Cross-model comparison: Lexicon models (VADER, TextBlob) vs.
              ML models (LR, SVM) on the identical 1000-review sample.
    Step 15 - Recommender system: sentiment-enhanced item-based CF.
    Step 16 - LLM summarisation: 10 reviews >100 words -> <=50-word summaries.
    Step 17 - LLM CSR response: auto-generate reply to a question review.

Usage
-----
    python main_phase2.py

    Optional flags (can be combined):
        --skip-ml          Skip ML modeling (Steps 11-13)
        --skip-comparison  Skip comparison experiment (Step 14)
        --skip-recommender Skip recommender system (Step 15)
        --skip-llm         Skip LLM tasks       (Steps 16-17)

Dataset reference:
    Ni, J., Li, J., & McAuley, J. (2019). Justifying recommendations using
    distantly-labeled reviews and fine-grained aspects. EMNLP-IJCNLP 2019.
    https://nijianmo.github.io/amazon/index.html
"""

import os
import sys
import argparse

# Ensure the Phase #2 directory is on the path
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from ml_modeling     import MLModeler
from model_comparison import ModelComparison
from recommender     import SentimentRecommender
from llm_tasks       import LLMTasks


# ------------------------------------------------------------------ #
#  Configuration
# ------------------------------------------------------------------ #
# Relative path from Phase#2/ to Phase#1/
_PHASE1_DIR = os.path.join(_THIS_DIR, "..", "Phase#1")
_PHASE2_DIR = _THIS_DIR


_DATA_FILE = os.path.join(_PHASE1_DIR, "Industrial_and_Scientific.json.gz")

# Fallback names if the primary name is not found
_DATA_FILE_ALTERNATIVES = [
    os.path.join(_PHASE1_DIR, "Industrial_and_Scientific.json"),
    os.path.join(_PHASE1_DIR, "Industrial_and_Scientific_5.json.gz"),
    os.path.join(_PHASE1_DIR, "Industrial_and_Scientific_5.json"),
]


def resolve_data_file() -> str:
    """Return the first existing data file path from the known alternatives."""
    if os.path.exists(_DATA_FILE):
        return _DATA_FILE
    for alt in _DATA_FILE_ALTERNATIVES:
        if os.path.exists(alt):
            return alt
    return _DATA_FILE   # will raise a FileNotFoundError inside the loader


def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase #2 Pipeline - Team #4 Industrial & Scientific"
    )
    parser.add_argument("--skip-ml",          action="store_true",
                        help="Skip ML modeling (Steps 11-13)")
    parser.add_argument("--skip-comparison",  action="store_true",
                        help="Skip cross-model comparison (Step 14)")
    parser.add_argument("--skip-recommender", action="store_true",
                        help="Skip recommender system (Step 15)")
    parser.add_argument("--skip-llm",         action="store_true",
                        help="Skip LLM tasks (Steps 16-17)")
    return parser.parse_args()


# ------------------------------------------------------------------ #
#  Main pipeline
# ------------------------------------------------------------------ #
def main():
    """Execute the full Phase #2 pipeline."""

    print("=" * 80)
    print("PHASE #2: SENTIMENT ANALYSIS & RECOMMENDER SYSTEMS")
    print("Team #4: Industrial and Scientific Dataset")
    print("=" * 80)

    args      = parse_args()
    data_file = resolve_data_file()

    print(f"\n  Data file  : {data_file}")
    print(f"  Phase#1 dir: {_PHASE1_DIR}")
    print(f"  Phase#2 dir: {_PHASE2_DIR}")
    print(f"  Data file exists: {os.path.exists(data_file)}")

    if not os.path.exists(data_file):
        print(
            "\n  [FAIL] ERROR: Data file not found.\n"
            "  Please download the Industrial and Scientific dataset from:\n"
            "    https://nijianmo.github.io/amazon/index.html\n"
            "  and place it in the Phase#1 directory."
        )
        sys.exit(1)

    ml_results          = None
    comparison_results  = None
    recommender_results = None
    llm_results         = None

    # ================================================================== #
    #  STEPS 11-13 - Machine Learning Approach
    # ================================================================== #
    if not args.skip_ml:
        print("\n" + "=" * 80)
        print("STEPS 11-13: MACHINE LEARNING APPROACH")
        print("=" * 80)

        modeler = MLModeler(
            data_file  = data_file,
            output_dir = _PHASE2_DIR,
            phase1_dir = _PHASE1_DIR,
        )
        ml_results = modeler.run_full_pipeline()
        print("\n  [OK] Steps 11-13 complete.")
    else:
        print("\n  [Skipped] Steps 11-13 (ML modeling)")

    # ================================================================== #
    #  STEP 14 - Cross-Model Comparison
    # ================================================================== #
    if not args.skip_comparison:
        print("\n" + "=" * 80)
        print("STEP 14: CROSS-MODEL COMPARISON EXPERIMENT")
        print("=" * 80)

        comparator = ModelComparison(
            phase1_dir = _PHASE1_DIR,
            phase2_dir = _PHASE2_DIR,
            output_dir = _PHASE2_DIR,
        )
        try:
            comparison_results = comparator.run()
            print("\n  [OK] Step 14 complete.")
        except FileNotFoundError as exc:
            print(f"\n  Warning: Step 14 skipped - {exc}")
            print("  Ensure both Phase #1 lexicon results and Phase #2 ML "
                  "models are available before running Step 14.")
    else:
        print("\n  [Skipped] Step 14 (model comparison)")

    # ================================================================== #
    #  STEP 15 - Recommender System
    # ================================================================== #
    if not args.skip_recommender:
        print("\n" + "=" * 80)
        print("STEP 15: RECOMMENDER SYSTEM")
        print("=" * 80)

        recommender = SentimentRecommender(
            data_file  = data_file,
            output_dir = _PHASE2_DIR,
        )
        recommender_results = recommender.run()
        print("\n  [OK] Step 15 complete.")
    else:
        print("\n  [Skipped] Step 15 (recommender system)")

    # ================================================================== #
    #  STEPS 16-17 - LLM Tasks
    # ================================================================== #
    if not args.skip_llm:
        print("\n" + "=" * 80)
        print("STEPS 16-17: LLM TASKS")
        print("=" * 80)
        print(
            "  Note: These steps download Hugging Face models (~250 MB - ~1 GB)\n"
            "  on the first run.  Subsequent runs use the local cache.\n"
            "  Requires: pip install transformers torch"
        )

        llm = LLMTasks(
            data_file  = data_file,
            output_dir = _PHASE2_DIR,
        )
        try:
            llm_results = llm.run()
            print("\n  [OK] Steps 16-17 complete.")
        except (ImportError, RuntimeError) as exc:
            print(
                f"\n  Warning: LLM steps skipped - {exc}\n"
                "  Install required packages:\n"
                "    pip install transformers torch sentencepiece sacremoses"
            )
    else:
        print("\n  [Skipped] Steps 16-17 (LLM tasks)")

    # ================================================================== #
    #  Final Summary
    # ================================================================== #
    print("\n" + "=" * 80)
    print("PHASE #2 PIPELINE - SUMMARY")
    print("=" * 80)

    if ml_results:
        lr  = ml_results.get("lr",  {})
        svm = ml_results.get("svm", {})
        print(f"\n  ML Results (Test Set):")
        print(f"    Logistic Regression - Accuracy: {lr.get('accuracy', 'N/A'):.4f}  "
              f"F1(macro): {lr.get('f1_macro', 'N/A'):.4f}")
        print(f"    Linear SVM          - Accuracy: {svm.get('accuracy', 'N/A'):.4f}  "
              f"F1(macro): {svm.get('f1_macro', 'N/A'):.4f}")

    if comparison_results:
        print(f"\n  Cross-Model Comparison (1000-Review Lexicon Sample):")
        for name, res in comparison_results.items():
            print(f"    {name:<25} F1(macro): {res.get('f1_macro', 'N/A'):.4f}")

    if recommender_results:
        oe = recommender_results.get("original_eval", {})
        ee = recommender_results.get("enhanced_eval", {})
        print(f"\n  Recommender System:")
        print(f"    Original CF   - MAE: {oe.get('mae', 'N/A'):.4f}  "
              f"RMSE: {oe.get('rmse', 'N/A'):.4f}")
        print(f"    Enhanced CF   - MAE: {ee.get('mae', 'N/A'):.4f}  "
              f"RMSE: {ee.get('rmse', 'N/A'):.4f}")

    if llm_results:
        n16 = len(llm_results.get("step16", []))
        print(f"\n  LLM Tasks:")
        print(f"    Step 16: {n16} reviews summarised")
        print(f"    Step 17: CSR response generated")

    print("\n" + "=" * 80)
    print("Phase #2 Pipeline Complete.")
    print("Output files are saved in the Phase#2 directory.")
    print("=" * 80)


if __name__ == "__main__":
    main()
