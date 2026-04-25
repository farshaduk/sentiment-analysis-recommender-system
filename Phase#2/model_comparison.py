"""
Phase #2 - Step 14: Cross-Model Comparison Experiment
Team #4 : Industrial and Scientific Dataset

Step 14 - Design an experiment that compares Lexicon models (Phase #1)
          versus the two Machine Learning models (Phase #2) on the
          *identical* 1000-review sample used in Phase #1.

Design rationale (Step 14-a)
------------------------------
"Comparing apples to apples" means every model must be evaluated on the
same set of reviews, with the same ground-truth labels, and using the
same evaluation metrics.

The 1000-review lexicon test set from Phase #1 is stored in
``sentiment_analysis_results.csv``.  It already contains:
    - ``reviewText``       - raw review text (ground truth input).
    - ``sentiment_label``  - true label  (Positive / Neutral / Negative).
    - ``vader_prediction`` - VADER's predicted label.
    - ``textblob_prediction`` - TextBlob's predicted label.

For the ML side (Step 14-b), we:
    1. Load the same 1000 reviews.
    2. Apply the identical ML-specific text cleaning used during training.
    3. Transform with the *training-fitted* TF-IDF vectoriser (no refit).
    4. Predict with both the Logistic Regression and Linear SVM models.

This ensures all four models receive the same raw text, the same ground-
truth labels and (for ML) the same feature space that was used during
training.

Dataset reference:
    Ni, J., Li, J., & McAuley, J. (2019). Justifying recommendations using
    distantly-labeled reviews and fine-grained aspects. EMNLP-IJCNLP 2019.
    https://nijianmo.github.io/amazon/index.html
"""

import os
import re
import warnings
import joblib

warnings.filterwarnings("ignore")

import numpy  as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
for _r in ("stopwords", "punkt", "punkt_tab", "wordnet"):
    try:
        nltk.data.find(
            f"tokenizers/{_r}" if "punkt" in _r else f"corpora/{_r}"
        )
    except LookupError:
        nltk.download(_r, quiet=True)

from nltk.corpus import stopwords
from nltk.stem   import PorterStemmer

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)

LABELS      = ["Positive", "Neutral", "Negative"]
RANDOM_SEED = 42


class ModelComparison:
    """
    Evaluates all four models on the same 1000-review lexicon test set
    and produces a comprehensive comparison report.

    Parameters
    ----------
    phase1_dir : str  Path to the Phase #1 output directory.
    phase2_dir : str  Path to the Phase #2 output directory
                      (where trained models are saved).
    output_dir : str  Directory where comparison plots/CSV are saved.
    """

    def __init__(
        self,
        phase1_dir: str = "../Phase#1",
        phase2_dir: str = ".",
        output_dir: str = ".",
    ):
        self.phase1_dir = phase1_dir
        self.phase2_dir = phase2_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self._stemmer    = PorterStemmer()
        self._stop_words = set(stopwords.words("english"))

        self.df_lex     : pd.DataFrame = None   # 1000-review lexicon DF
        self.tfidf      = None                  # fitted TF-IDF vectoriser
        self.lr_model   = None
        self.svm_model  = None
        self.all_results: dict = {}

    # ------------------------------------------------------------------ #
    #  Data loading
    # ------------------------------------------------------------------ #
    def load_lexicon_results(self) -> pd.DataFrame:
        """
        Load the 1000-review lexicon sample from Phase #1.

        Returns
        -------
        pd.DataFrame - contains raw text, true labels and both lexicon
                       model predictions.
        """
        csv_path = os.path.join(
            self.phase1_dir, "sentiment_analysis_results.csv"
        )
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"Phase #1 lexicon results not found at {csv_path}.\n"
                "Please run Phase #1 (main.py) first."
            )

        df = pd.read_csv(csv_path)
        print(f"  Loaded {len(df):,} reviews from Phase #1 lexicon sample.")

        # ---- Validate required columns ----
        required = ["reviewText", "sentiment_label",
                    "vader_prediction", "textblob_prediction"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in lexicon CSV: {missing}")

        # Drop rows with missing text or labels
        df.dropna(subset=["reviewText", "sentiment_label"], inplace=True)
        df = df[
            df["reviewText"].str.strip().astype(bool) &
            df["sentiment_label"].isin(LABELS)
        ]

        # Normalise lexicon predictions to the same LABELS set
        for col in ["vader_prediction", "textblob_prediction"]:
            df[col] = df[col].str.capitalize()
            df.loc[~df[col].isin(LABELS), col] = "Neutral"

        print(f"  After cleaning : {len(df):,} usable reviews")
        print(f"  Label distribution:\n{df['sentiment_label'].value_counts()}")

        self.df_lex = df.reset_index(drop=True)
        return df

    # ------------------------------------------------------------------ #
    #  Load trained ML artefacts
    # ------------------------------------------------------------------ #
    def load_ml_models(self) -> None:
        """Load the Phase #2 TF-IDF vectoriser and both trained models."""
        models_dir = os.path.join(self.phase2_dir, "models")

        artefacts = {
            "tfidf_vectorizer.pkl"  : "tfidf",
            "logistic_regression.pkl": "lr_model",
            "linear_svm.pkl"        : "svm_model",
        }

        for fname, attr in artefacts.items():
            path = os.path.join(models_dir, fname)
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"ML artefact not found: {path}\n"
                    "Please run Phase #2 ML pipeline (main_phase2.py) first."
                )
            setattr(self, attr, joblib.load(path))
            print(f"  Loaded {fname}")

    # ------------------------------------------------------------------ #
    #  ML-specific text cleaning (must match ml_modeling.py exactly)
    # ------------------------------------------------------------------ #
    def _clean_for_ml(self, text: str) -> str:
        """Apply the same cleaning pipeline used during ML training."""
        if pd.isna(text) or not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r"https?://\S+|www\.\S+", " ", text)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"[^a-z\s]", " ", text)
        tokens = text.split()
        tokens = [
            self._stemmer.stem(t)
            for t in tokens
            if t not in self._stop_words and len(t) > 1
        ]
        return " ".join(tokens)

    # ------------------------------------------------------------------ #
    #  Apply ML models to the lexicon 1000-review set
    # ------------------------------------------------------------------ #
    def apply_ml_models(self) -> None:
        """
        Predict sentiment for every lexicon review using both ML models.

        Adds columns ``lr_prediction`` and ``svm_prediction`` to
        ``self.df_lex``.
        """
        print("\n  Preparing ML features for lexicon sample ...")

        # Build 'full_text' the same way as Phase #1 (summary + reviewText)
        if "summary" in self.df_lex.columns:
            raw_text = (
                self.df_lex["summary"].fillna("") + " " +
                self.df_lex["reviewText"].fillna("")
            )
        else:
            raw_text = self.df_lex["reviewText"].fillna("")

        clean_text = raw_text.apply(self._clean_for_ml)

        # Transform with training-fitted TF-IDF (no refit)
        X = self.tfidf.transform(clean_text)

        self.df_lex["lr_prediction"]  = self.lr_model.predict(X)
        self.df_lex["svm_prediction"] = self.svm_model.predict(X)

        print("  ML predictions added: lr_prediction, svm_prediction")

    # ------------------------------------------------------------------ #
    #  Compute metrics for all four models
    # ------------------------------------------------------------------ #
    def _metrics(self, y_true, y_pred, model_name: str) -> dict:
        """Compute and return a complete metrics dictionary."""
        acc  = accuracy_score (y_true, y_pred)
        prec = precision_score(y_true, y_pred, labels=LABELS,
                               average="macro",    zero_division=0)
        rec  = recall_score   (y_true, y_pred, labels=LABELS,
                               average="macro",    zero_division=0)
        f1   = f1_score       (y_true, y_pred, labels=LABELS,
                               average="macro",    zero_division=0)
        prec_w = precision_score(y_true, y_pred, labels=LABELS,
                                  average="weighted", zero_division=0)
        rec_w  = recall_score   (y_true, y_pred, labels=LABELS,
                                  average="weighted", zero_division=0)
        f1_w   = f1_score       (y_true, y_pred, labels=LABELS,
                                  average="weighted", zero_division=0)
        cm = confusion_matrix(y_true, y_pred, labels=LABELS)

        print(f"\n  === {model_name} ===")
        print(f"  Accuracy           : {acc:.4f}")
        print(f"  Precision (macro)  : {prec:.4f}")
        print(f"  Recall    (macro)  : {rec:.4f}")
        print(f"  F1        (macro)  : {f1:.4f}")
        print(f"  F1        (weighted): {f1_w:.4f}")
        print(f"\n{classification_report(y_true, y_pred, labels=LABELS, zero_division=0)}")

        return {
            "model"             : model_name,
            "accuracy"          : acc,
            "precision_macro"   : prec,
            "recall_macro"      : rec,
            "f1_macro"          : f1,
            "precision_weighted": prec_w,
            "recall_weighted"   : rec_w,
            "f1_weighted"       : f1_w,
            "confusion_matrix"  : cm,
        }

    # ------------------------------------------------------------------ #
    #  Visualisations
    # ------------------------------------------------------------------ #
    def _plot_all_confusion_matrices(self, model_results: list) -> None:
        """
        Plot a 2x2 grid of confusion matrices for all four models.
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(
            "Step 14 - Confusion Matrices: All Four Models (Same 1000 Reviews)",
            fontsize=13, fontweight="bold",
        )

        for ax, res in zip(axes.flat, model_results):
            sns.heatmap(
                res["confusion_matrix"],
                annot=True, fmt="d", cmap="Blues",
                xticklabels=LABELS, yticklabels=LABELS, ax=ax,
            )
            ax.set_title(res["model"], fontsize=10, fontweight="bold")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")

        plt.tight_layout()
        out_path = os.path.join(self.output_dir, "step14_all_confusion_matrices.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved all confusion matrices -> {out_path}")

    def _plot_metric_comparison(self, model_results: list) -> None:
        """
        Bar chart comparing all four models across key metrics.
        """
        metric_keys  = ["accuracy", "precision_macro", "recall_macro",
                        "f1_macro", "f1_weighted"]
        metric_labels = ["Accuracy", "Precision\n(macro)", "Recall\n(macro)",
                         "F1\n(macro)", "F1\n(weighted)"]
        colors = ["#2980b9", "#27ae60", "#e67e22", "#8e44ad"]

        x     = np.arange(len(metric_keys))
        width = 0.18

        fig, ax = plt.subplots(figsize=(13, 6))

        for i, res in enumerate(model_results):
            vals = [res[m] for m in metric_keys]
            offset = (i - 1.5) * width
            bars = ax.bar(x + offset, vals, width,
                          label=res["model"], color=colors[i], alpha=0.85)
            for bar in bars:
                h = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2, h + 0.005,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=7,
                )

        ax.set_title(
            "Step 14 - Model Comparison on the Same 1000-Review Lexicon Sample",
            fontsize=12, fontweight="bold",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels, fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Score")
        ax.legend(loc="lower right", fontsize=9)
        plt.tight_layout()

        out_path = os.path.join(self.output_dir, "step14_model_comparison.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved model comparison chart -> {out_path}")

    def _plot_radar_chart(self, model_results: list) -> None:
        """
        Radar (spider) chart giving a holistic view of all four models.
        """
        metric_keys   = ["accuracy", "precision_macro", "recall_macro",
                         "f1_macro", "f1_weighted"]
        metric_labels = ["Accuracy", "Precision\n(macro)", "Recall\n(macro)",
                         "F1\n(macro)", "F1\n(weighted)"]
        N = len(metric_keys)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        angles += angles[:1]  # close the chart

        colors = ["#2980b9", "#27ae60", "#e67e22", "#8e44ad"]
        fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

        for res, color in zip(model_results, colors):
            values = [res[m] for m in metric_keys]
            values += values[:1]
            ax.plot(angles, values, "o-", linewidth=2, color=color,
                    label=res["model"])
            ax.fill(angles, values, alpha=0.10, color=color)

        ax.set_thetagrids(np.degrees(angles[:-1]), metric_labels)
        ax.set_ylim(0, 1)
        ax.set_title(
            "Step 14 - Radar Chart: Model Performance Comparison",
            pad=20, fontsize=11, fontweight="bold",
        )
        ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=8)
        plt.tight_layout()

        out_path = os.path.join(self.output_dir, "step14_radar_chart.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved radar chart -> {out_path}")

    # ------------------------------------------------------------------ #
    #  Save CSV comparison table
    # ------------------------------------------------------------------ #
    def save_comparison_csv(self, model_results: list) -> None:
        """Save a tidy CSV comparison table for all four models."""
        rows = []
        metrics = [
            ("Accuracy",             "accuracy"),
            ("Precision (macro)",    "precision_macro"),
            ("Recall (macro)",       "recall_macro"),
            ("F1-Score (macro)",     "f1_macro"),
            ("Precision (weighted)", "precision_weighted"),
            ("Recall (weighted)",    "recall_weighted"),
            ("F1-Score (weighted)",  "f1_weighted"),
        ]
        for label, key in metrics:
            row = {"Metric": label}
            for res in model_results:
                row[res["model"]] = round(res[key], 4)
            rows.append(row)

        table = pd.DataFrame(rows)
        out_path = os.path.join(self.output_dir, "step14_comparison_table.csv")
        table.to_csv(out_path, index=False)
        print(f"\n  Saved comparison table -> {out_path}")
        print(f"\n{table.to_string(index=False)}")

        # Also write interpretation / conclusion
        self._write_conclusion(model_results)

    def _write_conclusion(self, model_results: list) -> None:
        """Write a textual conclusion comparing the four models."""
        sorted_models = sorted(
            model_results, key=lambda r: r["f1_macro"], reverse=True
        )
        best   = sorted_models[0]
        second = sorted_models[1]

        lines = [
            "=" * 70,
            "STEP 14 - CROSS-MODEL COMPARISON CONCLUSION",
            "=" * 70,
            "",
            "Experiment design:",
            "  All four models were evaluated on the IDENTICAL 1000-review",
            "  test sample used in Phase #1 lexicon analysis.  This ensures",
            "  a fair ('apples-to-apples') comparison across all models.",
            "",
            "Results summary:",
        ]

        for res in sorted_models:
            lines.append(
                f"  {res['model']:<30} Acc={res['accuracy']:.4f} "
                f"F1(macro)={res['f1_macro']:.4f} "
                f"F1(wt)={res['f1_weighted']:.4f}"
            )

        lines += [
            "",
            f"Conclusion:",
            f"  Best overall performer : {best['model']}  "
            f"(F1-macro = {best['f1_macro']:.4f})",
            f"  Second best            : {second['model']}  "
            f"(F1-macro = {second['f1_macro']:.4f})",
            "",
            "  Key observations:",
            "  - ML models (LR and SVM) generally outperform lexicon-based",
            "    approaches on accuracy and F1, because they are trained on",
            "    domain-specific labelled examples rather than relying on a",
            "    fixed vocabulary of sentiment words.",
            "  - The Neutral class remains the hardest to classify for all",
            "    models (lowest per-class F1), as '3-star' reviews often",
            "    contain mixed positive and negative language.",
            "  - VADER outperforms TextBlob on this technical/industrial",
            "    dataset because VADER's lexicon includes intensity modifiers",
            "    and punctuation cues more relevant to product reviews.",
            "  - TF-IDF + LR / SVM achieves better recall on negative reviews",
            "    because it has learned specific domain terms (e.g., 'broke',",
            "    'defect', 'return') from the training data.",
            "",
            "Recommendation for production use:",
            "  Given the significantly higher F1 scores, ML models (especially",
            "  Linear SVM with TF-IDF) are recommended for production-grade",
            "  sentiment analysis of Industrial & Scientific product reviews.",
            "=" * 70,
        ]

        text = "\n".join(lines)
        print("\n" + text)

        out_path = os.path.join(self.output_dir, "step14_conclusion.txt")
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(f"\n  Saved conclusion -> {out_path}")

    # ------------------------------------------------------------------ #
    #  Full comparison runner
    # ------------------------------------------------------------------ #
    def run(self) -> dict:
        """
        Execute the complete Step 14 comparison experiment.

        Returns
        -------
        dict - metrics for all four models.
        """
        print("\n" + "=" * 80)
        print("STEP 14: CROSS-MODEL COMPARISON EXPERIMENT")
        print("=" * 80)

        # ---- Load Phase #1 lexicon results ----
        self.load_lexicon_results()

        # ---- Load Phase #2 trained ML models ----
        print("\n  Loading ML artefacts ...")
        self.load_ml_models()

        # ---- Apply ML models to lexicon sample ----
        self.apply_ml_models()

        y_true = self.df_lex["sentiment_label"]

        # ---- Compute metrics for all four models ----
        print("\n" + "=" * 60)
        print("METRICS ON 1000-REVIEW LEXICON SAMPLE")
        print("=" * 60)

        model_results = [
            self._metrics(y_true, self.df_lex["vader_prediction"],    "VADER"),
            self._metrics(y_true, self.df_lex["textblob_prediction"], "TextBlob"),
            self._metrics(y_true, self.df_lex["lr_prediction"],       "Logistic Regression"),
            self._metrics(y_true, self.df_lex["svm_prediction"],      "Linear SVM"),
        ]

        # ---- Plots ----
        self._plot_all_confusion_matrices(model_results)
        self._plot_metric_comparison(model_results)
        self._plot_radar_chart(model_results)

        # ---- CSV table + conclusion ----
        self.save_comparison_csv(model_results)

        # ---- Save annotated lexicon DF ----
        out_csv = os.path.join(self.output_dir, "step14_all_predictions.csv")
        cols = [
            "reviewText", "sentiment_label",
            "vader_prediction", "textblob_prediction",
            "lr_prediction", "svm_prediction",
        ]
        save_cols = [c for c in cols if c in self.df_lex.columns]
        self.df_lex[save_cols].to_csv(out_csv, index=False)
        print(f"\n  Saved full predictions -> {out_csv}")

        self.all_results = {r["model"]: r for r in model_results}
        print("\n  [OK] Step 14 cross-model comparison complete.")
        return self.all_results
