"""
Phase #1 - Steps 5-7: Sentiment Analysis Using the Lexicon Approach
Team #4: Industrial and Scientific Dataset

This module:
    Step 5 – Randomly selects 1000 reviews (stratified by sentiment label).
    Step 6 – Builds two sentiment analysis models (VADER & TextBlob).
    Step 7 – Validates both models and produces a comparison table with
             accuracy, precision, recall, F1-score, confusion matrices,
             per-class metrics, score distributions, and a written conclusion.

Pre-processing rationale (Step 4):
    VADER  – Minimal: keep capitalisation, punctuation, modifiers.
    TextBlob – Light cleaning: normalise whitespace, strip HTML artefacts.
    Both lexicons work best with near-raw text; heavy preprocessing
    (stop-word removal, stemming) would strip the cues they rely on.
"""

import pandas as pd
import numpy as np
import re
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

import nltk
for _r in ("stopwords", "punkt", "punkt_tab"):
    try:
        nltk.data.find(f"tokenizers/{_r}" if "punkt" in _r else f"corpora/{_r}")
    except LookupError:
        nltk.download(_r, quiet=True)

from nltk.corpus import stopwords
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)

# Ordered label list used everywhere for consistency
LABELS = ["Positive", "Neutral", "Negative"]


class SentimentAnalyzer:
    """Build, evaluate, and compare VADER and TextBlob lexicon models."""

    def __init__(
        self,
        df: pd.DataFrame,
        text_col: str = "reviewText",
        label_col: str = "sentiment_label",
    ):
        """
        Args:
            df:        DataFrame that includes the text and true-label columns.
            text_col:  Column name containing review text.
            label_col: Column name containing the ground-truth sentiment label.
        """
        self.df = df.copy()
        self.text_col = text_col
        self.label_col = label_col
        self.vader = SentimentIntensityAnalyzer()
        self.stop_words = set(stopwords.words("english"))

    # ------------------------------------------------------------------ #
    #  Step 4 – Per-lexicon preprocessing
    # ------------------------------------------------------------------ #
    @staticmethod
    def preprocess_for_vader(text: str) -> str:
        """
        VADER pre-processing (minimal by design).

        VADER is built to exploit capitalisation, punctuation emphasis,
        and degree modifiers.  We therefore only:
        * ensure the value is a non-null string
        * strip leading / trailing whitespace

        **Not applied:** lowercasing, stop-word removal, stemming.
        """
        if pd.isna(text):
            return ""
        return str(text).strip()

    @staticmethod
    def preprocess_for_textblob(text: str) -> str:
        """
        TextBlob pre-processing (light cleaning).

        TextBlob benefits from:
        * whitespace normalisation
        * removal of residual HTML tags

        **Not applied:** lowercasing (TextBlob handles it internally),
        stop-word removal, stemming.
        """
        if pd.isna(text):
            return ""
        t = str(text)
        t = re.sub(r"<[^>]+>", " ", t)        # strip HTML tags
        t = re.sub(r"\s+", " ", t).strip()     # normalise whitespace
        return t

    # ------------------------------------------------------------------ #
    #  Predictions
    # ------------------------------------------------------------------ #
    def predict_vader(self, text: str) -> tuple:
        """Return (label, compound_score) for VADER."""
        t = self.preprocess_for_vader(text)
        s = self.vader.polarity_scores(t)
        c = s["compound"]
        if c >= 0.05:
            return "Positive", c
        if c <= -0.05:
            return "Negative", c
        return "Neutral", c

    def predict_textblob(self, text: str) -> tuple:
        """Return (label, polarity) for TextBlob."""
        t = self.preprocess_for_textblob(text)
        p = TextBlob(t).sentiment.polarity
        if p > 0.1:
            return "Positive", p
        if p < -0.1:
            return "Negative", p
        return "Neutral", p

    # ------------------------------------------------------------------ #
    #  Step 5 & 6 – Sampling + model building
    # ------------------------------------------------------------------ #
    def build_models(
        self,
        sample_size: int = 1000,
        random_state: int = 42,
    ) -> pd.DataFrame:
        """
        Build both lexicon models on a random stratified sample.

        Uses stratified sampling by ``sentiment_label`` so that the class
        distribution in the sample mirrors the full dataset.

        Args:
            sample_size: Target number of reviews.
            random_state: Seed for reproducibility.

        Returns:
            DataFrame with prediction and score columns added.
        """
        print("\n" + "=" * 60)
        print("BUILDING SENTIMENT ANALYSIS MODELS")
        print("=" * 60)

        # --- Stratified sampling (Step 5) ---
        working = self.df.dropna(subset=[self.text_col, self.label_col])

        if len(working) > sample_size:
            try:
                sampled = working.groupby(self.label_col, group_keys=False).apply(
                    lambda g: g.sample(
                        n=max(1, int(sample_size * len(g) / len(working))),
                        random_state=random_state,
                    )
                )
                # Adjust to exact sample_size
                if len(sampled) < sample_size:
                    remaining = working.drop(sampled.index)
                    extra = remaining.sample(
                        n=sample_size - len(sampled), random_state=random_state
                    )
                    sampled = pd.concat([sampled, extra])
                elif len(sampled) > sample_size:
                    sampled = sampled.sample(n=sample_size, random_state=random_state)
                print(f"  Stratified sample: {len(sampled)} reviews from {len(working)}")
            except Exception:
                # Fall back to simple random sample
                sampled = working.sample(n=sample_size, random_state=random_state)
                print(f"  Random sample: {len(sampled)} reviews from {len(working)}")
        else:
            sampled = working.copy()
            print(f"  Using all {len(sampled)} reviews (< {sample_size})")

        # Print sample class distribution
        dist = sampled[self.label_col].value_counts()
        print("\n  Sample class distribution:")
        for lbl, cnt in dist.items():
            print(f"    {lbl}: {cnt} ({cnt / len(sampled) * 100:.1f}%)")

        # --- VADER predictions ---
        print("\n  Generating VADER predictions...")
        vader_results = sampled[self.text_col].apply(self.predict_vader)
        sampled["vader_prediction"] = vader_results.apply(lambda x: x[0])
        sampled["vader_compound"]   = vader_results.apply(lambda x: x[1])

        # --- TextBlob predictions ---
        print("  Generating TextBlob predictions...")
        tb_results = sampled[self.text_col].apply(self.predict_textblob)
        sampled["textblob_prediction"] = tb_results.apply(lambda x: x[0])
        sampled["textblob_polarity"]   = tb_results.apply(lambda x: x[1])

        self.results_df = sampled
        print("\n  Model building complete!")
        return sampled

    # ------------------------------------------------------------------ #
    #  Step 7a – Single-model evaluation
    # ------------------------------------------------------------------ #
    def evaluate_model(
        self,
        true_labels: pd.Series,
        pred_labels: pd.Series,
        model_name: str,
    ) -> dict:
        """
        Evaluate a single model and save its confusion matrix.

        Reports both **macro** and **weighted** averages:
        - Macro:    treats every class equally — fair on imbalanced data.
        - Weighted: accounts for class frequency — reflects overall volume.

        Also produces per-class classification report and confusion matrix.
        """
        print("\n" + "=" * 60)
        print(f"EVALUATION: {model_name}")
        print("=" * 60)

        acc = accuracy_score(true_labels, pred_labels)

        # --- Macro averages (primary – fair across imbalanced classes) ---
        prec_macro = precision_score(true_labels, pred_labels, average="macro", zero_division=0)
        rec_macro  = recall_score(true_labels, pred_labels, average="macro", zero_division=0)
        f1_macro   = f1_score(true_labels, pred_labels, average="macro", zero_division=0)

        # --- Weighted averages (secondary – reflects dataset distribution) ---
        prec_wt = precision_score(true_labels, pred_labels, average="weighted", zero_division=0)
        rec_wt  = recall_score(true_labels, pred_labels, average="weighted", zero_division=0)
        f1_wt   = f1_score(true_labels, pred_labels, average="weighted", zero_division=0)

        cm = confusion_matrix(true_labels, pred_labels, labels=LABELS)
        report = classification_report(
            true_labels, pred_labels, labels=LABELS, zero_division=0
        )

        print(f"\n  Accuracy         : {acc:.4f}")
        print(f"\n  --- Macro (primary – treats each class equally) ---")
        print(f"  Precision (macro): {prec_macro:.4f}")
        print(f"  Recall    (macro): {rec_macro:.4f}")
        print(f"  F1-Score  (macro): {f1_macro:.4f}")
        print(f"\n  --- Weighted (secondary – accounts for class size) ---")
        print(f"  Precision (weighted): {prec_wt:.4f}")
        print(f"  Recall    (weighted): {rec_wt:.4f}")
        print(f"  F1-Score  (weighted): {f1_wt:.4f}")
        print(f"\n  Confusion Matrix:\n{cm}")
        print(f"\n  Classification Report:\n{report}")

        # ---- Confusion matrix heatmap ----
        plt.figure(figsize=(7, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=LABELS, yticklabels=LABELS)
        plt.title(f"Confusion Matrix – {model_name}")
        plt.ylabel("True Label"); plt.xlabel("Predicted Label")
        fname = f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
        plt.tight_layout(); plt.savefig(fname, dpi=300, bbox_inches="tight"); plt.close()
        print(f"  Saved: {fname}")

        return {
            "accuracy": acc,
            "precision_macro": prec_macro, "recall_macro": rec_macro, "f1_macro": f1_macro,
            "precision_wt": prec_wt, "recall_wt": rec_wt, "f1_wt": f1_wt,
            "confusion_matrix": cm, "classification_report": report,
        }

    # ------------------------------------------------------------------ #
    #  Step 7b – Compare both models
    # ------------------------------------------------------------------ #
    def compare_models(self) -> dict:
        """
        Compare VADER and TextBlob:
        * Side-by-side metrics table
        * Bar chart of metrics
        * Score distribution histograms
        * Agreement pie chart
        * Written conclusion
        """
        if not hasattr(self, "results_df"):
            raise ValueError("Call build_models() first.")

        print("\n" + "=" * 60)
        print("MODEL COMPARISON")
        print("=" * 60)

        rdf = self.results_df

        vader_res = self.evaluate_model(
            rdf[self.label_col], rdf["vader_prediction"], "VADER"
        )
        tb_res = self.evaluate_model(
            rdf[self.label_col], rdf["textblob_prediction"], "TextBlob"
        )

        # ---- Comparison table (both macro & weighted) ----
        rows = [
            "Accuracy",
            "Precision (macro)",  "Recall (macro)",  "F1-Score (macro)",
            "Precision (weighted)", "Recall (weighted)", "F1-Score (weighted)",
        ]
        v_vals = [
            vader_res["accuracy"],
            vader_res["precision_macro"], vader_res["recall_macro"], vader_res["f1_macro"],
            vader_res["precision_wt"],    vader_res["recall_wt"],    vader_res["f1_wt"],
        ]
        t_vals = [
            tb_res["accuracy"],
            tb_res["precision_macro"], tb_res["recall_macro"], tb_res["f1_macro"],
            tb_res["precision_wt"],    tb_res["recall_wt"],    tb_res["f1_wt"],
        ]
        cmp_df = pd.DataFrame({
            "Metric": rows, "VADER": v_vals, "TextBlob": t_vals,
            "Difference (V-T)": [v - t for v, t in zip(v_vals, t_vals)],
        })

        print("\n" + "=" * 60)
        print("COMPARISON TABLE")
        print("=" * 60)
        print(cmp_df.to_string(index=False))
        cmp_df.to_csv("model_comparison_table.csv", index=False)
        print("  Saved: model_comparison_table.csv")

        # ---- Visualisation ------------------------------------------- #
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # (a) Metrics bar chart (macro – primary metrics for imbalanced data)
        chart_labels = ["Acc", "Prec\n(macro)", "Rec\n(macro)", "F1\n(macro)",
                        "Prec\n(wt)", "Rec\n(wt)", "F1\n(wt)"]
        x = np.arange(len(chart_labels)); w = 0.35
        axes[0, 0].bar(x - w / 2, v_vals, w, label="VADER", color="steelblue")
        axes[0, 0].bar(x + w / 2, t_vals, w, label="TextBlob", color="coral")
        axes[0, 0].set_xticks(x); axes[0, 0].set_xticklabels(chart_labels, rotation=0, fontsize=8)
        axes[0, 0].set_ylabel("Score"); axes[0, 0].set_title("Performance Comparison")
        axes[0, 0].legend(); axes[0, 0].grid(axis="y", alpha=0.3)
        axes[0, 0].set_ylim(0, 1)

        # (b) VADER compound score distribution
        if "vader_compound" in rdf.columns:
            for lbl, colour in zip(LABELS, ["green", "grey", "red"]):
                subset = rdf[rdf[self.label_col] == lbl]["vader_compound"]
                axes[0, 1].hist(subset, bins=30, alpha=0.5, label=lbl, color=colour)
            axes[0, 1].set_xlabel("VADER Compound Score")
            axes[0, 1].set_ylabel("Count")
            axes[0, 1].set_title("VADER Score Distribution by True Label")
            axes[0, 1].legend()

        # (c) TextBlob polarity distribution
        if "textblob_polarity" in rdf.columns:
            for lbl, colour in zip(LABELS, ["green", "grey", "red"]):
                subset = rdf[rdf[self.label_col] == lbl]["textblob_polarity"]
                axes[1, 0].hist(subset, bins=30, alpha=0.5, label=lbl, color=colour)
            axes[1, 0].set_xlabel("TextBlob Polarity")
            axes[1, 0].set_ylabel("Count")
            axes[1, 0].set_title("TextBlob Polarity Distribution by True Label")
            axes[1, 0].legend()

        # (d) Agreement pie chart
        agree = int((rdf["vader_prediction"] == rdf["textblob_prediction"]).sum())
        disagree = len(rdf) - agree
        axes[1, 1].pie(
            [agree, disagree],
            labels=[f"Agree ({agree})", f"Disagree ({disagree})"],
            autopct="%1.1f%%", colors=["lightgreen", "lightcoral"],
        )
        axes[1, 1].set_title("VADER vs TextBlob Agreement")

        plt.tight_layout()
        plt.savefig("model_comparison.png", dpi=300, bbox_inches="tight"); plt.close()
        print("  Saved: model_comparison.png")

        # ---- Written conclusion -------------------------------------- #
        conclusion = self._generate_conclusion(vader_res, tb_res, agree, len(rdf))
        print(conclusion)

        # ---- Error analysis (misclassified examples) ----------------- #
        error_report = self._error_analysis(rdf)

        # Save conclusion + error analysis to file
        with open("model_comparison_conclusion.txt", "w", encoding="utf-8") as fh:
            fh.write(cmp_df.to_string(index=False))
            fh.write("\n\n")
            fh.write(conclusion)
            fh.write("\n\n")
            fh.write(error_report)
        print("  Saved: model_comparison_conclusion.txt")

        return {"vader": vader_res, "textblob": tb_res, "comparison_table": cmp_df}

    # ------------------------------------------------------------------ #
    #  Error analysis – misclassified examples
    # ------------------------------------------------------------------ #
    def _error_analysis(self, rdf: pd.DataFrame, n_examples: int = 5) -> str:
        """
        Inspect misclassified reviews to understand *why* models fail.

        For each model, shows up to ``n_examples`` misclassified reviews
        grouped by error type (e.g. Positive→Neutral, Negative→Positive).
        This helps identify patterns such as sarcasm, mixed sentiment,
        or very short/ambiguous reviews that mislead the lexicons.

        Args:
            rdf:        Results DataFrame with predictions and true labels.
            n_examples: Number of example texts to show per error type.

        Returns:
            Formatted error-analysis report string.
        """
        print("\n" + "=" * 60)
        print("ERROR ANALYSIS – MISCLASSIFIED EXAMPLES")
        print("=" * 60)

        lines = [
            "=" * 60,
            "ERROR ANALYSIS – MISCLASSIFIED EXAMPLES",
            "=" * 60,
        ]

        text_col = self.text_col

        for model_name, pred_col in [("VADER", "vader_prediction"),
                                     ("TextBlob", "textblob_prediction")]:
            misclassified = rdf[rdf[self.label_col] != rdf[pred_col]]
            lines.append(f"\n--- {model_name} ---")
            lines.append(f"Total misclassified: {len(misclassified)} / {len(rdf)} "
                         f"({len(misclassified)/len(rdf)*100:.1f}%)")

            if misclassified.empty:
                lines.append("  No misclassifications found.")
                continue

            # Group by (true → predicted) error type
            error_types = (
                misclassified
                .groupby([self.label_col, pred_col])
                .size()
                .sort_values(ascending=False)
            )

            lines.append("\n  Error breakdown (True → Predicted : count):")
            for (true_lbl, pred_lbl), count in error_types.items():
                lines.append(f"    {true_lbl:>8s} → {pred_lbl:<8s} : {count}")

                # Show sample misclassified texts for this error type
                samples = misclassified[
                    (misclassified[self.label_col] == true_lbl) &
                    (misclassified[pred_col] == pred_lbl)
                ].head(n_examples)

                for _, row in samples.iterrows():
                    txt = str(row[text_col])[:120]
                    lines.append(f"      • \"{txt}...\"")

            lines.append("")

        # Common failure patterns
        lines.append("=" * 60)
        lines.append("COMMON FAILURE PATTERNS")
        lines.append("=" * 60)
        lines.append("  • Sarcasm / irony — lexicons take words at face value.")
        lines.append("  • Mixed sentiment — 'Great product but shipping was awful.'")
        lines.append("  • Short / vague reviews — too little text for reliable scoring.")
        lines.append("  • Neutral reviews misclassified — hardest class for both models")
        lines.append("    because neutral language overlaps with weak positive/negative.")
        lines.append("")

        report = "\n".join(lines)
        print(report)
        return report

    # ------------------------------------------------------------------ #
    #  Conclusion helper
    # ------------------------------------------------------------------ #
    @staticmethod
    def _generate_conclusion(v: dict, t: dict, agree: int, total: int) -> str:
        """Produce a brief analytical conclusion comparing both models."""
        better_macro = "VADER" if v["f1_macro"] >= t["f1_macro"] else "TextBlob"
        better_wt    = "VADER" if v["f1_wt"]    >= t["f1_wt"]    else "TextBlob"
        lines = [
            "\n" + "=" * 60,
            "CONCLUSION",
            "=" * 60,
            "",
            "Primary metric — Macro F1 (treats each class equally, fair for imbalanced data):",
            f"  • VADER   : {v['f1_macro']:.4f}",
            f"  • TextBlob: {t['f1_macro']:.4f}",
            f"  → **{better_macro}** performs better on macro F1.",
            "",
            "Secondary metric — Weighted F1 (accounts for class frequency):",
            f"  • VADER   : {v['f1_wt']:.4f}",
            f"  • TextBlob: {t['f1_wt']:.4f}",
            f"  → **{better_wt}** performs better on weighted F1.",
            "",
            f"Accuracy — VADER: {v['accuracy']:.4f}, TextBlob: {t['accuracy']:.4f}",
            f"Agreement — {agree}/{total} predictions ({agree/total*100:.1f}%).",
            "",
            "Note: Amazon reviews are heavily skewed toward Positive ratings.",
            "Macro averaging is the more informative metric because it",
            "exposes how well each model handles the minority classes",
            "(Neutral and Negative), rather than inflating scores via",
            "the dominant Positive class.",
            "",
            "VADER tends to perform better on informal, short reviews with",
            "exclamation marks and capitalisation, while TextBlob may be",
            "more conservative, classifying more reviews as Neutral.",
            "",
            "Both lexicon-based models are limited by their inability to",
            "learn from data.  Phase #2 will explore supervised machine-learning",
            "models that can be trained on the labelled dataset for improved",
            "performance.",
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    #  Persistence
    # ------------------------------------------------------------------ #
    def save_results(self, output_file: str = "sentiment_analysis_results.csv"):
        """Save the full results DataFrame (with predictions) to CSV."""
        if not hasattr(self, "results_df"):
            raise ValueError("Call build_models() first.")
        self.results_df.to_csv(output_file, index=False)
        print(f"\n  Results saved to: {output_file}")


if __name__ == "__main__":
    pass
