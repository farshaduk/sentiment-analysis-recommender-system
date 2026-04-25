"""
Phase #1 - Step 2: Text Basic Pre-processing
Team #4: Industrial and Scientific Dataset

This module handles all pre-processing required before building sentiment
analysis models:
    (a) Label data based on ratings  → Positive / Neutral / Negative
    (b) Select appropriate columns with justification
    (c) Detect and (optionally) remove outliers
    (d) Basic text cleaning  (nulls, HTML, URLs, special chars, whitespace)
    (e) Create helper columns  (review length, word count)

Design Note:
    Minimal text preprocessing is applied at this stage because VADER is
    designed to work with raw text (capitalisation, punctuation, and
    modifiers carry sentiment signals).  TextBlob also works well with
    lightly cleaned text.  Heavy preprocessing (stop-word removal,
    stemming) is intentionally avoided so that lexicon tools can exploit
    full linguistic cues.
"""

import pandas as pd
import numpy as np
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict, List


class TextPreprocessor:
    """
    Applies pre-processing steps required for sentiment analysis.

    Attributes:
        df (pd.DataFrame):          Working copy of the dataset.
        preprocessing_steps (list): Log of every step applied.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Args:
            df: Raw Amazon review DataFrame.
        """
        self.df = df.copy()
        self.preprocessing_steps: list = []

    # ------------------------------------------------------------------ #
    #  Step 2-b  Column selection
    # ------------------------------------------------------------------ #
    def select_columns(
        self,
        columns: List[str],
        justification: Dict[str, str] = None,
    ) -> pd.DataFrame:
        """
        Keep only the columns relevant to sentiment analysis.

        Args:
            columns:       List of column names to retain.
            justification: {column_name: reason_string} explaining each
                           choice (printed and logged for the report).

        Returns:
            The filtered DataFrame.
        """
        print("\n" + "=" * 60)
        print("COLUMN SELECTION")
        print("=" * 60)

        available = list(self.df.columns)
        selected = [c for c in columns if c in available]
        missing  = [c for c in columns if c not in available]

        if missing:
            print(f"  Warning – columns not available: {missing}")
            print(f"  Available columns: {available}")

        # Always keep product & user IDs when present
        essential = [c for c in ("asin", "reviewerID") if c in available]
        keep = list(dict.fromkeys(selected + essential))  # unique, ordered
        self.df = self.df[keep]

        print(f"\n  Selected columns: {keep}")

        if justification:
            print("\n  Column Justification:")
            for col, reason in justification.items():
                if col in keep:
                    print(f"    • {col}: {reason}")

        self.preprocessing_steps.append({
            "step": "column_selection",
            "columns": keep,
            "justification": justification,
        })
        return self.df

    # ------------------------------------------------------------------ #
    #  Step 2-a  Sentiment labelling
    # ------------------------------------------------------------------ #
    def label_sentiment(self, rating_col: str = "overall") -> pd.DataFrame:
        """
        Create a ``sentiment_label`` column based on the star rating:

        * Ratings 4, 5 → **Positive**
        * Rating  3     → **Neutral**
        * Ratings 1, 2 → **Negative**

        Args:
            rating_col: Name of the rating column.

        Returns:
            DataFrame with the new ``sentiment_label`` column.
        """
        print("\n" + "=" * 60)
        print("SENTIMENT LABELLING")
        print("=" * 60)

        if rating_col not in self.df.columns:
            raise ValueError(f"Rating column '{rating_col}' not found.")

        def _label(r):
            if pd.isna(r):
                return "Unknown"
            if r >= 4:
                return "Positive"
            if r == 3:
                return "Neutral"
            return "Negative"

        self.df["sentiment_label"] = self.df[rating_col].apply(_label)

        counts = self.df["sentiment_label"].value_counts()
        print("\n  Sentiment Label Distribution:")
        for lbl, cnt in counts.items():
            print(f"    {lbl}: {cnt} ({cnt / len(self.df) * 100:.2f}%)")

        # Visualisation
        fig, ax = plt.subplots(figsize=(7, 5))
        counts.plot(kind="bar", color=["green", "grey", "red", "blue"], edgecolor="black", ax=ax)
        ax.set_xlabel("Sentiment"); ax.set_ylabel("Count")
        ax.set_title("Sentiment Label Distribution"); ax.tick_params(axis="x", rotation=0)
        plt.tight_layout(); plt.savefig("sentiment_distribution.png", dpi=300, bbox_inches="tight"); plt.close()
        print("  Saved: sentiment_distribution.png")

        self.preprocessing_steps.append({
            "step": "sentiment_labelling",
            "mapping": {"Positive": "4-5", "Neutral": "3", "Negative": "1-2"},
            "distribution": counts.to_dict(),
        })
        return self.df

    # ------------------------------------------------------------------ #
    #  Step 2-c  Outlier detection
    # ------------------------------------------------------------------ #
    def detect_outliers(
        self,
        method: str = "iqr",
        columns: List[str] = None,
    ) -> dict:
        """
        Detect outliers in numeric columns.

        Args:
            method:  ``'iqr'`` (Inter-Quartile Range) or ``'zscore'``.
            columns: Columns to check; defaults to all numeric columns.

        Returns:
            dict: ``{column: {method, lower_bound, upper_bound, count, %}}``
        """
        print("\n" + "=" * 60)
        print("OUTLIER DETECTION")
        print("=" * 60)

        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns.tolist()

        info: dict = {}

        for col in columns:
            if col not in self.df.columns:
                continue

            if method == "iqr":
                q1 = self.df[col].quantile(0.25)
                q3 = self.df[col].quantile(0.75)
                iqr = q3 - q1
                lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                out = self.df[(self.df[col] < lo) | (self.df[col] > hi)]
                info[col] = {
                    "method": "IQR", "lower": lo, "upper": hi,
                    "count": len(out), "%": round(len(out) / len(self.df) * 100, 2),
                }
                print(f"\n  {col}:  bounds=[{lo:.2f}, {hi:.2f}]  "
                      f"outliers={len(out)} ({info[col]['%']}%)")

            elif method == "zscore":
                z = np.abs((self.df[col] - self.df[col].mean()) / self.df[col].std())
                out = self.df[z > 3]
                info[col] = {
                    "method": "Z-Score", "threshold": 3,
                    "count": len(out), "%": round(len(out) / len(self.df) * 100, 2),
                }
                print(f"\n  {col}:  z>3  outliers={len(out)} ({info[col]['%']}%)")

        self.preprocessing_steps.append({
            "step": "outlier_detection", "method": method, "info": info,
        })
        return info

    # ------------------------------------------------------------------ #
    #  Outlier removal (optional)
    # ------------------------------------------------------------------ #
    def remove_outliers(self, column: str, method: str = "iqr") -> pd.DataFrame:
        """
        Remove outlier rows for a given column.

        Args:
            column: Column to trim.
            method: ``'iqr'`` or ``'zscore'``.

        Returns:
            Trimmed DataFrame.
        """
        if column not in self.df.columns:
            print(f"  Warning: '{column}' not found"); return self.df

        n0 = len(self.df)

        if method == "iqr":
            q1 = self.df[column].quantile(0.25)
            q3 = self.df[column].quantile(0.75)
            iqr = q3 - q1
            self.df = self.df[
                (self.df[column] >= q1 - 1.5 * iqr) &
                (self.df[column] <= q3 + 1.5 * iqr)
            ]
        elif method == "zscore":
            z = np.abs((self.df[column] - self.df[column].mean()) / self.df[column].std())
            self.df = self.df[z <= 3]

        removed = n0 - len(self.df)
        print(f"  Removed {removed} outliers from '{column}' ({removed / n0 * 100:.2f}%)")
        self.preprocessing_steps.append({
            "step": "outlier_removal", "column": column,
            "method": method, "removed": removed,
        })
        return self.df

    # ------------------------------------------------------------------ #
    #  Step 4 – Basic text cleaning
    # ------------------------------------------------------------------ #
    def clean_text_basic(self, text_col: str = "reviewText") -> pd.DataFrame:
        """
        Apply minimal, justified text cleaning.

        Steps applied:
            1. Drop rows with null / empty text.
            2. Convert to string.
            3. Remove HTML tags  (artefacts from web scraping).
            4. Remove URLs       (irrelevant to sentiment).
            5. Normalise whitespace.

        **Not** applied (and why):
            - Stop-word removal:  VADER uses these for context.
            - Lowercasing:        VADER treats CAPS as emphasis.
            - Stemming / lemma:   Both lexicons work on surface forms.

        Args:
            text_col: Column containing review text.

        Returns:
            Cleaned DataFrame.
        """
        if text_col not in self.df.columns:
            print(f"  Warning: '{text_col}' not found"); return self.df

        print("\n" + "=" * 60)
        print("BASIC TEXT CLEANING")
        print("=" * 60)

        n0 = len(self.df)

        # 1. Drop nulls
        self.df = self.df[self.df[text_col].notna()]
        null_removed = n0 - len(self.df)
        if null_removed:
            print(f"  Removed {null_removed} rows with null text")

        # 2. String conversion
        self.df[text_col] = self.df[text_col].astype(str)

        # 3. Remove HTML tags
        self.df[text_col] = self.df[text_col].apply(lambda t: re.sub(r"<[^>]+>", " ", t))

        # 4. Remove URLs
        self.df[text_col] = self.df[text_col].apply(
            lambda t: re.sub(r"http\S+|www\.\S+", " ", t)
        )

        # 5. Normalise whitespace
        self.df[text_col] = self.df[text_col].apply(lambda t: re.sub(r"\s+", " ", t).strip())

        # Drop empty strings
        self.df = self.df[self.df[text_col].str.strip() != ""]

        # Add helper columns
        self.df["review_word_count"] = self.df[text_col].apply(lambda t: len(t.split()))
        self.df["review_char_count"] = self.df[text_col].apply(len)

        print(f"  Final dataset size: {len(self.df)} reviews")
        print("\n  Preprocessing justification:")
        print("    • HTML removal   – web scraping artefacts add noise")
        print("    • URL removal    – links carry no sentiment information")
        print("    • Whitespace fix – ensures consistent tokenisation")
        print("    • No lowercasing – VADER uses CAPS as emphasis signal")
        print("    • No stop-words  – both lexicons need context words")
        print("    • No stemming    – lexicons match on surface forms")

        self.preprocessing_steps.append({
            "step": "basic_text_cleaning",
            "null_removed": null_removed,
            "final_count": len(self.df),
        })
        return self.df

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #
    def get_preprocessing_summary(self) -> dict:
        """Return a log of all preprocessing steps applied."""
        return {
            "steps": self.preprocessing_steps,
            "final_size": len(self.df),
            "columns": list(self.df.columns),
        }

    def save_preprocessed_data(self, output_path: str):
        """Persist the preprocessed DataFrame to CSV."""
        self.df.to_csv(output_path, index=False)
        print(f"\n  Preprocessed data saved to: {output_path}")


if __name__ == "__main__":
    pass
