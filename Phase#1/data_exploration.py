"""
Phase #1 - Step 1: Dataset Data Exploration
Team #4: Industrial and Scientific Dataset

This module performs comprehensive data exploration as required by Phase #1:
    (a) Counts, averages
    (b) Distribution of the number of reviews across products
    (c) Distribution of the number of reviews per product
    (d) Distribution of reviews per user
    (e) Review lengths and outliers
    (f) Analyze lengths
    (g) Check for duplicates

Additional analyses:
    - Missing values
    - Rating distribution & correlation with review length
    - Word cloud visualisation
    - Summary-field analysis

Reference:
    Ni, J., Li, J., & McAuley, J. (2019). Justifying recommendations using
    distantly-labeled reviews and fine-grained aspects.  EMNLP-IJCNLP 2019.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")                       # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import json
import os


class DataExplorer:
    """Performs comprehensive data exploration on Amazon review datasets."""

    # ------------------------------------------------------------------ #
    #  Constructor & data loading
    # ------------------------------------------------------------------ #
    def __init__(self, data_path: str = None, df: pd.DataFrame = None):
        """
        Initialise the explorer.

        Args:
            data_path: Path to a JSON / JSON-GZ review file.
            df:        Pre-loaded DataFrame (takes priority over data_path).

        Raises:
            ValueError: If neither argument is provided.
        """
        if df is not None:
            self.df = df.copy()
        elif data_path:
            self.df = self._load_data(data_path)
        else:
            raise ValueError("Either data_path or df must be provided.")

        print(f"Dataset loaded: {len(self.df)} reviews")
        print(f"Columns: {list(self.df.columns)}")

    @staticmethod
    def _load_data(file_path: str) -> pd.DataFrame:
        """Load Amazon review data from a line-delimited JSON (.json / .json.gz)."""
        import gzip
        print(f"Loading data from {file_path}...")
        data = []
        opener = gzip.open if file_path.endswith(".gz") else open
        with opener(file_path, "rt", encoding="utf-8") as fh:
            for line in fh:
                try:
                    data.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue
        return pd.DataFrame(data)

    # ------------------------------------------------------------------ #
    #  1-a  Basic counts & averages
    # ------------------------------------------------------------------ #
    def basic_counts(self) -> dict:
        """Return basic counts, averages, data-type overview."""
        print("\n" + "=" * 60)
        print("BASIC COUNTS AND AVERAGES")
        print("=" * 60)

        res: dict = {
            "total_reviews": len(self.df),
            "total_columns": len(self.df.columns),
            "column_names": list(self.df.columns),
        }

        if "asin" in self.df.columns:
            res["unique_products"] = int(self.df["asin"].nunique())
        if "reviewerID" in self.df.columns:
            res["unique_users"] = int(self.df["reviewerID"].nunique())

        if "overall" in self.df.columns:
            res["average_rating"] = round(self.df["overall"].mean(), 4)
            res["median_rating"] = float(self.df["overall"].median())
            res["std_rating"] = round(self.df["overall"].std(), 4)
            res["rating_distribution"] = (
                self.df["overall"].value_counts().sort_index().to_dict()
            )

        if "reviewText" in self.df.columns:
            self.df["review_length"] = self.df["reviewText"].astype(str).apply(len)
            self.df["word_count"] = (
                self.df["reviewText"]
                .astype(str)
                .apply(lambda x: len(x.split()) if x != "nan" else 0)
            )
            res["avg_review_chars"] = round(self.df["review_length"].mean(), 2)
            res["median_review_chars"] = float(self.df["review_length"].median())
            res["avg_word_count"] = round(self.df["word_count"].mean(), 2)
            res["median_word_count"] = float(self.df["word_count"].median())

        if "summary" in self.df.columns:
            self.df["summary_length"] = self.df["summary"].astype(str).apply(len)
            res["avg_summary_chars"] = round(self.df["summary_length"].mean(), 2)

        # Data-type overview
        print("\nData Types:")
        print(self.df.dtypes.to_string())

        print("\nKey Statistics:")
        for k, v in res.items():
            if isinstance(v, dict):
                print(f"\n  {k}:")
                for rk, rv in sorted(v.items()):
                    print(f"    Rating {rk}: {rv} reviews")
            elif isinstance(v, list):
                print(f"  {k}: {v}")
            else:
                print(f"  {k}: {v}")

        return res

    # ------------------------------------------------------------------ #
    #  1-b/c  Reviews per product
    # ------------------------------------------------------------------ #
    def distribution_reviews_per_product(self) -> tuple:
        """Analyse and visualise the distribution of reviews per product."""
        print("\n" + "=" * 60)
        print("DISTRIBUTION OF REVIEWS PER PRODUCT")
        print("=" * 60)

        if "asin" not in self.df.columns:
            print("Warning: 'asin' column not found"); return None

        rpp = self.df.groupby("asin").size()
        stats = {
            "total_products": len(rpp),
            "min": int(rpp.min()), "max": int(rpp.max()),
            "mean": round(rpp.mean(), 2), "median": float(rpp.median()),
            "std": round(rpp.std(), 2),
            "products_1_review": int((rpp == 1).sum()),
            "products_5plus": int((rpp >= 5).sum()),
        }
        print("\nStatistics:")
        for k, v in stats.items():
            print(f"  {k}: {v}")

        top10 = rpp.nlargest(10)
        print("\nTop 10 Most Reviewed Products:")
        for asin, cnt in top10.items():
            print(f"  {asin}: {cnt} reviews")

        # --- visualisation ---
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        axes[0].hist(rpp, bins=50, edgecolor="black", color="steelblue")
        axes[0].set(xlabel="# Reviews", ylabel="# Products",
                    title="Distribution of Reviews per Product"); axes[0].set_yscale("log")
        rpp.value_counts().head(20).sort_index().plot(
            kind="bar", ax=axes[1], color="coral", edgecolor="black")
        axes[1].set(xlabel="# Reviews", ylabel="# Products",
                    title="Top 20 Review Counts per Product"); axes[1].tick_params(axis="x", rotation=45)
        axes[2].boxplot(rpp, vert=True)
        axes[2].set(ylabel="Reviews per Product", title="Box Plot – Reviews per Product")
        plt.tight_layout(); plt.savefig("reviews_per_product_distribution.png", dpi=300, bbox_inches="tight"); plt.close()
        print("Saved: reviews_per_product_distribution.png")
        return stats, rpp

    # ------------------------------------------------------------------ #
    #  1-d  Reviews per user
    # ------------------------------------------------------------------ #
    def distribution_reviews_per_user(self) -> tuple:
        """Analyse and visualise the distribution of reviews per user."""
        print("\n" + "=" * 60)
        print("DISTRIBUTION OF REVIEWS PER USER")
        print("=" * 60)

        if "reviewerID" not in self.df.columns:
            print("Warning: 'reviewerID' column not found"); return None

        rpu = self.df.groupby("reviewerID").size()
        stats = {
            "total_users": len(rpu),
            "min": int(rpu.min()), "max": int(rpu.max()),
            "mean": round(rpu.mean(), 2), "median": float(rpu.median()),
            "std": round(rpu.std(), 2),
            "users_1_review": int((rpu == 1).sum()),
            "users_10plus": int((rpu >= 10).sum()),
        }
        print("\nStatistics:")
        for k, v in stats.items():
            print(f"  {k}: {v}")

        top10 = rpu.nlargest(10)
        print("\nTop 10 Most Active Users:")
        for uid, cnt in top10.items():
            print(f"  {uid}: {cnt} reviews")

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        axes[0].hist(rpu, bins=50, edgecolor="black", color="teal")
        axes[0].set(xlabel="# Reviews", ylabel="# Users",
                    title="Distribution of Reviews per User"); axes[0].set_yscale("log")
        rpu.value_counts().head(20).sort_index().plot(
            kind="bar", ax=axes[1], color="salmon", edgecolor="black")
        axes[1].set(xlabel="# Reviews", ylabel="# Users",
                    title="Top 20 Review Counts per User"); axes[1].tick_params(axis="x", rotation=45)
        axes[2].boxplot(rpu, vert=True)
        axes[2].set(ylabel="Reviews per User", title="Box Plot – Reviews per User")
        plt.tight_layout(); plt.savefig("reviews_per_user_distribution.png", dpi=300, bbox_inches="tight"); plt.close()
        print("Saved: reviews_per_user_distribution.png")
        return stats, rpu

    # ------------------------------------------------------------------ #
    #  1-e/f  Review lengths & outliers
    # ------------------------------------------------------------------ #
    def analyze_review_lengths(self) -> tuple:
        """Analyse review lengths (words & chars) and flag outliers via IQR."""
        print("\n" + "=" * 60)
        print("REVIEW LENGTH ANALYSIS")
        print("=" * 60)

        if "reviewText" not in self.df.columns:
            print("Warning: 'reviewText' column not found"); return None

        if "word_count" not in self.df.columns:
            self.df["word_count"] = (
                self.df["reviewText"].astype(str)
                .apply(lambda x: len(x.split()) if x != "nan" else 0)
            )
        if "char_count" not in self.df.columns:
            self.df["char_count"] = self.df["reviewText"].astype(str).apply(len)

        wc = self.df["word_count"]
        stats = {
            "mean_words": round(wc.mean(), 2), "median_words": float(wc.median()),
            "std_words": round(wc.std(), 2),
            "min_words": int(wc.min()), "max_words": int(wc.max()),
            "mean_chars": round(self.df["char_count"].mean(), 2),
            "p25": float(wc.quantile(0.25)), "p75": float(wc.quantile(0.75)),
            "p95": float(wc.quantile(0.95)), "p99": float(wc.quantile(0.99)),
        }
        print("\nWord Count Statistics:")
        for k, v in stats.items():
            print(f"  {k}: {v}")

        # IQR outlier detection
        Q1, Q3 = wc.quantile(0.25), wc.quantile(0.75)
        IQR = Q3 - Q1
        lo, hi = max(0, Q1 - 1.5 * IQR), Q3 + 1.5 * IQR
        outliers = self.df[(wc < lo) | (wc > hi)]
        print(f"\nIQR Outlier Detection:")
        print(f"  Q1={Q1}, Q3={Q3}, IQR={IQR}")
        print(f"  Bounds: [{lo:.0f}, {hi:.0f}]")
        print(f"  Outliers: {len(outliers)} ({len(outliers)/len(self.df)*100:.2f}%)")

        # Visualisation
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes[0, 0].hist(wc, bins=50, edgecolor="black", color="steelblue")
        axes[0, 0].set(xlabel="Word Count", ylabel="Freq", title="Word Count Distribution")
        axes[0, 1].boxplot(wc, vert=True)
        axes[0, 1].set(ylabel="Word Count", title="Box Plot – Word Counts")
        axes[1, 0].hist(wc, bins=50, edgecolor="black", color="coral", log=True)
        axes[1, 0].set(xlabel="Word Count", ylabel="Freq (log)", title="Word Count (Log)")
        idx = self.df.sample(min(2000, len(self.df)), random_state=42).index
        axes[1, 1].scatter(self.df.loc[idx, "word_count"], self.df.loc[idx, "char_count"],
                           alpha=0.3, s=10, color="purple")
        axes[1, 1].set(xlabel="Word Count", ylabel="Char Count", title="Words vs Chars")
        plt.tight_layout(); plt.savefig("review_length_analysis.png", dpi=300, bbox_inches="tight"); plt.close()
        print("Saved: review_length_analysis.png")
        return stats, outliers

    # ------------------------------------------------------------------ #
    #  1-g  Duplicate detection
    # ------------------------------------------------------------------ #
    def check_duplicates(self) -> dict:
        """Detect duplicate reviews by text, (reviewer, product) pair, and full row."""
        print("\n" + "=" * 60)
        print("DUPLICATE DETECTION")
        print("=" * 60)
        info: dict = {}

        if "reviewText" in self.df.columns:
            dup = self.df.duplicated(subset=["reviewText"], keep=False)
            info["dup_text_rows"] = int(dup.sum())
            info["dup_text_unique"] = int(self.df[dup]["reviewText"].nunique()) if dup.any() else 0
            print(f"  Rows with duplicate text: {info['dup_text_rows']}")
            print(f"  Unique duplicate texts:   {info['dup_text_unique']}")

        if {"reviewerID", "asin"}.issubset(self.df.columns):
            dup_c = self.df.duplicated(subset=["reviewerID", "asin"], keep=False)
            info["dup_reviewer_product"] = int(dup_c.sum())
            print(f"  Duplicate reviewer+product pairs: {info['dup_reviewer_product']}")

        # Only check hashable columns for full-row duplicates (skip dict/list cols like 'style', 'image')
        hashable_cols = [c for c in self.df.columns
                         if self.df[c].dropna().apply(lambda x: isinstance(x, (str, int, float, bool))).all()]
        if hashable_cols:
            full = self.df.duplicated(subset=hashable_cols, keep=False)
            info["fully_duplicate_rows"] = int(full.sum())
        else:
            info["fully_duplicate_rows"] = 0
        print(f"  Fully duplicate rows: {info['fully_duplicate_rows']}")
        return info

    # ------------------------------------------------------------------ #
    #  Missing values
    # ------------------------------------------------------------------ #
    def missing_values_analysis(self) -> pd.DataFrame:
        """Analyse missing values per column."""
        print("\n" + "=" * 60)
        print("MISSING VALUES ANALYSIS")
        print("=" * 60)
        miss = self.df.isnull().sum()
        pct = round((miss / len(self.df)) * 100, 2)
        tbl = pd.DataFrame({"Missing": miss, "%": pct}).sort_values("Missing", ascending=False)
        print(tbl.to_string())
        return tbl

    # ------------------------------------------------------------------ #
    #  Rating distribution
    # ------------------------------------------------------------------ #
    def rating_distribution_analysis(self):
        """Analyse and visualise rating distribution (1-5)."""
        print("\n" + "=" * 60)
        print("RATING DISTRIBUTION ANALYSIS")
        print("=" * 60)
        if "overall" not in self.df.columns:
            print("Warning: 'overall' column not found"); return None
        rc = self.df["overall"].value_counts().sort_index()
        rp = round((rc / len(self.df)) * 100, 2)
        print("\nRating Distribution:")
        for r in sorted(rc.index):
            print(f"  Rating {r}: {rc[r]} reviews ({rp[r]:.2f}%)")

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        rc.plot(kind="bar", ax=axes[0], color="steelblue", edgecolor="black")
        axes[0].set(xlabel="Rating", ylabel="Count", title="Rating Counts"); axes[0].tick_params(axis="x", rotation=0)
        rp.plot(kind="bar", ax=axes[1], color="coral", edgecolor="black")
        axes[1].set(xlabel="Rating", ylabel="%", title="Rating %"); axes[1].tick_params(axis="x", rotation=0)
        axes[2].pie(rc, labels=[f"R{int(r)}" for r in rc.index], autopct="%1.1f%%",
                    startangle=90, colors=sns.color_palette("coolwarm", len(rc)))
        axes[2].set_title("Rating Proportion")
        plt.tight_layout(); plt.savefig("rating_distribution.png", dpi=300, bbox_inches="tight"); plt.close()
        print("Saved: rating_distribution.png")

        # --- Class imbalance analysis ---
        majority_rating = rc.idxmax()
        majority_pct = rp[majority_rating]
        print(f"\n  ⚠ DATASET IMBALANCE DETECTED:")
        print(f"    Rating {int(majority_rating)} dominates with {majority_pct:.1f}% of all reviews.")
        if majority_pct > 50:
            print(f"    This means the dataset is heavily skewed toward "
                  f"{'Positive' if majority_rating >= 4 else 'Negative' if majority_rating <= 2 else 'Neutral'} sentiment.")
            print(f"    Implication for modelling:")
            print(f"      • Accuracy alone is misleading (a naive classifier predicting")
            print(f"        the majority class would score ~{majority_pct:.0f}%).")
            print(f"      • Use macro-averaged metrics (precision, recall, F1) to")
            print(f"        evaluate model performance fairly across all classes.")
            print(f"      • Stratified sampling is recommended to preserve class ratios.")

        return rc, rp

    # ------------------------------------------------------------------ #
    #  Rating vs Length
    # ------------------------------------------------------------------ #
    def rating_vs_length_analysis(self):
        """Explore correlation between rating and review length."""
        print("\n" + "=" * 60)
        print("RATING vs REVIEW LENGTH")
        print("=" * 60)
        if "overall" not in self.df.columns or "reviewText" not in self.df.columns:
            return None
        if "word_count" not in self.df.columns:
            self.df["word_count"] = (
                self.df["reviewText"].astype(str)
                .apply(lambda x: len(x.split()) if x != "nan" else 0)
            )
        agg = self.df.groupby("overall")["word_count"].agg(["mean", "median", "std", "count"]).round(2)
        print(agg.to_string())

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        self.df.boxplot(column="word_count", by="overall", ax=axes[0])
        axes[0].set(xlabel="Rating", ylabel="Word Count", title="Review Length by Rating"); plt.suptitle("")
        agg["mean"].plot(kind="bar", ax=axes[1], color="teal", edgecolor="black")
        axes[1].set(xlabel="Rating", ylabel="Avg Word Count", title="Avg Length by Rating")
        axes[1].tick_params(axis="x", rotation=0)
        plt.tight_layout(); plt.savefig("rating_vs_length.png", dpi=300, bbox_inches="tight"); plt.close()
        print("Saved: rating_vs_length.png")
        return agg

    # ------------------------------------------------------------------ #
    #  Word cloud
    # ------------------------------------------------------------------ #
    def generate_word_cloud(self):
        """Generate a word cloud from review texts."""
        print("\n" + "=" * 60)
        print("WORD CLOUD")
        print("=" * 60)
        if "reviewText" not in self.df.columns:
            return
        text = " ".join(self.df["reviewText"].dropna().astype(str))
        wc = WordCloud(width=1200, height=600, background_color="white",
                       max_words=200, collocations=False, random_state=42).generate(text)
        plt.figure(figsize=(14, 7)); plt.imshow(wc, interpolation="bilinear")
        plt.axis("off"); plt.title("Word Cloud of Review Texts", fontsize=16)
        plt.tight_layout(); plt.savefig("word_cloud.png", dpi=300, bbox_inches="tight"); plt.close()
        print("Saved: word_cloud.png")

    # ------------------------------------------------------------------ #
    #  Summary field analysis
    # ------------------------------------------------------------------ #
    def summary_field_analysis(self) -> dict:
        """Analyse the 'summary' column length and content."""
        print("\n" + "=" * 60)
        print("SUMMARY FIELD ANALYSIS")
        print("=" * 60)
        if "summary" not in self.df.columns:
            print("Warning: 'summary' column not found"); return None
        self.df["summary_word_count"] = (
            self.df["summary"].astype(str)
            .apply(lambda x: len(x.split()) if x != "nan" else 0)
        )
        stats = {
            "mean_words": round(self.df["summary_word_count"].mean(), 2),
            "median_words": float(self.df["summary_word_count"].median()),
            "max_words": int(self.df["summary_word_count"].max()),
            "min_words": int(self.df["summary_word_count"].min()),
            "empty": int((self.df["summary_word_count"] == 0).sum()),
        }
        for k, v in stats.items():
            print(f"  {k}: {v}")
        return stats

    # ------------------------------------------------------------------ #
    #  Orchestrator
    # ------------------------------------------------------------------ #
    def run_full_exploration(self, save_results: bool = True) -> dict:
        """
        Execute every exploration step and optionally persist to disk.

        Returns:
            dict: All exploration results keyed by analysis name.
        """
        print("\n" + "=" * 80)
        print("COMPREHENSIVE DATA EXPLORATION")
        print("=" * 80)

        results = {}
        results["basic_counts"]         = self.basic_counts()
        results["missing_values"]       = self.missing_values_analysis()
        results["rating_distribution"]  = self.rating_distribution_analysis()
        results["reviews_per_product"]  = self.distribution_reviews_per_product()
        results["reviews_per_user"]     = self.distribution_reviews_per_user()
        results["review_lengths"]       = self.analyze_review_lengths()
        results["rating_vs_length"]     = self.rating_vs_length_analysis()
        results["summary_analysis"]     = self.summary_field_analysis()
        results["duplicates"]           = self.check_duplicates()
        self.generate_word_cloud()

        if save_results:
            with open("data_exploration_summary.txt", "w", encoding="utf-8") as fh:
                fh.write("DATA EXPLORATION SUMMARY\n")
                fh.write("Team #4: Industrial and Scientific Dataset\n")
                fh.write("=" * 80 + "\n\n")
                for key, val in results.items():
                    fh.write(f"\n{'='*60}\n{key.upper()}\n{'='*60}\n")
                    if isinstance(val, dict):
                        for k, v in val.items():
                            fh.write(f"  {k}: {v}\n")
                    elif isinstance(val, tuple):
                        for item in val:
                            fh.write(f"  {item}\n")
                    elif isinstance(val, pd.DataFrame):
                        fh.write(val.to_string() + "\n")
                    else:
                        fh.write(f"  {val}\n")
                    fh.write("\n")
            print("\nSaved: data_exploration_summary.txt")

        print("\n" + "=" * 80)
        print("EXPLORATION COMPLETE")
        print("=" * 80)
        return results


if __name__ == "__main__":
    pass
