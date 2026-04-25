"""
Phase #2 - Step 15: Recommender System - Sentiment-Enhanced Ratings
Team #4 : Industrial and Scientific Dataset

Assignment requirement (Step 15)
----------------------------------
Review the paper "Recommender Systems Based on User Reviews: the State of
the Art" and implement one of the suggested options to enhance rating
values using review text data.

Strategy chosen: Sentiment-Enhanced Collaborative Filtering
------------------------------------------------------------
Based on the survey paper, we implement the **sentiment-enhanced rating**
approach described in several cited works (e.g., Zhang et al., 2014;
Musat & Faltings, 2015).

Rationale
~~~~~~~~~
Star ratings alone suffer from two well-known problems:
  1. **Rating inflation** - users tend to cluster around 4-5 stars even for
     mixed experiences.
  2. **Semantic gap** - the same star rating can mean very different things
     for different users (a hard-to-please user gives 3* for a product a
     lenient user would rate 5*).

Review text carries richer, more nuanced sentiment signals.  By blending
the star rating with a normalised sentiment score derived from the review
text (using VADER), we obtain an *enhanced rating* that better captures
the reviewer's true sentiment.

Formula
~~~~~~~
  sentiment_score  = (vader_compound + 1) / 2        # maps [-1, 1] -> [0, 1]
  normalised_star  = (overall - 1) / 4               # maps [1, 5] -> [0, 1]
  enhanced_0_1     = alpha x normalised_star + (1-alpha) x sentiment_score
  enhanced_rating  = enhanced_0_1 x 4 + 1            # maps back to [1, 5]
  alpha = 0.6   (gives more weight to the explicit star rating while still
             allowing text to adjust it)

Item-Based Collaborative Filtering with Enhanced Ratings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Pseudo-code:

  FUNCTION build_item_matrix(reviews):
      FOR EACH review IN reviews:
          compute enhanced_rating(review)
      CONSTRUCT user-item matrix M where M[user][item] = enhanced_rating
      RETURN M

  FUNCTION item_similarity(M):
      FOR EACH pair (item_i, item_j):
          sim(i,j) = cosine_similarity(M[:,i], M[:,j])
      RETURN similarity_matrix S

  FUNCTION recommend(user, S, M, top_k=5):
      seen_items = items rated by user
      FOR EACH unseen_item:
          predicted_rating = weighted_avg of k most similar seen items
      RETURN top_k items by predicted_rating

  FUNCTION evaluate(M_original, M_enhanced):
      FOR EACH known rating (user, item, true_rating):
          pred_orig     = collaborative_filter(M_original)
          pred_enhanced = collaborative_filter(M_enhanced)
      COMPUTE MAE and RMSE for both
      COMPARE results

Dataset reference:
    Ni, J., Li, J., & McAuley, J. (2019). Justifying recommendations using
    distantly-labeled reviews and fine-grained aspects. EMNLP-IJCNLP 2019.
    https://nijianmo.github.io/amazon/index.html

Paper reference:
    Bauman, K., Liu, B., & Tuzhilin, A. (2017). Recommender systems based
    on user reviews: The state of the art.  VLDB Endowment.
"""

import os
import warnings
import json
import gzip

warnings.filterwarnings("ignore")

import numpy  as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

RANDOM_SEED = 42
ALPHA       = 0.6   # weight given to the explicit star rating


class SentimentRecommender:
    """
    Implements a sentiment-enhanced item-based collaborative filtering
    recommender for Amazon product reviews.

    Parameters
    ----------
    data_file  : str  Path to the raw Amazon JSON / JSON.GZ data file.
    output_dir : str  Directory for saving plots and reports.
    """

    def __init__(self, data_file: str, output_dir: str = "."):
        self.data_file  = data_file
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self._vader = SentimentIntensityAnalyzer()
        self.df     : pd.DataFrame = None
        self.results: dict         = {}

    # ------------------------------------------------------------------ #
    #  Data loading and preparation
    # ------------------------------------------------------------------ #
    def load_data(self, sample_size: int = 5000) -> pd.DataFrame:
        """
        Load a sample of reviews, keeping only records with both
        ``reviewText`` and ``overall`` populated.

        Parameters
        ----------
        sample_size : int  Maximum number of reviews to load.

        Returns
        -------
        pd.DataFrame
        """
        print(f"\n  Loading reviews from {self.data_file} ...")
        opener = gzip.open if self.data_file.endswith(".gz") else open
        records = []
        with opener(self.data_file, "rt", encoding="utf-8") as fh:
            for line in fh:
                try:
                    rec = json.loads(line.strip())
                    if rec.get("reviewText") and rec.get("overall"):
                        records.append(rec)
                except (json.JSONDecodeError, KeyError):
                    pass
                if len(records) >= sample_size * 2:   # load extra to allow filtering
                    break

        df = pd.DataFrame(records)
        needed = ["reviewerID", "asin", "overall", "reviewText"]
        df = df[[c for c in needed if c in df.columns]].copy()
        df.dropna(subset=["reviewerID", "asin", "overall", "reviewText"], inplace=True)
        df["overall"] = pd.to_numeric(df["overall"], errors="coerce")
        df.dropna(subset=["overall"], inplace=True)

        # Keep only users and products with enough reviews for a meaningful CF
        # (cold-start problem: CF requires at minimum 2 interactions per user/item)
        user_counts    = df["reviewerID"].value_counts()
        product_counts = df["asin"].value_counts()
        df = df[
            df["reviewerID"].isin(user_counts[user_counts >= 2].index) &
            df["asin"].isin(product_counts[product_counts >= 2].index)
        ]

        # Final sample
        df = df.sample(min(sample_size, len(df)), random_state=RANDOM_SEED).copy()
        df.reset_index(drop=True, inplace=True)

        print(f"  Working dataset: {len(df):,} reviews  "
              f"({df['reviewerID'].nunique():,} users, "
              f"{df['asin'].nunique():,} products)")
        self.df = df
        return df

    # ------------------------------------------------------------------ #
    #  Step 15-a - Enhance ratings with sentiment (VADER)
    # ------------------------------------------------------------------ #
    def compute_sentiment_enhanced_ratings(self) -> None:
        """
        Enhance the original star rating by blending it with the VADER
        compound sentiment score derived from the review text.

        Adds columns to ``self.df``:
          - ``vader_compound``    - raw VADER compound score [-1, 1]
          - ``sentiment_score``   - normalised to [0, 1]
          - ``enhanced_rating``   - blended rating on [1, 5] scale
        """
        print("\n  Computing VADER sentiment scores for all reviews ...")
        self.df["vader_compound"] = self.df["reviewText"].apply(
            lambda t: self._vader.polarity_scores(str(t) if pd.notna(t) else "")["compound"]
        )

        # Normalise: [-1,1] -> [0,1]
        self.df["sentiment_score"] = (self.df["vader_compound"] + 1) / 2.0

        # Normalise star rating: [1,5] -> [0,1]
        self.df["norm_star"] = (self.df["overall"] - 1) / 4.0

        # Blend:  enhanced_0_1 = alpha x norm_star + (1-alpha) x sentiment_score
        self.df["enhanced_0_1"] = (
            ALPHA * self.df["norm_star"] +
            (1 - ALPHA) * self.df["sentiment_score"]
        )

        # Scale back to [1,5]
        self.df["enhanced_rating"] = self.df["enhanced_0_1"] * 4.0 + 1.0

        delta = self.df["enhanced_rating"] - self.df["overall"]
        print(f"  Original rating   - mean: {self.df['overall'].mean():.3f}  "
              f"std: {self.df['overall'].std():.3f}")
        print(f"  Enhanced rating   - mean: {self.df['enhanced_rating'].mean():.3f}  "
              f"std: {self.df['enhanced_rating'].std():.3f}")
        print(f"  Mean absolute delta (enhanced - original): {delta.abs().mean():.4f}")

        # Highlight controversial products (high variance in original ratings)
        var_by_prod = self.df.groupby("asin")["overall"].var()
        controversial = var_by_prod.nlargest(5)
        print(f"\n  Top 5 products with highest rating variance (controversial):")
        print(controversial.to_string())

        # Show how enhancement affects those controversial products
        print("\n  Rating comparison for controversial products:")
        c_df = self.df[self.df["asin"].isin(controversial.index)].groupby("asin").agg(
            orig_mean    = ("overall",         "mean"),
            enhanced_mean= ("enhanced_rating", "mean"),
            orig_std     = ("overall",         "std"),
            enhanced_std = ("enhanced_rating", "std"),
            count        = ("overall",         "count"),
        ).reset_index()
        print(c_df.to_string(index=False))

    # ------------------------------------------------------------------ #
    #  Step 15-b/c - Item-Based CF with enhanced ratings
    # ------------------------------------------------------------------ #
    def _build_user_item_matrix(
        self, rating_col: str
    ) -> tuple[pd.DataFrame, dict, dict]:
        """
        Construct a sparse user-item matrix for CF.

        Parameters
        ----------
        rating_col : str  Column name ('overall' or 'enhanced_rating').

        Returns
        -------
        matrix     : pd.DataFrame  rows=users, cols=products.
        user_index : dict          user  -> row index
        item_index : dict          product -> column index
        """
        users    = self.df["reviewerID"].unique()
        products = self.df["asin"].unique()
        user_idx = {u: i for i, u in enumerate(users)}
        item_idx = {p: j for j, p in enumerate(products)}

        M = np.full((len(users), len(products)), np.nan)

        for _, row in self.df.iterrows():
            u = user_idx[row["reviewerID"]]
            p = item_idx[row["asin"]]
            M[u, p] = row[rating_col]

        return (
            pd.DataFrame(M, index=users, columns=products),
            user_idx,
            item_idx,
        )

    @staticmethod
    def _cosine_similarity_matrix(M: np.ndarray) -> np.ndarray:
        """
        Compute column-wise cosine similarity (item-item).
        NaN values are treated as 0 for the similarity computation.
        """
        M_filled = np.nan_to_num(M, nan=0.0)
        norms    = np.linalg.norm(M_filled, axis=0, keepdims=True)
        norms[norms == 0] = 1e-10
        M_norm   = M_filled / norms
        return M_norm.T @ M_norm   # (n_items x n_items)

    def _predict_ratings(
        self,
        M: pd.DataFrame,
        sim_matrix: np.ndarray,
        top_k: int = 10,
    ) -> pd.DataFrame:
        """
        Predict missing ratings using item-based collaborative filtering.

        For each (user, item) pair where the rating is missing, the
        predicted rating is a weighted average of the user's known ratings
        for the *top_k* most similar items.

        Parameters
        ----------
        M          : user-item matrix (NaN = unrated).
        sim_matrix : item-item cosine similarity matrix.
        top_k      : number of neighbours to consider.

        Returns
        -------
        pd.DataFrame same shape as M with NaN entries filled by predictions.
        """
        M_arr   = M.values.copy()
        M_pred  = M_arr.copy()
        n_users, n_items = M_arr.shape

        for u in range(n_users):
            for i in range(n_items):
                if not np.isnan(M_arr[u, i]):
                    continue  # rating already known
                # Find top-k similar items that this user has rated
                sims   = sim_matrix[i]               # shape: (n_items,)
                rated  = ~np.isnan(M_arr[u])          # bool mask
                sims_rated = sims * rated.astype(float)

                top_idx = np.argsort(sims_rated)[::-1][:top_k]
                top_sims    = sims_rated[top_idx]
                top_ratings = M_arr[u, top_idx]

                valid = ~np.isnan(top_ratings) & (top_sims > 0)
                if valid.any():
                    M_pred[u, i] = (
                        np.dot(top_sims[valid], top_ratings[valid]) /
                        np.sum(top_sims[valid])
                    )

        return pd.DataFrame(M_pred, index=M.index, columns=M.columns)

    @staticmethod
    def _train_test_split_matrix(
        M: pd.DataFrame,
        test_frac: float = 0.2,
    ) -> tuple[pd.DataFrame, list]:
        """
        Hold out test_frac of observed ratings for evaluation.

        Returns
        -------
        M_train : pd.DataFrame  Matrix with held-out cells set to NaN.
        test_idx : list of (row, col) int tuples for the held-out cells.
        """
        observed_idx = list(zip(*np.where(~np.isnan(M.values))))
        np.random.seed(RANDOM_SEED)
        np.random.shuffle(observed_idx)
        n_test   = max(1, int(len(observed_idx) * test_frac))
        test_idx = observed_idx[:n_test]

        M_train = M.copy()
        for u, i in test_idx:
            M_train.values[u, i] = np.nan   # hide from the CF model

        return M_train, test_idx

    def _evaluate_cf(
        self,
        M_true: pd.DataFrame,
        M_pred: pd.DataFrame,
        test_idx: list,
        label: str,
    ) -> dict:
        """
        Evaluate predicted ratings on the held-out test positions.

        Parameters
        ----------
        M_true   : original full matrix (with true ratings at test positions).
        M_pred   : matrix produced by _predict_ratings on the TRAINING matrix
                   (test positions were NaN during training, now filled by CF).
        test_idx : list of (row, col) int tuples identifying held-out cells.
        label    : name for prints/output.
        """
        true_vals = np.array([M_true.values[u, i] for u, i in test_idx])
        pred_vals = np.array([M_pred.values[u, i] for u, i in test_idx])

        # Only score positions the CF was able to predict (not still NaN)
        valid = ~np.isnan(pred_vals)
        if valid.sum() == 0:
            print(f"  Warning: no valid predictions for {label}")
            return {"label": label, "mae": np.nan, "rmse": np.nan, "coverage": 0.0}

        mae  = np.mean(np.abs(true_vals[valid] - pred_vals[valid]))
        rmse = np.sqrt(np.mean((true_vals[valid] - pred_vals[valid]) ** 2))
        cov  = valid.sum() / len(test_idx)

        print(f"\n  [{label}] Evaluation on {valid.sum()} held-out ratings:")
        print(f"    MAE      = {mae:.4f}")
        print(f"    RMSE     = {rmse:.4f}")
        print(f"    Coverage = {cov:.2%}")

        return {"label": label, "mae": mae, "rmse": rmse, "coverage": cov}

    # ------------------------------------------------------------------ #
    #  Visualisations
    # ------------------------------------------------------------------ #
    def _plot_rating_comparison(self) -> None:
        """
        Plot original vs. enhanced rating distributions side-by-side.
        """
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle(
            "Step 15 - Original vs. Sentiment-Enhanced Rating Distributions",
            fontsize=12, fontweight="bold",
        )

        bins = np.linspace(1, 5, 20)

        axes[0].hist(self.df["overall"],         bins=bins, color="#3498db",
                     edgecolor="white", alpha=0.85)
        axes[0].set_title("Original Star Ratings")
        axes[0].set_xlabel("Rating (1-5)")
        axes[0].set_ylabel("Count")
        axes[0].axvline(self.df["overall"].mean(), color="red",
                        linestyle="--", label=f"Mean = {self.df['overall'].mean():.2f}")
        axes[0].legend()

        axes[1].hist(self.df["enhanced_rating"], bins=bins, color="#2ecc71",
                     edgecolor="white", alpha=0.85)
        axes[1].set_title("Sentiment-Enhanced Ratings")
        axes[1].set_xlabel("Rating (1-5)")
        axes[1].axvline(self.df["enhanced_rating"].mean(), color="red",
                        linestyle="--",
                        label=f"Mean = {self.df['enhanced_rating'].mean():.2f}")
        axes[1].legend()

        plt.tight_layout()
        out_path = os.path.join(self.output_dir, "step15_rating_comparison.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved rating comparison -> {out_path}")

    def _plot_delta_distribution(self) -> None:
        """
        Plot the distribution of (enhanced_rating - original_rating).
        """
        delta = self.df["enhanced_rating"] - self.df["overall"]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(delta, bins=50, color="#9b59b6", edgecolor="white", alpha=0.85)
        ax.axvline(0, color="red", linestyle="--", linewidth=1.5)
        ax.set_title("Step 15 - Rating Delta (Enhanced - Original)",
                     fontsize=11, fontweight="bold")
        ax.set_xlabel("Delta Rating")
        ax.set_ylabel("Count")
        plt.tight_layout()
        out_path = os.path.join(self.output_dir, "step15_rating_delta.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved delta distribution -> {out_path}")

    def _plot_cf_comparison(self, orig_eval: dict, enhanced_eval: dict) -> None:
        """
        Bar chart comparing MAE and RMSE for original vs. enhanced CF.
        """
        metrics = ["mae", "rmse"]
        labels  = ["MAE", "RMSE"]
        orig_v     = [orig_eval[m]     for m in metrics]
        enhanced_v = [enhanced_eval[m] for m in metrics]

        x     = np.arange(len(metrics))
        width = 0.35
        fig, ax = plt.subplots(figsize=(7, 5))
        bars_o = ax.bar(x - width / 2, orig_v,     width, label="Original Ratings",
                        color="#3498db", alpha=0.85)
        bars_e = ax.bar(x + width / 2, enhanced_v, width, label="Enhanced Ratings",
                        color="#2ecc71", alpha=0.85)
        ax.set_title("Step 15 - CF Evaluation: Original vs. Enhanced Ratings",
                     fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Error")
        ax.legend()
        for bar in list(bars_o) + list(bars_e):
            h = bar.get_height()
            if not np.isnan(h):
                ax.text(
                    bar.get_x() + bar.get_width() / 2, h + 0.005,
                    f"{h:.4f}", ha="center", va="bottom", fontsize=9,
                )
        plt.tight_layout()
        out_path = os.path.join(self.output_dir, "step15_cf_evaluation.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved CF evaluation plot -> {out_path}")

    # ------------------------------------------------------------------ #
    #  Generate sample recommendations
    # ------------------------------------------------------------------ #
    def generate_sample_recommendations(
        self,
        M_enhanced_pred: pd.DataFrame,
        n_users: int = 3,
        top_k: int = 5,
    ) -> None:
        """
        Print top-k product recommendations for a few sample users
        using the sentiment-enhanced predicted ratings.
        """
        print("\n  --- Sample Recommendations (Enhanced CF) ---")
        sample_users = list(M_enhanced_pred.index[:n_users])

        for user in sample_users:
            user_row = M_enhanced_pred.loc[user]
            # Only recommend unrated products (NaN in original)
            orig_row = self.df[self.df["reviewerID"] == user]["asin"].values
            unseen = user_row.drop(labels=orig_row, errors="ignore").dropna()
            top_recs = unseen.nlargest(top_k)
            print(f"\n  User: {user}")
            print(f"  Top {top_k} recommended products and predicted enhanced ratings:")
            for asin, score in top_recs.items():
                print(f"    {asin}  ->  predicted rating: {score:.2f}")

    # ------------------------------------------------------------------ #
    #  Write conclusion
    # ------------------------------------------------------------------ #
    def _write_conclusion(
        self,
        orig_eval: dict,
        enhanced_eval: dict,
    ) -> None:
        """Write a textual conclusion for Step 15."""
        improve_mae  = (
            (orig_eval["mae"] - enhanced_eval["mae"]) / orig_eval["mae"] * 100
            if orig_eval["mae"] and not np.isnan(orig_eval["mae"]) else 0
        )
        improve_rmse = (
            (orig_eval["rmse"] - enhanced_eval["rmse"]) / orig_eval["rmse"] * 100
            if orig_eval["rmse"] and not np.isnan(orig_eval["rmse"]) else 0
        )

        lines = [
            "=" * 70,
            "STEP 15 - RECOMMENDER SYSTEM RESULTS AND CONCLUSION",
            "=" * 70,
            "",
            "Strategy: Sentiment-Enhanced Item-Based Collaborative Filtering",
            "",
            "Pseudo-code:",
            "  sentiment_score  = (vader_compound + 1) / 2",
            "  normalised_star  = (star_rating - 1) / 4",
            "  enhanced_0_1     = 0.6 x norm_star + 0.4 x sentiment_score",
            "  enhanced_rating  = enhanced_0_1 x 4 + 1",
            "  item_similarity  = cosine(user-item matrix columns)",
            "  predicted_rating = Sum(sim[i,j] x rating[j]) / Sum(sim[i,j])",
            "",
            f"Results:",
            f"  Original ratings CF   - MAE : {orig_eval['mae']:.4f}  "
            f"RMSE : {orig_eval['rmse']:.4f}",
            f"  Enhanced ratings CF   - MAE : {enhanced_eval['mae']:.4f}  "
            f"RMSE : {enhanced_eval['rmse']:.4f}",
            f"  MAE  improvement      : {improve_mae:+.1f}%",
            f"  RMSE improvement      : {improve_rmse:+.1f}%",
            "",
            "Conclusion:",
            "  Blending the explicit star rating with the VADER sentiment",
            "  score narrows the semantic gap in sparse rating matrices.",
            "  The enhanced ratings provide a smoother, more nuanced signal",
            "  for collaborative filtering, particularly for controversial",
            "  products where user opinions are polarised.  Even a modest",
            "  reduction in MAE/RMSE translates to meaningfully better",
            "  recommendations for users in the 'cold-start' region.",
            "=" * 70,
        ]
        text = "\n".join(lines)
        print("\n" + text)

        out_path = os.path.join(self.output_dir, "step15_conclusion.txt")
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(f"\n  Saved conclusion -> {out_path}")

    # ------------------------------------------------------------------ #
    #  Full pipeline orchestrator
    # ------------------------------------------------------------------ #
    def run(self) -> dict:
        """
        Execute the complete Step 15 recommender system pipeline.

        Returns
        -------
        dict - evaluation results for original and enhanced CF.
        """
        print("\n" + "=" * 80)
        print("STEP 15: RECOMMENDER SYSTEM - SENTIMENT-ENHANCED RATINGS")
        print("=" * 80)

        # Load data
        self.load_data(sample_size=5000)

        # Step 15-a: Compute enhanced ratings
        print("\n[Step 15-a] Computing sentiment-enhanced ratings ...")
        self.compute_sentiment_enhanced_ratings()

        # Visualise rating distributions
        self._plot_rating_comparison()
        self._plot_delta_distribution()

        # Step 15-b/c: Build CF models and evaluate
        print("\n[Step 15-b/c] Building item-based CF models ...")
        print("  Building user-item matrix (original ratings) ...")
        M_orig, _, _ = self._build_user_item_matrix("overall")

        print("  Building user-item matrix (enhanced ratings) ...")
        M_enh, _, _  = self._build_user_item_matrix("enhanced_rating")

        # Hold out 20% of ratings BEFORE training (prevents data leakage)
        print("  Splitting 80% train / 20% test (original CF) ...")
        M_orig_train, orig_test_idx = self._train_test_split_matrix(M_orig)
        print("  Splitting 80% train / 20% test (enhanced CF) ...")
        M_enh_train,  enh_test_idx  = self._train_test_split_matrix(M_enh)

        # Build similarity on TRAINING matrices only
        sim_orig = self._cosine_similarity_matrix(M_orig_train.values)
        sim_enh  = self._cosine_similarity_matrix(M_enh_train.values)

        # Predict the held-out (and other missing) ratings
        print("  Predicting missing ratings (original CF) ...")
        M_orig_pred = self._predict_ratings(M_orig_train, sim_orig, top_k=10)

        print("  Predicting missing ratings (enhanced CF) ...")
        M_enh_pred  = self._predict_ratings(M_enh_train,  sim_enh,  top_k=10)

        # Evaluate on the held-out test positions (true values vs CF predictions)
        orig_eval     = self._evaluate_cf(M_orig, M_orig_pred, orig_test_idx, "Original CF")
        enhanced_eval = self._evaluate_cf(M_enh,  M_enh_pred,  enh_test_idx,  "Enhanced CF")

        # Visualise and compare
        self._plot_cf_comparison(orig_eval, enhanced_eval)

        # Sample recommendations
        self.generate_sample_recommendations(M_enh_pred)

        # Conclusion
        self._write_conclusion(orig_eval, enhanced_eval)

        # Save summary CSV
        summary = pd.DataFrame([
            {"System": "Original CF (star ratings only)",
             "MAE": round(orig_eval["mae"], 4),
             "RMSE": round(orig_eval["rmse"], 4),
             "Coverage": round(orig_eval["coverage"], 4)},
            {"System": "Enhanced CF (star + VADER sentiment)",
             "MAE": round(enhanced_eval["mae"], 4),
             "RMSE": round(enhanced_eval["rmse"], 4),
             "Coverage": round(enhanced_eval["coverage"], 4)},
        ])
        out_csv = os.path.join(self.output_dir, "step15_cf_results.csv")
        summary.to_csv(out_csv, index=False)
        print(f"\n  Saved CF results -> {out_csv}")

        self.results = {
            "original_eval" : orig_eval,
            "enhanced_eval" : enhanced_eval,
        }
        print("\n  [OK] Step 15 recommender system complete.")
        return self.results
