"""
Phase #2 - Steps 11-13: Sentiment Analysis - Machine Learning Approach
Team #4 : Industrial and Scientific Dataset

Steps covered here
------------------
Step 11 - Select >=2000-review subset, run data exploration, pre-process
          text, build a TF-IDF representation, perform a stratified 70/30
          split by rating field, and build two ML classifiers:
            - Logistic Regression
            - Linear Support Vector Machine (LinearSVC)
Step 12 - Record training process (cross-validation scores and hyper-
          parameter tuning via GridSearchCV).
Step 13 - Test both models on the hold-out set and report accuracy,
          precision, recall, F1-score and confusion matrix.

Text representation justification
-----------------------------------
TF-IDF (Term Frequency - Inverse Document Frequency) was chosen because:
  1. It captures term importance within a document while downweighting
     terms that appear in many documents (common words), which is exactly
     what sentiment analysis needs.
  2. It is computationally efficient and interpretable.
  3. Works well with short-to-medium length documents such as product
     reviews.
  4. Both LR and SVM have been shown to achieve state-of-the-art results
     on text classification tasks when combined with TF-IDF features.

Pre-processing justification (ML-specific)
-------------------------------------------
For ML-based classification, more aggressive pre-processing is beneficial
because the model learns weighted features rather than relying on
linguistics-based heuristics:
  - Lower-casing        - reduces vocabulary size without losing meaning.
  - URL / HTML removal  - removes noise that carries no sentiment signal.
  - Punctuation removal - not semantically meaningful for bag-of-words.
  - Stop-word removal   - reduces vocabulary; stop words carry little
                          discriminative power for sentiment.
  - Stemming (Porter)   - groups morphological variants, reducing sparsity.

Dataset reference:
    Ni, J., Li, J., & McAuley, J. (2019). Justifying recommendations using
    distantly-labeled reviews and fine-grained aspects. EMNLP-IJCNLP 2019.
    https://nijianmo.github.io/amazon/index.html
"""

import os
import sys
import json
import gzip
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

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model            import LogisticRegression
from sklearn.svm                     import LinearSVC
from sklearn.pipeline                import Pipeline
from sklearn.model_selection         import (
    train_test_split, StratifiedKFold, GridSearchCV, cross_val_score,
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)
from sklearn.calibration import CalibratedClassifierCV

# ------------------------------------------------------------------ #
#  Constants
# ------------------------------------------------------------------ #
LABELS      = ["Positive", "Neutral", "Negative"]
RANDOM_SEED = 42


class MLModeler:
    """
    Orchestrates the full machine-learning pipeline for Phase #2.

    Attributes
    ----------
    data_file  : str   Path to the raw Amazon JSON / JSON.GZ data file.
    output_dir : str   Directory for saving plots, reports and models.
    phase1_dir : str   Path to Phase #1 directory (used to exclude the
                       1 000-review lexicon sample from ML training).
    """

    def __init__(
        self,
        data_file : str,
        output_dir: str = ".",
        phase1_dir: str = "../Phase#1",
    ):
        self.data_file  = data_file
        self.output_dir = output_dir
        self.phase1_dir = phase1_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "models"), exist_ok=True)

        # Will be populated during the pipeline
        self.df_full:   pd.DataFrame  = None  # full loaded dataset
        self.df_ml:     pd.DataFrame  = None  # 2 000+ review subset
        self.tfidf:     TfidfVectorizer = None
        self.lr_pipeline  = None   # fitted LR pipeline
        self.svm_pipeline = None   # fitted SVM pipeline
        self.X_train = None
        self.X_test  = None
        self.y_train = None
        self.y_test  = None
        self.results: dict = {}

        self._stemmer    = PorterStemmer()
        self._stop_words = set(stopwords.words("english"))

    # ------------------------------------------------------------------ #
    #  Data loading
    # ------------------------------------------------------------------ #
    def load_amazon_data(self) -> pd.DataFrame:
        """
        Load Amazon review data from a line-delimited JSON or JSON.GZ file.

        Returns
        -------
        pd.DataFrame  - raw review records.
        """
        print(f"\n  Loading data from: {self.data_file}")
        if not os.path.exists(self.data_file):
            raise FileNotFoundError(
                f"Data file not found: {self.data_file}\n"
                "Please ensure Industrial_and_Scientific.json.gz is present."
            )

        opener = gzip.open if self.data_file.endswith(".gz") else open
        records = []
        with opener(self.data_file, "rt", encoding="utf-8") as fh:
            for ln, line in enumerate(fh, 1):
                try:
                    records.append(json.loads(line.strip()))
                except json.JSONDecodeError as exc:
                    print(f"    Warning - skipping line {ln}: {exc}")

        df = pd.DataFrame(records)
        print(f"  Loaded {len(df):,} total reviews")
        self.df_full = df
        return df

    # ------------------------------------------------------------------ #
    #  Subset selection (Step 11-a)
    # ------------------------------------------------------------------ #
    def select_ml_subset(self, min_reviews: int = 2500) -> pd.DataFrame:
        """
        Select a representative subset (>=2000 reviews) for ML.

        The 1000 reviews used for lexicon analysis in Phase #1 are
        identified via ``sentiment_analysis_results.csv`` and excluded
        from training to ensure the comparison in Step 14 is fair
        (no data leakage between lexicon test set and ML training set).

        Parameters
        ----------
        min_reviews : int  Minimum number of reviews to retain.

        Returns
        -------
        pd.DataFrame - the ML subset.
        """
        df = self.df_full.copy()

        # Keep only columns we care about
        needed = ["reviewText", "overall", "summary", "asin", "reviewerID"]
        available = [c for c in needed if c in df.columns]
        df = df[available].copy()

        # Drop rows without review text or overall rating
        df.dropna(subset=["reviewText", "overall"], inplace=True)
        df = df[df["reviewText"].str.strip().astype(bool)]
        df["overall"] = pd.to_numeric(df["overall"], errors="coerce")
        df.dropna(subset=["overall"], inplace=True)
        df["overall"] = df["overall"].astype(float)

        # ---- Exclude Phase #1 lexicon sample ----
        lexicon_csv = os.path.join(self.phase1_dir, "sentiment_analysis_results.csv")
        if os.path.exists(lexicon_csv):
            lex = pd.read_csv(lexicon_csv)
            # Build a set of (reviewText, overall) tuples to exclude
            lex_pairs = set(
                zip(lex["reviewText"].astype(str), lex["overall"].astype(float))
            )
            before = len(df)
            mask = ~df.apply(
                lambda r: (str(r["reviewText"]), float(r["overall"])) in lex_pairs,
                axis=1,
            )
            df = df[mask]
            excluded = before - len(df)
            print(f"  Excluded {excluded:,} Phase-1 lexicon reviews from ML data")
        else:
            print("  Warning - could not find sentiment_analysis_results.csv; "
                  "no exclusion applied.")

        # ---- Stratified sample (balanced classes) ----
        df["sentiment_label"] = df["overall"].apply(self._label)

        # Sample at least min_reviews, keeping class balance as close as possible
        n_per_class = max(min_reviews // 3, (df["sentiment_label"].value_counts().min()))
        sampled = (
            df.groupby("sentiment_label", group_keys=False)
            .apply(lambda g: g.sample(
                min(len(g), n_per_class + min_reviews // 3),
                random_state=RANDOM_SEED,
            ))
        )

        # If we still have fewer than min_reviews, take the rest randomly
        if len(sampled) < min_reviews:
            remaining = df[~df.index.isin(sampled.index)]
            extra = remaining.sample(
                min(len(remaining), min_reviews - len(sampled)),
                random_state=RANDOM_SEED,
            )
            sampled = pd.concat([sampled, extra])

        sampled = sampled.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
        print(f"  ML subset size: {len(sampled):,} reviews")
        print(f"  Label distribution:\n{sampled['sentiment_label'].value_counts()}")

        self.df_ml = sampled
        return sampled

    # ------------------------------------------------------------------ #
    #  Data exploration (Step 11-b)
    # ------------------------------------------------------------------ #
    def explore_subset(self) -> None:
        """
        Perform data exploration on the ML subset and save visualisations.

        Explores:
          - Label (sentiment class) distribution
          - Review length distribution (words and characters)
          - Rating distribution
          - Missing-value counts
          - Top products and users by review count
        """
        df = self.df_ml.copy()
        print("\n" + "=" * 60)
        print("STEP 11-b: DATA EXPLORATION (ML SUBSET)")
        print("=" * 60)

        # ---- Basic stats ----
        print(f"\n  Total reviews    : {len(df):,}")
        print(f"  Unique products  : {df['asin'].nunique():,}")
        print(f"  Unique reviewers : {df['reviewerID'].nunique():,}")
        print(f"  Missing reviewText: {df['reviewText'].isna().sum()}")
        print(f"  Missing overall   : {df['overall'].isna().sum()}")
        print(f"\n  Rating distribution:\n{df['overall'].value_counts().sort_index()}")
        print(f"\n  Sentiment label distribution:\n{df['sentiment_label'].value_counts()}")

        # ---- Review word / character counts ----
        df["wc"] = df["reviewText"].str.split().str.len()
        df["cc"] = df["reviewText"].str.len()
        print(f"\n  Review word count  - mean: {df['wc'].mean():.1f}, "
              f"median: {df['wc'].median():.0f}, "
              f"max: {df['wc'].max()}")
        print(f"  Review char count  - mean: {df['cc'].mean():.1f}, "
              f"median: {df['cc'].median():.0f}")

        # ---- Plots ----
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Phase #2 - ML Subset Data Exploration", fontsize=14, fontweight="bold")

        # Label distribution
        counts = df["sentiment_label"].value_counts()
        axes[0, 0].bar(counts.index, counts.values, color=["#2ecc71", "#f1c40f", "#e74c3c"])
        axes[0, 0].set_title("Sentiment Label Distribution")
        axes[0, 0].set_xlabel("Sentiment Label")
        axes[0, 0].set_ylabel("Count")
        for bar in axes[0, 0].patches:
            axes[0, 0].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 5,
                str(int(bar.get_height())),
                ha="center", va="bottom", fontsize=9,
            )

        # Rating distribution
        rating_counts = df["overall"].value_counts().sort_index()
        axes[0, 1].bar(
            rating_counts.index.astype(str), rating_counts.values, color="#3498db"
        )
        axes[0, 1].set_title("Star Rating Distribution")
        axes[0, 1].set_xlabel("Star Rating")
        axes[0, 1].set_ylabel("Count")

        # Review word count histogram
        axes[1, 0].hist(
            df["wc"].clip(upper=500), bins=50, color="#9b59b6", edgecolor="white"
        )
        axes[1, 0].set_title("Review Word Count Distribution (capped at 500)")
        axes[1, 0].set_xlabel("Word Count")
        axes[1, 0].set_ylabel("Frequency")

        # Box plot: word count per sentiment class
        classes = LABELS
        data_per_class = [
            df.loc[df["sentiment_label"] == c, "wc"].values for c in classes
        ]
        axes[1, 1].boxplot(data_per_class, labels=classes, patch_artist=True,
                           boxprops=dict(facecolor="#85c1e9"))
        axes[1, 1].set_title("Review Length by Sentiment Class")
        axes[1, 1].set_xlabel("Sentiment")
        axes[1, 1].set_ylabel("Word Count")

        plt.tight_layout()
        out_path = os.path.join(self.output_dir, "ml_subset_exploration.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\n  Saved exploration plot -> {out_path}")

    # ------------------------------------------------------------------ #
    #  Pre-processing (Step 11-b / Step 4 justification)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _label(overall: float) -> str:
        """Map numeric rating to sentiment label (same scheme as Phase #1)."""
        if overall >= 4:
            return "Positive"
        elif overall == 3:
            return "Neutral"
        else:
            return "Negative"

    def _clean_for_ml(self, text: str) -> str:
        """
        ML-specific text cleaning pipeline.

        Steps applied (with justification in module docstring):
          1. Null guard
          2. Lower-case
          3. Remove URLs
          4. Remove HTML tags
          5. Remove non-alphabetic characters
          6. Tokenise
          7. Stop-word removal
          8. Porter stemming
          9. Re-join tokens
        """
        if pd.isna(text) or not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r"https?://\S+|www\.\S+", " ", text)   # URLs
        text = re.sub(r"<[^>]+>", " ", text)                  # HTML
        text = re.sub(r"[^a-z\s]", " ", text)                 # non-alpha
        tokens = text.split()
        tokens = [
            self._stemmer.stem(t)
            for t in tokens
            if t not in self._stop_words and len(t) > 1
        ]
        return " ".join(tokens)

    def preprocess(self) -> None:
        """
        Apply ML-specific pre-processing to the subset and label the data.

        Adds columns:
          - ``sentiment_label``  - Positive / Neutral / Negative
          - ``clean_text``       - stemmed, stop-word-free text
          - ``wc``               - original word count
          - ``cc``               - original character count
        Removes:
          - Rows where clean_text is empty after cleaning.
          - Outliers: reviews with word count > 3x the 99th percentile.
        """
        print("\n" + "=" * 60)
        print("STEP 11-b: PRE-PROCESSING")
        print("=" * 60)

        df = self.df_ml.copy()

        # ---- Labels ----
        df["sentiment_label"] = df["overall"].apply(self._label)

        # ---- Word / char counts (before cleaning, for outlier detection) ----
        df["wc"] = df["reviewText"].str.split().str.len().fillna(0).astype(int)
        df["cc"] = df["reviewText"].str.len().fillna(0).astype(int)

        # ---- Outlier removal: extreme-length reviews ----
        p99 = df["wc"].quantile(0.99)
        threshold = p99 * 3
        before = len(df)
        df = df[df["wc"] <= threshold]
        print(f"  Removed {before - len(df)} extreme-length outliers "
              f"(word count > {threshold:.0f})")

        # ---- ML-specific text cleaning ----
        print("  Applying ML text cleaning (lower-case, remove URLs/HTML/punct, "
              "stop-word removal, stemming) ...")
        df["clean_text"] = df["full_text"].fillna("") if "full_text" in df.columns \
            else (df["summary"].fillna("") + " " + df["reviewText"].fillna(""))
        df["clean_text"] = df["clean_text"].apply(self._clean_for_ml)

        # Drop rows where cleaned text is empty
        before = len(df)
        df = df[df["clean_text"].str.strip().astype(bool)]
        print(f"  Removed {before - len(df)} rows with empty clean_text after cleaning")

        print(f"  Final ML subset size: {len(df):,}")
        print(f"  Label distribution:\n{df['sentiment_label'].value_counts()}")

        self.df_ml = df.reset_index(drop=True)

    # ------------------------------------------------------------------ #
    #  Train / test split - stratified 70 / 30 (Step 11-d)
    # ------------------------------------------------------------------ #
    def split_data(self) -> None:
        """
        Perform a stratified 70%/30% train-test split based on the
        ``sentiment_label`` (i.e. using the rating field as requested).
        """
        print("\n" + "=" * 60)
        print("STEP 11-d: STRATIFIED 70/30 TRAIN-TEST SPLIT")
        print("=" * 60)

        X = self.df_ml["clean_text"]
        y = self.df_ml["sentiment_label"]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size   = 0.30,
            stratify    = y,
            random_state= RANDOM_SEED,
        )

        print(f"  Training set : {len(self.X_train):,} reviews ({len(self.X_train)/len(X)*100:.1f}%)")
        print(f"  Test set     : {len(self.X_test):,}  reviews ({len(self.X_test)/len(X)*100:.1f}%)")
        print(f"\n  Training label distribution:\n{self.y_train.value_counts()}")
        print(f"\n  Test label distribution:\n{self.y_test.value_counts()}")

    # ------------------------------------------------------------------ #
    #  TF-IDF representation (Step 11-c)
    # ------------------------------------------------------------------ #
    def build_tfidf(self) -> None:
        """
        Fit a TF-IDF vectoriser on the training data only (no data leakage).

        Configuration choices:
          - ngram_range=(1, 2)  - unigrams + bigrams capture phrase-level
            sentiment signals (e.g. "not good", "highly recommend").
          - max_features=30 000 - limits vocabulary to prevent overfitting.
          - sublinear_tf=True   - applies log(tf)+1 scaling to reduce the
            effect of very high term frequencies.
          - min_df=3            - ignores very rare terms (noise).
        """
        print("\n" + "=" * 60)
        print("STEP 11-c: TF-IDF TEXT REPRESENTATION")
        print("=" * 60)

        self.tfidf = TfidfVectorizer(
            ngram_range  = (1, 2),
            max_features = 30_000,
            sublinear_tf = True,
            min_df       = 3,
        )
        self.X_train_tfidf = self.tfidf.fit_transform(self.X_train)
        self.X_test_tfidf  = self.tfidf.transform(self.X_test)

        print(f"  Vocabulary size : {len(self.tfidf.vocabulary_):,} features")
        print(f"  Training matrix : {self.X_train_tfidf.shape}")
        print(f"  Test matrix     : {self.X_test_tfidf.shape}")
        print(f"  n-gram range    : (1, 2)")
        print(f"  sublinear_tf    : True")
        print(f"  min_df          : 3")

        # Save vectoriser for later use in Step 14
        joblib.dump(
            self.tfidf,
            os.path.join(self.output_dir, "models", "tfidf_vectorizer.pkl"),
        )
        print("  Saved TF-IDF vectorizer -> models/tfidf_vectorizer.pkl")

    # ------------------------------------------------------------------ #
    #  Model 1 - Logistic Regression (Step 11-e-i)
    # ------------------------------------------------------------------ #
    def train_logistic_regression(self) -> None:
        """
        Train a Logistic Regression classifier with hyper-parameter tuning.

        Why Logistic Regression?
          - Probabilistic output (useful for confidence-based decisions).
          - Works very well with TF-IDF features.
          - Fast to train; highly interpretable (feature coefficients map
            directly to sentiment-driving words).
          - L2/L1 regularisation prevents overfitting on sparse TF-IDF.

        Tuning:
          - 5-fold stratified GridSearch over C = {0.01, 0.1, 1, 10}
          - Solver: lbfgs / saga (for larger datasets).
        """
        print("\n" + "=" * 60)
        print("STEP 11-e / 12: LOGISTIC REGRESSION TRAINING")
        print("=" * 60)

        lr = LogisticRegression(
            multi_class = "multinomial",
            max_iter    = 1000,
            random_state= RANDOM_SEED,
        )
        param_grid = {"C": [0.1, 1.0, 10.0]}
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

        print("  Running 5-fold GridSearchCV (C in {0.1, 1, 10}) ...")
        grid_lr = GridSearchCV(
            lr, param_grid, cv=cv,
            scoring="f1_macro", n_jobs=-1, verbose=0,
        )
        grid_lr.fit(self.X_train_tfidf, self.y_train)

        best_C   = grid_lr.best_params_["C"]
        best_cv  = grid_lr.best_score_
        print(f"  Best C = {best_C}  |  CV F1-macro = {best_cv:.4f}")

        # Refit with best params (GridSearchCV already stores best_estimator_)
        self.lr_model = grid_lr.best_estimator_

        # Cross-validation scores with best model
        cv_scores = cross_val_score(
            self.lr_model, self.X_train_tfidf, self.y_train,
            cv=cv, scoring="f1_macro",
        )
        print(f"  Cross-val F1-macro scores: {np.round(cv_scores, 4)}")
        print(f"  Mean: {cv_scores.mean():.4f}  +/-  Std: {cv_scores.std():.4f}")

        # Save model
        joblib.dump(
            self.lr_model,
            os.path.join(self.output_dir, "models", "logistic_regression.pkl"),
        )
        print("  Saved model -> models/logistic_regression.pkl")

        self.results["LR_cv_f1_mean"] = cv_scores.mean()
        self.results["LR_cv_f1_std"]  = cv_scores.std()
        self.results["LR_best_C"]     = best_C

    # ------------------------------------------------------------------ #
    #  Model 2 - Linear SVM (Step 11-e-ii)
    # ------------------------------------------------------------------ #
    def train_svm(self) -> None:
        """
        Train a Linear SVM classifier (LinearSVC) with hyper-parameter tuning.

        Why Linear SVM?
          - Historically the best-performing classifier for high-dimensional
            sparse text features.
          - The maximum-margin decision boundary generalises well on TF-IDF.
          - LinearSVC scales linearly with number of samples and features.

        Tuning:
          - 5-fold stratified GridSearch over C = {0.01, 0.1, 1}
          - Wrapped with CalibratedClassifierCV for probability estimates.
        """
        print("\n" + "=" * 60)
        print("STEP 11-e / 12: LINEAR SVM TRAINING")
        print("=" * 60)

        base_svm = LinearSVC(max_iter=2000, random_state=RANDOM_SEED)
        param_grid = {"C": [0.01, 0.1, 1.0]}
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

        print("  Running 5-fold GridSearchCV (C in {0.01, 0.1, 1}) ...")
        grid_svm = GridSearchCV(
            base_svm, param_grid, cv=cv,
            scoring="f1_macro", n_jobs=-1, verbose=0,
        )
        grid_svm.fit(self.X_train_tfidf, self.y_train)

        best_C  = grid_svm.best_params_["C"]
        best_cv = grid_svm.best_score_
        print(f"  Best C = {best_C}  |  CV F1-macro = {best_cv:.4f}")

        # Wrap best LinearSVC with calibration for probability support
        self.svm_model = CalibratedClassifierCV(
            LinearSVC(C=best_C, max_iter=2000, random_state=RANDOM_SEED),
            cv=5,
        )
        self.svm_model.fit(self.X_train_tfidf, self.y_train)

        # Cross-validation scores
        cv_scores = cross_val_score(
            grid_svm.best_estimator_,
            self.X_train_tfidf, self.y_train,
            cv=cv, scoring="f1_macro",
        )
        print(f"  Cross-val F1-macro scores: {np.round(cv_scores, 4)}")
        print(f"  Mean: {cv_scores.mean():.4f}  +/-  Std: {cv_scores.std():.4f}")

        # Save model
        joblib.dump(
            self.svm_model,
            os.path.join(self.output_dir, "models", "linear_svm.pkl"),
        )
        print("  Saved model -> models/linear_svm.pkl")

        self.results["SVM_cv_f1_mean"] = cv_scores.mean()
        self.results["SVM_cv_f1_std"]  = cv_scores.std()
        self.results["SVM_best_C"]     = best_C

    # ------------------------------------------------------------------ #
    #  Evaluation helper (Step 13)
    # ------------------------------------------------------------------ #
    def evaluate_model(self, model, model_name: str) -> dict:
        """
        Evaluate a fitted model on the hold-out test set.

        Parameters
        ----------
        model      : fitted scikit-learn estimator with predict().
        model_name : label string used for printing and plot filenames.

        Returns
        -------
        dict - accuracy, precision, recall, F1 (macro + weighted),
               and confusion matrix.
        """
        y_pred = model.predict(self.X_test_tfidf)

        acc  = accuracy_score (self.y_test, y_pred)
        prec = precision_score(self.y_test, y_pred, labels=LABELS,
                               average="macro", zero_division=0)
        rec  = recall_score   (self.y_test, y_pred, labels=LABELS,
                               average="macro", zero_division=0)
        f1   = f1_score       (self.y_test, y_pred, labels=LABELS,
                               average="macro", zero_division=0)
        prec_w = precision_score(self.y_test, y_pred, labels=LABELS,
                                  average="weighted", zero_division=0)
        rec_w  = recall_score   (self.y_test, y_pred, labels=LABELS,
                                  average="weighted", zero_division=0)
        f1_w   = f1_score       (self.y_test, y_pred, labels=LABELS,
                                  average="weighted", zero_division=0)
        cm = confusion_matrix(self.y_test, y_pred, labels=LABELS)

        print(f"\n  === {model_name} - Test Results ===")
        print(f"  Accuracy           : {acc:.4f}")
        print(f"  Precision (macro)  : {prec:.4f}")
        print(f"  Recall    (macro)  : {rec:.4f}")
        print(f"  F1-Score  (macro)  : {f1:.4f}")
        print(f"  Precision (weighted): {prec_w:.4f}")
        print(f"  Recall    (weighted): {rec_w:.4f}")
        print(f"  F1-Score  (weighted): {f1_w:.4f}")
        print(f"\n  Classification Report:\n"
              f"{classification_report(self.y_test, y_pred, labels=LABELS, zero_division=0)}")

        self._plot_confusion_matrix(cm, model_name)

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
            "y_pred"            : y_pred,
        }

    def _plot_confusion_matrix(self, cm: np.ndarray, model_name: str) -> None:
        """Produce and save a labelled confusion matrix heat-map."""
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=LABELS, yticklabels=LABELS, ax=ax,
        )
        ax.set_title(f"Confusion Matrix - {model_name}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        fname = model_name.lower().replace(" ", "_") + "_confusion_matrix.png"
        out_path = os.path.join(self.output_dir, fname)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Confusion matrix saved -> {out_path}")

    # ------------------------------------------------------------------ #
    #  Comparison plot
    # ------------------------------------------------------------------ #
    def plot_model_comparison(
        self,
        lr_metrics : dict,
        svm_metrics: dict,
    ) -> None:
        """
        Save a side-by-side bar chart comparing LR and SVM on key metrics.
        """
        metrics = ["accuracy", "precision_macro", "recall_macro", "f1_macro",
                   "f1_weighted"]
        labels  = ["Accuracy", "Precision\n(macro)", "Recall\n(macro)",
                   "F1\n(macro)", "F1\n(weighted)"]

        lr_vals  = [lr_metrics[m]  for m in metrics]
        svm_vals = [svm_metrics[m] for m in metrics]

        x = np.arange(len(metrics))
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 5))
        bars_lr  = ax.bar(x - width / 2, lr_vals,  width, label="Logistic Regression",
                          color="#2980b9", alpha=0.85)
        bars_svm = ax.bar(x + width / 2, svm_vals, width, label="Linear SVM",
                          color="#e67e22", alpha=0.85)

        ax.set_title("Phase #2 - ML Model Comparison (Test Set)", fontsize=12,
                     fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Score")
        ax.legend(loc="lower right")

        for bar in list(bars_lr) + list(bars_svm):
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2, h + 0.01,
                f"{h:.3f}", ha="center", va="bottom", fontsize=8,
            )

        plt.tight_layout()
        out_path = os.path.join(self.output_dir, "ml_model_comparison.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\n  Saved model comparison plot -> {out_path}")

    # ------------------------------------------------------------------ #
    #  Save comparison table (CSV)
    # ------------------------------------------------------------------ #
    def save_comparison_table(
        self,
        lr_metrics : dict,
        svm_metrics: dict,
    ) -> None:
        """Save a CSV comparison table of both ML models."""
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
            rows.append({
                "Metric"              : label,
                "Logistic Regression" : round(lr_metrics[key],  4),
                "Linear SVM"          : round(svm_metrics[key], 4),
                "Difference (LR-SVM)" : round(lr_metrics[key] - svm_metrics[key], 4),
            })

        table_df = pd.DataFrame(rows)
        out_path = os.path.join(self.output_dir, "ml_comparison_table.csv")
        table_df.to_csv(out_path, index=False)
        print(f"  Saved comparison table -> {out_path}")
        print(f"\n{table_df.to_string(index=False)}")

    # ------------------------------------------------------------------ #
    #  Top informative features
    # ------------------------------------------------------------------ #
    def print_top_features(self, n: int = 15) -> None:
        """
        Display the most informative TF-IDF features for each class
        (Logistic Regression only - coefficients are interpretable).
        """
        if self.lr_model is None or self.tfidf is None:
            return
        print("\n  Top TF-IDF features (Logistic Regression):")
        feature_names = np.array(self.tfidf.get_feature_names_out())
        for i, cls in enumerate(self.lr_model.classes_):
            coefs = self.lr_model.coef_[i]
            top_idx = np.argsort(coefs)[-n:][::-1]
            print(f"  {cls}: {', '.join(feature_names[top_idx])}")

    # ------------------------------------------------------------------ #
    #  Full pipeline orchestrator
    # ------------------------------------------------------------------ #
    def run_full_pipeline(self) -> dict:
        """
        Execute the complete Phase #2 ML pipeline (Steps 11-13).

        Returns
        -------
        dict - all collected metrics for both models.
        """
        print("\n" + "=" * 80)
        print("PHASE #2 - MACHINE LEARNING APPROACH (STEPS 11-13)")
        print("=" * 80)

        # Step 11-a: Load data and select subset
        self.load_amazon_data()
        self.select_ml_subset(min_reviews=2500)

        # Step 11-b: Explore subset
        self.explore_subset()

        # Step 11-b (cont.): Pre-process
        self.preprocess()

        # Step 11-d: Stratified split
        self.split_data()

        # Step 11-c: TF-IDF representation
        self.build_tfidf()

        # Steps 11-e / 12: Train both models
        self.train_logistic_regression()
        self.train_svm()

        # Step 13: Evaluate on test set
        print("\n" + "=" * 60)
        print("STEP 13: TESTING BOTH MODELS ON HOLD-OUT TEST SET")
        print("=" * 60)

        lr_metrics  = self.evaluate_model(self.lr_model,  "Logistic Regression")
        svm_metrics = self.evaluate_model(self.svm_model, "Linear SVM")

        # Visualise and save
        self.plot_model_comparison(lr_metrics, svm_metrics)
        self.save_comparison_table(lr_metrics, svm_metrics)
        self.print_top_features()

        self.results["lr"]  = lr_metrics
        self.results["svm"] = svm_metrics

        print("\n  [OK] Phase #2 ML pipeline complete.")
        return self.results
