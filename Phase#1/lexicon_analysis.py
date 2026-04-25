"""
Phase #1 - Step 3: Study Lexicon Packages
Team #4: Industrial and Scientific Dataset

This module studies and compares three lexicon-based sentiment tools:
    1. VADER  (Valence Aware Dictionary and Sentiment Reasoner)
    2. TextBlob
    3. SentiWordNet

After comparison the team selects **VADER** and **TextBlob** for model
building, with full justification documented in the comparison report.

"""

import pandas as pd
import numpy as np
import time
import warnings
warnings.filterwarnings("ignore")

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

import nltk
for res in ("punkt", "punkt_tab", "averaged_perceptron_tagger",
            "averaged_perceptron_tagger_eng",
            "wordnet", "sentiwordnet", "stopwords"):
    try:
        nltk.data.find(f"tokenizers/{res}" if "punkt" in res
                       else f"taggers/{res}" if "tagger" in res
                       else f"corpora/{res}")
    except LookupError:
        nltk.download(res, quiet=True)

from nltk.corpus import sentiwordnet as swn, stopwords
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.stem import WordNetLemmatizer


class LexiconAnalyzer:
    """Compare VADER, TextBlob, and SentiWordNet on sample texts."""

    def __init__(self):
        """Initialise the three lexicon analysers."""
        self.vader = SentimentIntensityAnalyzer()
        self.lemmatizer = WordNetLemmatizer()
        self._stop = set(stopwords.words("english"))

    # ------------------------------------------------------------------ #
    #  VADER
    # ------------------------------------------------------------------ #
    def analyze_vader(self, text: str) -> dict:
        """
        Analyse sentiment using VADER.

        VADER is attuned to social-media / informal text and handles:
        capitalisation, punctuation emphasis, degree modifiers,
        conjunctions (``but``), and negation.

        Returns:
            dict with keys: compound, positive, neutral, negative, label.
        """
        s = self.vader.polarity_scores(str(text))
        return {
            "compound": s["compound"],
            "positive": s["pos"],
            "neutral": s["neu"],
            "negative": s["neg"],
            "label": self._vader_label(s["compound"]),
        }

    @staticmethod
    def _vader_label(compound: float) -> str:
        if compound >= 0.05:
            return "Positive"
        if compound <= -0.05:
            return "Negative"
        return "Neutral"

    # ------------------------------------------------------------------ #
    #  TextBlob
    # ------------------------------------------------------------------ #
    def analyze_textblob(self, text: str) -> dict:
        """
        Analyse sentiment using TextBlob.

        TextBlob returns **polarity** ∈ [-1, 1] and **subjectivity** ∈ [0, 1].

        Returns:
            dict with keys: polarity, subjectivity, label.
        """
        blob = TextBlob(str(text))
        p = blob.sentiment.polarity
        return {
            "polarity": p,
            "subjectivity": blob.sentiment.subjectivity,
            "label": self._textblob_label(p),
        }

    @staticmethod
    def _textblob_label(polarity: float) -> str:
        if polarity > 0.1:
            return "Positive"
        if polarity < -0.1:
            return "Negative"
        return "Neutral"

    # ------------------------------------------------------------------ #
    #  SentiWordNet
    # ------------------------------------------------------------------ #
    def _get_wn_pos(self, tag: str) -> str:
        """Map a Penn-Treebank POS tag to a WordNet POS character."""
        if tag.startswith("J"):
            return "a"
        if tag.startswith("V"):
            return "v"
        if tag.startswith("N"):
            return "n"
        if tag.startswith("R"):
            return "r"
        return "n"

    def analyze_sentiwordnet(self, text: str) -> dict:
        """
        Analyse sentiment using SentiWordNet.

        Each synset receives positivity, negativity, and objectivity
        scores.  The first (most common) synset is used for each lemma.

        Returns:
            dict with keys: positive_score, negative_score,
            objectivity_score, sentiment_score, label.
        """
        tokens = word_tokenize(str(text).lower())
        tagged = pos_tag(tokens)

        pos_s = neg_s = obj_s = 0.0
        count = 0

        for word, tag in tagged:
            if word in self._stop or not word.isalnum():
                continue
            wn_pos = self._get_wn_pos(tag)
            lemma = self.lemmatizer.lemmatize(word, pos=wn_pos)
            synsets = list(swn.senti_synsets(lemma, wn_pos))
            if synsets:
                ss = synsets[0]
                pos_s += ss.pos_score()
                neg_s += ss.neg_score()
                obj_s += ss.obj_score()
                count += 1

        if count:
            pos_s /= count; neg_s /= count; obj_s /= count

        score = pos_s - neg_s
        return {
            "positive_score": round(pos_s, 4),
            "negative_score": round(neg_s, 4),
            "objectivity_score": round(obj_s, 4),
            "sentiment_score": round(score, 4),
            "label": self._swn_label(score),
        }

    @staticmethod
    def _swn_label(score: float) -> str:
        if score > 0.1:
            return "Positive"
        if score < -0.1:
            return "Negative"
        return "Neutral"

    # ------------------------------------------------------------------ #
    #  Comparison utilities
    # ------------------------------------------------------------------ #
    def compare_lexicons(self, sample_texts: list) -> pd.DataFrame:
        """
        Run all three lexicons on a list of sample texts.

        Returns:
            DataFrame with label and score columns for each tool.
        """
        rows = []
        for i, txt in enumerate(sample_texts):
            v = self.analyze_vader(txt)
            t = self.analyze_textblob(txt)
            s = self.analyze_sentiwordnet(txt)
            rows.append({
                "id": i,
                "text": (txt[:80] + "...") if len(str(txt)) > 80 else txt,
                "vader_label": v["label"], "vader_compound": v["compound"],
                "tb_label": t["label"], "tb_polarity": t["polarity"],
                "swn_label": s["label"], "swn_score": s["sentiment_score"],
            })
        return pd.DataFrame(rows)

    def _time_lexicons(self, texts: list) -> dict:
        """Measure wall-clock time per lexicon on the given texts."""
        times = {}
        for name, fn in [("VADER", self.analyze_vader),
                         ("TextBlob", self.analyze_textblob),
                         ("SentiWordNet", self.analyze_sentiwordnet)]:
            t0 = time.perf_counter()
            for t in texts:
                fn(t)
            times[name] = round(time.perf_counter() - t0, 4)
        return times

    # ------------------------------------------------------------------ #
    #  Report generation
    # ------------------------------------------------------------------ #
    def generate_comparison_report(
        self,
        sample_texts: list,
        output_file: str = "lexicon_comparison_report.txt",
    ) -> str:
        """
        Generate a detailed comparison report saved to *output_file*.

        The report includes:
        * Strengths / weaknesses of each lexicon
        * Side-by-side results on sample texts
        * Agreement rates between lexicon pairs
        * Execution time comparison
        * Final recommendation with justification

        Args:
            sample_texts: List of review strings.
            output_file:  Path for the saved report.

        Returns:
            The full report as a string.
        """
        cmp = self.compare_lexicons(sample_texts)
        times = self._time_lexicons(sample_texts)

        R = []  # report lines
        R.append("=" * 80)
        R.append("LEXICON PACKAGES COMPARISON REPORT")
        R.append("Team #4 – Industrial and Scientific Dataset")
        R.append("=" * 80)

        # ---- 1. VADER ------------------------------------------------ #
        R.append("\n1. VADER (Valence Aware Dictionary and Sentiment Reasoner)")
        R.append("-" * 60)
        R.append("Strengths:")
        R.append("  • Designed for social-media / informal text (product reviews)")
        R.append("  • Handles CAPITALISATION as emphasis")
        R.append("  • Handles punctuation emphasis (e.g. 'great!!!')")
        R.append("  • Handles degree modifiers ('very', 'extremely')")
        R.append("  • Handles conjunctions & negation ('but', 'not good')")
        R.append("  • Very fast — no heavy NLP pipeline required")
        R.append("  • Provides a compound score that aggregates all cues")
        R.append("Weaknesses:")
        R.append("  • May under-perform on formal / technical text")
        R.append("  • Rule-based — cannot be fine-tuned on domain data")

        # ---- 2. TextBlob --------------------------------------------- #
        R.append("\n2. TextBlob")
        R.append("-" * 60)
        R.append("Strengths:")
        R.append("  • Simple API — polarity + subjectivity in one call")
        R.append("  • Good general-purpose sentiment analyser")
        R.append("  • Subjectivity score can help filter factual reviews")
        R.append("  • Built on Pattern library with proven accuracy")
        R.append("Weaknesses:")
        R.append("  • Less sophisticated handling of informal language")
        R.append("  • No explicit handling of capitalisation / punctuation emphasis")

        # ---- 3. SentiWordNet ----------------------------------------- #
        R.append("\n3. SentiWordNet")
        R.append("-" * 60)
        R.append("Strengths:")
        R.append("  • Based on WordNet — semantic relationships available")
        R.append("  • Separate positive / negative / objectivity scores")
        R.append("  • Useful for word-level sentiment inspection")
        R.append("Weaknesses:")
        R.append("  • Requires heavy preprocessing (tokenise, POS-tag, lemmatise)")
        R.append("  • Significantly slower than VADER and TextBlob")
        R.append("  • First-synset heuristic may miss correct sense")
        R.append("  • Does not handle negation or degree modifiers")

        # ---- Sample results ------------------------------------------ #
        R.append("\n" + "=" * 80)
        R.append("SAMPLE COMPARISON RESULTS")
        R.append("=" * 80)
        R.append(cmp.to_string(index=False))

        # ---- Agreement ----------------------------------------------- #
        R.append("\n" + "=" * 80)
        R.append("AGREEMENT ANALYSIS")
        R.append("=" * 80)
        n = len(cmp)
        vt = int((cmp["vader_label"] == cmp["tb_label"]).sum())
        vs = int((cmp["vader_label"] == cmp["swn_label"]).sum())
        ts = int((cmp["tb_label"] == cmp["swn_label"]).sum())
        R.append(f"  VADER – TextBlob      : {vt}/{n} ({vt/n*100:.1f}%)")
        R.append(f"  VADER – SentiWordNet  : {vs}/{n} ({vs/n*100:.1f}%)")
        R.append(f"  TextBlob – SentiWordNet: {ts}/{n} ({ts/n*100:.1f}%)")

        # ---- Execution time ------------------------------------------ #
        R.append("\n" + "=" * 80)
        R.append("EXECUTION TIME (seconds for all sample texts)")
        R.append("=" * 80)
        for name, t in times.items():
            R.append(f"  {name:15s}: {t:.4f} s")

        # ---- Recommendation ----------------------------------------- #
        R.append("\n" + "=" * 80)
        R.append("RECOMMENDATION  (selected for model building)")
        R.append("=" * 80)
        R.append("""
For Amazon product reviews (Industrial & Scientific), we select:

  ★ VADER
    – Product reviews contain informal language, emphasis (!!!),
      capitalisation, and modifiers that VADER is specifically built for.
    – Fastest execution — practical for larger datasets.
    – Requires minimal preprocessing, preserving original text signals.

  ★ TextBlob
    – Provides a reliable second opinion with a different methodology.
    – Subjectivity score adds an extra dimension for analysis.
    – Good balance between accuracy and simplicity.

  ✗ SentiWordNet  (not selected)
    – Requires heavy NLP preprocessing which can remove sentiment cues.
    – Significantly slower — impractical at scale.
    – Word-sense disambiguation (first-synset heuristic) is unreliable
      on informal product reviews.
""")

        report = "\n".join(R)

        with open(output_file, "w", encoding="utf-8") as fh:
            fh.write(report)
        print(f"  Report saved: {output_file}")
        print("\n" + report)

        return report


if __name__ == "__main__":
    analyzer = LexiconAnalyzer()
    demo = [
        "This product is amazing! It works perfectly and exceeded my expectations.",
        "The item arrived broken and the quality is terrible. Very disappointed.",
        "It's okay, nothing special. Does what it's supposed to do.",
        "GREAT product!!! Highly recommend to everyone!",
        "Not bad, but could be better. The price is reasonable though.",
    ]
    analyzer.generate_comparison_report(demo)
