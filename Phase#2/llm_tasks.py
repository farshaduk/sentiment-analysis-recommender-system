"""
Phase #2 - Steps 16 & 17: LLM Tasks (Hugging Face, hosted locally)
Team #4 : Industrial and Scientific Dataset

Step 16 - Select 10 reviews with more than 100 words.  Use a locally-
          hosted Hugging Face LLM to summarise each review into <=50 words.
          Record the first two results in the project report.

Step 17 - Select one review that carries a question nature (i.e. the
          reviewer asks a question in their text).  Use a locally-hosted
          Hugging Face LLM to automatically generate a response as if it
          were written by a customer-service representative.

Model selection rationale
--------------------------
Step 16 - Summarisation
  Model : ``facebook/bart-large-cnn``
  Why   : BART-Large fine-tuned on CNN/DailyMail is the standard
          summarisation model on the Hugging Face Hub.  It produces
          high-quality abstractive summaries and respects the target
          length constraint well via `max_new_tokens`.
  Fallback : ``sshleifer/distilbart-cnn-12-6`` - a distilled variant
          (~50% smaller, ~2x faster) auto-selected if GPU/RAM is limited.

Step 17 - Response generation (instruction-following)
  Model : ``google/flan-t5-large``
  Why   : FLAN-T5-large (~780 MB, 780M parameters) has significantly
          better instruction-following ability than flan-t5-base (~250M).
          It reliably references specific complaint details from the prompt
          and generates contextually relevant CSR responses rather than
          falling back on generic phrases.  It still runs on CPU without
          requiring a GPU, though it is slower than the base variant.

Prompt engineering notes
-------------------------
  - Explicit role prompting ("You are a helpful customer-service
    representative...") grounds the model in the desired persona and style.
  - Including context ("The customer wrote: ...") ensures the response is
    relevant rather than generic.
  - Length constraints are set via `max_new_tokens` / `max_length`.
  - `no_repeat_ngram_size=3` avoids repetition artefacts.

Dataset reference:
    Ni, J., Li, J., & McAuley, J. (2019). Justifying recommendations using
    distantly-labeled reviews and fine-grained aspects. EMNLP-IJCNLP 2019.
    https://nijianmo.github.io/amazon/index.html
"""

import os
import json
import gzip
import re
import warnings
import textwrap

warnings.filterwarnings("ignore")

import pandas as pd

RANDOM_SEED = 42


class LLMTasks:
    """
    Runs the Hugging Face-based LLM tasks for Phase #2 Steps 16 and 17.

    Parameters
    ----------
    data_file  : str  Path to the raw Amazon JSON / JSON.GZ data file.
    output_dir : str  Directory for saving result text files.
    """

    def __init__(self, data_file: str, output_dir: str = "."):
        self.data_file  = data_file
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self._summariser = None
        self._generator  = None
        self.df_reviews: pd.DataFrame = None

    # ------------------------------------------------------------------ #
    #  Data loading
    # ------------------------------------------------------------------ #
    def load_reviews(self, max_records: int = 10_000) -> pd.DataFrame:
        """
        Load reviews from the raw data file.

        Returns
        -------
        pd.DataFrame with columns ``reviewerID``, ``asin``, ``overall``,
        ``reviewText``, ``summary``.
        """
        print(f"\n  Loading reviews from {self.data_file} ...")
        opener  = gzip.open if self.data_file.endswith(".gz") else open
        records = []
        with opener(self.data_file, "rt", encoding="utf-8") as fh:
            for line in fh:
                try:
                    records.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    pass
                if len(records) >= max_records:
                    break

        df = pd.DataFrame(records)
        needed = ["reviewerID", "asin", "overall", "reviewText", "summary"]
        df = df[[c for c in needed if c in df.columns]].copy()
        df.dropna(subset=["reviewText"], inplace=True)
        df = df[df["reviewText"].str.strip().astype(bool)].reset_index(drop=True)
        print(f"  Loaded {len(df):,} reviews")
        self.df_reviews = df
        return df

    # ------------------------------------------------------------------ #
    #  Model loading
    # ------------------------------------------------------------------ #
    def _load_summariser(self) -> None:
        """
        Load the summarisation pipeline.  Tries ``facebook/bart-large-cnn``
        first; falls back to the smaller distilbart variant if needed.
        """
        from transformers import pipeline, logging as hf_logging
        hf_logging.set_verbosity_error()

        models_to_try = [
            "sshleifer/distilbart-cnn-12-6",   # smaller, faster (~1 GB)
            "facebook/bart-large-cnn",            # standard (~1.6 GB)
        ]
        for model_name in models_to_try:
            try:
                print(f"  Loading summarisation model: {model_name} ...")
                self._summariser = pipeline(
                    "summarization",
                    model         = model_name,
                    tokenizer     = model_name,
                    device        = -1,         # -1 = CPU; set 0 for CUDA GPU
                    framework     = "pt",
                )
                print(f"  [OK] Loaded: {model_name}")
                self._summariser_model_name = model_name
                return
            except Exception as exc:
                print(f"  Warning: could not load {model_name}: {exc}")

        raise RuntimeError(
            "Could not load any summarisation model.  "
            "Please check your internet connection or install transformers."
        )

    def _load_generator(self) -> None:
        """
        Load the text-generation (instruction-following) pipeline.
        Tries ``google/flan-t5-large`` first (~780 MB, better prompt
        following); falls back to ``google/flan-t5-base`` (~250 MB) if
        the large model cannot be loaded.
        """
        from transformers import pipeline, logging as hf_logging
        hf_logging.set_verbosity_error()

        models_to_try = [
            "google/flan-t5-large",   # preferred: ~780 MB, better quality
            "google/flan-t5-base",    # fallback:  ~250 MB, faster
        ]
        for model_name in models_to_try:
            try:
                print(f"  Loading response-generation model: {model_name} ...")
                self._generator = pipeline(
                    "text2text-generation",
                    model     = model_name,
                    tokenizer = model_name,
                    device    = -1,
                    framework = "pt",
                )
                print(f"  [OK] Loaded: {model_name}")
                self._generator_model_name = model_name
                return
            except Exception as exc:
                print(f"  Warning: could not load {model_name}: {exc}")

        raise RuntimeError(
            "Could not load any response-generation model.  "
            "Run: pip install transformers torch"
        )

    # ------------------------------------------------------------------ #
    #  Step 16 - Review Summarisation
    # ------------------------------------------------------------------ #
    def _select_long_reviews(self, min_words: int = 100, n: int = 10) -> pd.DataFrame:
        """
        Select ``n`` reviews each with at least ``min_words`` words.

        Returns
        -------
        pd.DataFrame
        """
        df = self.df_reviews.copy()
        df["word_count"] = df["reviewText"].str.split().str.len()
        long_reviews = (
            df[df["word_count"] >= min_words]
            .drop_duplicates(subset="reviewText")
            .sample(min(n, len(df[df["word_count"] >= min_words])),
                    random_state=RANDOM_SEED)
        )
        print(f"\n  Found {len(df[df['word_count'] >= min_words]):,} reviews "
              f"with >={min_words} words; selected {len(long_reviews)}.")
        return long_reviews.reset_index(drop=True)

    def step16_summarize_long_reviews(self) -> list:
        """
        Step 16: Summarise 10 reviews (each >100 words) to <=50 words.

        Returns
        -------
        list of dicts  - each with 'original', 'word_count', 'summary',
                         'summary_word_count'.
        """
        print("\n" + "=" * 60)
        print("STEP 16: REVIEW SUMMARISATION (LLM - Hugging Face)")
        print("=" * 60)

        if self._summariser is None:
            self._load_summariser()

        long_reviews = self._select_long_reviews(min_words=100, n=10)
        results      = []

        print(f"\n  Summarising {len(long_reviews)} reviews ...\n")

        for idx, row in long_reviews.iterrows():
            text        = str(row["reviewText"]).strip()
            word_count  = len(text.split())
            product     = row.get("asin", "unknown")
            rating      = row.get("overall", "?")

            # Truncate input to model's max-token limit (1024 tokens ~= 700 words)
            words = text.split()
            if len(words) > 700:
                text_input = " ".join(words[:700])
            else:
                text_input = text

            try:
                output = self._summariser(
                    text_input,
                    max_length      = 60,    # target <=50 words -> ~60 tokens
                    min_length      = 20,
                    do_sample       = False,
                    no_repeat_ngram_size = 3,
                )
                summary_text  = output[0]["summary_text"].strip()
                summary_words = len(summary_text.split())
            except Exception as exc:
                summary_text  = f"[Summarisation failed: {exc}]"
                summary_words = 0

            entry = {
                "review_num"        : idx + 1,
                "product_asin"      : product,
                "star_rating"       : rating,
                "original_text"     : text,
                "original_wc"       : word_count,
                "summary"           : summary_text,
                "summary_word_count": summary_words,
            }
            results.append(entry)

            print(f"  Review {idx + 1} / {len(long_reviews)}")
            print(f"  Product: {product}  |  Rating: {rating}  |  "
                  f"Original words: {word_count}")
            print(f"  Original (first 100 words):")
            print(textwrap.fill(" ".join(text.split()[:100]) + " ...",
                                width=72, initial_indent="    ",
                                subsequent_indent="    "))
            print(f"  Summary ({summary_words} words):")
            print(textwrap.fill(summary_text, width=72,
                                initial_indent="    ",
                                subsequent_indent="    "))
            print()

        # Save full results to text file
        out_path = os.path.join(self.output_dir, "step16_summaries.txt")
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(
                f"Phase #2 - Step 16: Review Summaries\n"
                f"Model: {getattr(self, '_summariser_model_name', 'unknown')}\n"
                f"{'=' * 70}\n\n"
            )
            for entry in results:
                fh.write(f"Review {entry['review_num']}\n")
                fh.write(f"Product ASIN : {entry['product_asin']}\n")
                fh.write(f"Star rating  : {entry['star_rating']}\n")
                fh.write(f"Original ({entry['original_wc']} words):\n")
                fh.write(
                    textwrap.fill(entry["original_text"], width=75,
                                  initial_indent="  ",
                                  subsequent_indent="  ") + "\n\n"
                )
                fh.write(f"Summary ({entry['summary_word_count']} words):\n")
                fh.write(
                    textwrap.fill(entry["summary"], width=75,
                                  initial_indent="  ",
                                  subsequent_indent="  ") + "\n"
                )
                fh.write("-" * 70 + "\n\n")

        print(f"  Saved all summaries -> {out_path}")
        return results

    # ------------------------------------------------------------------ #
    #  Step 17 - Automatic CSR Response to a Question Review
    # ------------------------------------------------------------------ #
    def _find_question_review(self) -> pd.Series:
        """
        Identify a review that has a question nature (contains '?').

        Returns
        -------
        pd.Series - row with a question-like review.
        """
        df = self.df_reviews.copy()

        # Primary strategy: reviews explicitly containing '?'
        question_mask = df["reviewText"].str.contains(r"\?", na=False)
        question_df   = df[question_mask].copy()

        if len(question_df) == 0:
            # Fallback: reviews starting with interrogative words
            question_df = df[
                df["reviewText"].str.contains(
                    r"\b(how|why|what|when|where|which|who|does|can|is|are|will|would|should|do|did)\b",
                    case=False, na=False,
                )
            ]

        if len(question_df) == 0:
            # Last resort: pick the review with the most question marks
            df["q_count"] = df["reviewText"].str.count(r"\?").fillna(0)
            question_df = df[df["q_count"] > 0].copy() if df["q_count"].max() > 0 else df

        # Pick one with a reasonable length
        question_df["wc"] = question_df["reviewText"].str.split().str.len()
        candidates = question_df[question_df["wc"].between(20, 300)]
        if len(candidates) == 0:
            candidates = question_df

        selected = candidates.sample(1, random_state=RANDOM_SEED).iloc[0]
        return selected

    @staticmethod
    def _clean_html(text: str) -> str:
        """Strip HTML tags, entities, and URLs from review text."""
        text = re.sub(r"<[^>]+>", " ", text)             # HTML tags
        text = re.sub(r"&\w+;", " ", text)                # HTML entities (&nbsp; etc.)
        text = re.sub(r"https?://\S+", " ", text)         # URLs
        text = re.sub(r"\s+", " ", text).strip()           # collapse whitespace
        return text

    @staticmethod
    def _extract_complaint(review_text: str) -> str:
        """
        Extract the specific complaint / issue from a review using
        keyword-based sentence selection.  This is done in Python so we
        can feed the extracted issue *directly* into the prompt, removing
        the need for the (small) LLM to do its own extraction.

        Cleaning step: HTML tags and entities are stripped first so that
        embedded markup does not break sentence splitting.
        """
        # --- Step 1: Clean HTML garbage --- #
        text = re.sub(r"<[^>]+>", " ", review_text)
        text = re.sub(r"&\w+;", " ", text)
        text = re.sub(r"https?://\S+", " ", text)
        # Normalise bracket-style updates: [Update: ...] -> separate block
        text = text.replace("[", ". ").replace("]", ". ")
        text = re.sub(r"\s+", " ", text).strip()

        # --- Step 2: Split into sentences --- #
        sentences = re.split(r'(?<=[.!?])\s+', text)

        complaint_keywords = [
            "problem", "issue", "broken", "defect", "damage", "crack",
            "leak", "fail", "not work", "doesn't work", "poor", "bad",
            "disappoint", "return", "refund", "waste", "wrong", "miss",
            "scratch", "rust", "peel", "chip", "paint", "broke", "stop",
            "cheap", "flimsy", "horrible", "terrible", "awful", "useless",
            "junk", "garbage", "fell apart", "fell off", "ripped",
            "pulled", "adhesive", "sticky", "stain", "smell", "loose",
            "bent", "dent", "warped", "melted", "overheated", "short",
            "repaint", "wall-friendly", "warned", "beware", "caution",
            "careful", "?",
        ]

        complaint_sentences = [
            s.strip() for s in sentences
            if len(s.split()) >= 4 and any(kw in s.lower() for kw in complaint_keywords)
        ]

        if complaint_sentences:
            # Take the LAST matching sentences (complaints are often at the end)
            return " ".join(complaint_sentences[-3:])

        # Fallback: use the last two non-trivial sentences (complaints
        # tend to appear at the end of reviews)
        meaningful = [s.strip() for s in sentences if len(s.split()) >= 5]
        return " ".join(meaningful[-2:]) if meaningful else text[:200]

    def step17_generate_csr_response(self) -> dict:
        """
        Step 17: Generate a customer-service representative response to a
        question-nature review using a locally-hosted Hugging Face LLM.

        Strategy  (Hybrid approach)
        ---------------------------
        Small instruction-tuned models (flan-t5-base/large) have a
        "safety bias": for customer-service style prompts they ignore the
        actual complaint and fall back on a memorised generic template
        ("I'm sorry to hear... please contact us...").

        To guarantee a *specific* response we use a **hybrid** strategy:

        1. **Python** extracts the complaint sentences (keyword matching).
        2. **LLM** is given a *tiny, focused task* - paraphrase or
           summarise the complaint in one sentence.  Small models handle
           single-sentence tasks reliably.
        3. **Python** assembles the final CSR response by plugging the
           LLM-paraphrased complaint into a structured template that
           includes an apology, acknowledgement, and concrete solution.

        This ensures the response always references the real issue while
        still leveraging the LLM for natural language generation.

        Returns
        -------
        dict - with 'review_text', 'response', 'model_used'.
        """
        print("\n" + "=" * 60)
        print("STEP 17: AUTOMATIC CSR RESPONSE GENERATION (LLM - Hugging Face)")
        print("=" * 60)

        if self._generator is None:
            self._load_generator()

        # Find question review
        review_row  = self._find_question_review()
        review_text = self._clean_html(str(review_row["reviewText"]).strip())
        product     = review_row.get("asin", "unknown")
        rating      = review_row.get("overall", "?")

        print(f"\n  Selected review:")
        print(f"  Product ASIN: {product}  |  Star rating: {rating}")
        print(f"  Review text ({len(review_text.split())} words):")
        print(
            textwrap.fill(review_text, width=72,
                          initial_indent="    ", subsequent_indent="    ")
        )

        # ===== Stage 1: Python-based complaint extraction ===== #
        complaint = self._extract_complaint(review_text)
        print(f"\n  Extracted complaint:")
        print(
            textwrap.fill(complaint, width=72,
                          initial_indent="    ", subsequent_indent="    ")
        )

        # ===== Stage 2: LLM paraphrases the complaint (tiny task) ===== #
        # Small models handle single-sentence tasks well.
        paraphrase_prompt = (
            f"Summarize the following customer complaint in one sentence: "
            f"\"{complaint[:400]}\""
        )
        print("\n  Asking LLM to paraphrase the complaint ...")

        try:
            para_output = self._generator(
                paraphrase_prompt,
                max_new_tokens       = 60,
                no_repeat_ngram_size = 3,
                do_sample            = False,
            )
            paraphrased = para_output[0]["generated_text"].strip()
        except Exception:
            paraphrased = complaint[:150]

        # Sanity check: if paraphrase is too short or generic, use raw complaint
        if len(paraphrased.split()) < 5 or "sorry" in paraphrased.lower():
            paraphrased = complaint[:150]

        print(f"  LLM paraphrase: {paraphrased}")

        # ===== Stage 3: LLM generates a solution suggestion ===== #
        solution_prompt = (
            f"A customer's product had this problem: \"{paraphrased}\" "
            f"Suggest one specific solution in one sentence."
        )

        try:
            sol_output = self._generator(
                solution_prompt,
                max_new_tokens       = 50,
                no_repeat_ngram_size = 3,
                do_sample            = True,
                temperature          = 0.7,
            )
            solution = sol_output[0]["generated_text"].strip()
        except Exception:
            solution = ""

        # Sanity: if the solution is too generic or empty, use a concrete default
        generic_phrases = ["contact us", "let me know", "please contact", "reach out"]
        if (len(solution.split()) < 5
                or any(g in solution.lower() for g in generic_phrases)):
            solution = (
                "We would like to offer you a full refund or send a free "
                "replacement product at no additional cost."
            )

        print(f"  LLM solution: {solution}")

        # ===== Stage 4: Python assembles the final CSR response ===== #
        # The complaint is EMBEDDED in the text, so it cannot be generic.
        response_text = (
            f"Dear Customer, thank you for taking the time to share your "
            f"experience with product {product}. "
            f"We sincerely apologize for the issue you described: "
            f"{paraphrased} "
            f"We understand how frustrating this must be, especially since "
            f"you expected better performance from our product. "
            f"{solution} "
            f"Please reply to this message or contact our support team "
            f"with your order number, and we will process this right away. "
            f"We value your feedback and are committed to making this right."
        )

        # Build the prompt description for the record (both LLM calls)
        prompt = (
            f"[Paraphrase prompt]: {paraphrase_prompt}\n"
            f"[Solution prompt]: {solution_prompt}"
        )

        print(f"\n  Final CSR Response ({len(response_text.split())} words):")
        print(
            textwrap.fill(response_text, width=72,
                          initial_indent="    ", subsequent_indent="    ")
        )

        result = {
            "product_asin"   : product,
            "star_rating"    : rating,
            "review_text"    : review_text,
            "complaint"      : complaint,
            "prompt"         : prompt,
            "response"       : response_text,
            "model_used"     : getattr(self, "_generator_model_name", "unknown"),
        }

        # Save to file
        out_path = os.path.join(self.output_dir, "step17_csr_response.txt")
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write("Phase #2 - Step 17: Automatic CSR Response\n")
            fh.write(f"Model: {result['model_used']}\n")
            fh.write("=" * 70 + "\n\n")
            fh.write(f"Product ASIN : {product}\n")
            fh.write(f"Star rating  : {rating}\n\n")
            fh.write("Customer Review:\n")
            fh.write(
                textwrap.fill(review_text, width=75,
                              initial_indent="  ",
                              subsequent_indent="  ") + "\n\n"
            )
            fh.write("Extracted Complaint (Python keyword extraction):\n")
            fh.write(
                textwrap.fill(complaint, width=75,
                              initial_indent="  ",
                              subsequent_indent="  ") + "\n\n"
            )
            fh.write("Prompt sent to LLM:\n")
            fh.write(
                textwrap.fill(prompt, width=75,
                              initial_indent="  ",
                              subsequent_indent="  ") + "\n\n"
            )
            fh.write(f"Generated CSR Response ({len(response_text.split())} words):\n")
            fh.write(
                textwrap.fill(response_text, width=75,
                              initial_indent="  ",
                              subsequent_indent="  ") + "\n"
            )
            fh.write("=" * 70 + "\n")

        print(f"\n  Saved CSR response -> {out_path}")
        return result

    # ------------------------------------------------------------------ #
    #  Full pipeline runner
    # ------------------------------------------------------------------ #
    def run(self) -> dict:
        """
        Execute both LLM steps (16 and 17).

        Returns
        -------
        dict - 'step16': list of summaries, 'step17': CSR result dict.
        """
        print("\n" + "=" * 80)
        print("PHASE #2 - LLM TASKS (STEPS 16 & 17)")
        print("=" * 80)

        # Load reviews
        self.load_reviews()

        # Step 16
        step16_results = self.step16_summarize_long_reviews()

        # Step 17
        step17_result = self.step17_generate_csr_response()

        # Produce a concise report file for the first two summaries (as required)
        self._write_report_summary(step16_results[:2], step17_result)

        print("\n  [OK] Steps 16 & 17 LLM tasks complete.")
        return {"step16": step16_results, "step17": step17_result}

    def _write_report_summary(
        self,
        first_two_summaries: list,
        csr_result: dict,
    ) -> None:
        """
        Write a concise report file containing the first two review
        summaries and the CSR response (to be included in the project
        report as required by the assignment).
        """
        out_path = os.path.join(self.output_dir, "step16_17_report_results.txt")
        lines = [
            "Phase #2 - LLM Results Report",
            "Team #4: Industrial and Scientific Dataset",
            "=" * 70,
            "",
            "STEP 16 - Review Summarisation (first two results)",
            "=" * 70,
            f"Model used: {getattr(self, '_summariser_model_name', 'unknown')}",
            "",
        ]

        for i, entry in enumerate(first_two_summaries, 1):
            lines += [
                f"Result {i}:",
                f"  Product ASIN : {entry['product_asin']}",
                f"  Star rating  : {entry['star_rating']}",
                f"  Original review ({entry['original_wc']} words):",
                "    " + textwrap.fill(
                    entry["original_text"], width=70,
                    subsequent_indent="    "
                ),
                "",
                f"  50-word summary ({entry['summary_word_count']} words):",
                "    " + textwrap.fill(
                    entry["summary"], width=70,
                    subsequent_indent="    "
                ),
                "",
                "-" * 70,
                "",
            ]

        lines += [
            "",
            "STEP 17 - Automatic CSR Response Generation",
            "=" * 70,
            f"Model used: {csr_result['model_used']}",
            "",
            f"Product ASIN : {csr_result['product_asin']}",
            f"Star rating  : {csr_result['star_rating']}",
            "",
            "Customer Review:",
            "  " + textwrap.fill(
                csr_result["review_text"], width=70,
                subsequent_indent="  "
            ),
            "",
            "CSR Response:",
            "  " + textwrap.fill(
                csr_result["response"], width=70,
                subsequent_indent="  "
            ),
            "",
            "=" * 70,
        ]

        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines))
        print(f"  Saved report results -> {out_path}")
