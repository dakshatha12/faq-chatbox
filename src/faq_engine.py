"""
faq_engine.py
=============
Core NLP engine for FAQ matching.

Pipeline:
  1. Load FAQs from JSON
  2. Preprocess text  (tokenize → lowercase → remove stopwords → lemmatize)
  3. Build TF-IDF matrix over FAQ questions
  4. At query time: preprocess query → cosine-similarity → return best match
"""

import json
import re
import string
import logging
from pathlib import Path

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ── ensure NLTK assets are present ────────────────────────────────────────────
for resource in ("punkt", "stopwords", "wordnet", "punkt_tab", "omw-1.4"):
    try:
        nltk.data.find(f"tokenizers/{resource}" if resource.startswith("punkt") else f"corpora/{resource}")
    except LookupError:
        nltk.download(resource, quiet=True)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


class FAQEngine:
    """Loads FAQs, preprocesses them, and answers user queries via cosine similarity."""

    CONFIDENCE_THRESHOLD = 0.15   # below this → no good match found

    def __init__(self, faq_path: str | Path):
        self.faq_path = Path(faq_path)
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words("english"))
        # Keep question-words — they matter for intent
        self.stop_words -= {"what", "when", "where", "who", "how", "why", "which", "can", "do", "is", "are"}

        self.faqs: list[dict] = []
        self.vectorizer: TfidfVectorizer | None = None
        self.tfidf_matrix = None

        self._load_and_build()

    # ── public API ─────────────────────────────────────────────────────────────

    def get_answer(self, user_query: str) -> dict:
        """
        Returns a dict:
          {
            "answer": str,
            "question": str,       # matched FAQ question
            "category": str,
            "confidence": float,   # 0.0 – 1.0
            "matched": bool
          }
        """
        if not user_query.strip():
            return self._no_match("Please type a question.")

        cleaned = self.preprocess(user_query)
        if not cleaned:
            return self._no_match("I didn't understand that. Could you rephrase?")

        query_vec = self.vectorizer.transform([cleaned])
        sims = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        best_idx = int(np.argmax(sims))
        score = float(sims[best_idx])

        logger.info("Query: %r  →  score=%.3f  idx=%d", user_query, score, best_idx)

        if score < self.CONFIDENCE_THRESHOLD:
            return self._no_match(
                "I'm not sure about that. Could you rephrase, or try asking about "
                "orders, payments, shipping, returns, or your account?"
            )

        faq = self.faqs[best_idx]
        return {
            "answer": faq["answer"],
            "question": faq["question"],
            "category": faq.get("category", "General"),
            "confidence": round(score, 3),
            "matched": True,
        }

    def get_categories(self) -> list[str]:
        """Return unique FAQ categories."""
        seen = []
        for f in self.faqs:
            c = f.get("category", "General")
            if c not in seen:
                seen.append(c)
        return seen

    def get_faqs_by_category(self, category: str) -> list[dict]:
        return [f for f in self.faqs if f.get("category") == category]

    def all_questions(self) -> list[str]:
        return [f["question"] for f in self.faqs]

    # ── preprocessing ──────────────────────────────────────────────────────────

    def preprocess(self, text: str) -> str:
        """
        Full NLP preprocessing pipeline:
          lowercase → expand contractions → strip punctuation →
          tokenize → remove stopwords → lemmatize → rejoin
        """
        text = text.lower()
        text = self._expand_contractions(text)
        text = re.sub(r"[^a-z0-9\s]", " ", text)   # remove punctuation
        text = re.sub(r"\s+", " ", text).strip()

        tokens = word_tokenize(text)
        tokens = [t for t in tokens if t not in self.stop_words and len(t) > 1]
        tokens = [self.lemmatizer.lemmatize(t) for t in tokens]

        return " ".join(tokens)

    # ── internal helpers ───────────────────────────────────────────────────────

    def _load_and_build(self):
        """Load JSON, preprocess questions, and fit TF-IDF vectorizer."""
        with open(self.faq_path, encoding="utf-8") as f:
            self.faqs = json.load(f)
        logger.info("Loaded %d FAQs from %s", len(self.faqs), self.faq_path)

        processed_questions = [self.preprocess(faq["question"]) for faq in self.faqs]

        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),      # unigrams + bigrams
            min_df=1,
            sublinear_tf=True,       # log(tf) smoothing
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(processed_questions)
        logger.info("TF-IDF matrix built: shape=%s", self.tfidf_matrix.shape)

    def _no_match(self, message: str) -> dict:
        return {
            "answer": message,
            "question": "",
            "category": "",
            "confidence": 0.0,
            "matched": False,
        }

    @staticmethod
    def _expand_contractions(text: str) -> str:
        contractions = {
            "can't": "cannot", "won't": "will not", "don't": "do not",
            "i'm": "i am", "i've": "i have", "it's": "it is",
            "what's": "what is", "where's": "where is", "how's": "how is",
            "that's": "that is", "there's": "there is", "isn't": "is not",
            "aren't": "are not", "wasn't": "was not", "weren't": "were not",
            "haven't": "have not", "hasn't": "has not", "didn't": "did not",
            "doesn't": "does not", "wouldn't": "would not", "couldn't": "could not",
            "shouldn't": "should not", "i'll": "i will", "you'll": "you will",
            "he'll": "he will", "she'll": "she will", "we'll": "we will",
            "they'll": "they will", "i'd": "i would", "you'd": "you would",
        }
        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)
        return text


# ── quick smoke-test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    engine = FAQEngine(Path(__file__).parent.parent / "data" / "faqs.json")

    test_queries = [
        "forgot my password",
        "where is my package?",
        "can I return the product",
        "do you take UPI payments",
        "I got a broken item",
        "xyz random gibberish question",
    ]

    print("\n" + "=" * 60)
    for q in test_queries:
        result = engine.get_answer(q)
        print(f"\nQ: {q}")
        print(f"   Matched: {result['matched']}  |  Confidence: {result['confidence']}")
        if result["matched"]:
            print(f"   FAQ:  {result['question']}")
        print(f"   Ans:  {result['answer'][:80]}...")
    print("=" * 60)
