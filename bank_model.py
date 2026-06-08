import math
import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


CACHE_PATH = Path("banking77_model_cache.pkl")
SPLITS = {
    "train": "data/train-00000-of-00001.parquet",
    "test": "data/test-00000-of-00001.parquet",
}
DATASET_ROOT = "hf://datasets/mteb/banking77/"


def lemmatize(word):
    word = word.lower()
    irregulars = {
        "spent": "spend",
        "withdrew": "withdraw",
        "withdrawn": "withdraw",
        "paid": "pay",
        "frozen": "freeze",
        "froze": "freeze",
        "lost": "lose",
        "stolen": "steal",
        "taken": "take",
        "took": "take",
        "held": "hold",
        "broken": "break",
        "swiped": "swipe",
        "dealt": "deal",
        "chosen": "choose",
        "bought": "buy",
        "sold": "sell",
        "sent": "send",
        "lent": "lend",
        "borrowed": "borrow",
        "kept": "keep",
        "found": "find",
        "spoken": "speak",
        "told": "tell",
        "meant": "mean",
        "understood": "understand",
        "thought": "think",
        "caught": "catch",
        "fed": "feed",
        "written": "write",
        "am": "be",
        "is": "be",
        "are": "be",
        "was": "be",
        "were": "be",
        "has": "have",
        "had": "have",
        "does": "do",
        "did": "do",
        "goes": "go",
        "went": "go",
        "declined": "decline",
        "charged": "charge",
        "activated": "activate",
        "better": "good",
        "best": "good",
        "worse": "bad",
        "worst": "bad",
        "more": "much",
        "most": "much",
        "less": "little",
        "least": "little",
        "farther": "far",
        "farthest": "far",
        "further": "far",
        "furthest": "far",
        "earlier": "early",
        "earliest": "early",
        "later": "late",
        "latest": "late",
        "costlier": "costly",
        "costliest": "costly",
        "easier": "easy",
        "easiest": "easy",
        "busier": "busy",
        "busiest": "busy",
        "trustier": "trusty",
        "trustiest": "trusty",
        "speedier": "speedy",
        "speediest": "speedy",
    }

    if word in irregulars:
        return irregulars[word]
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("es") and (
        word.endswith("ses")
        or word.endswith("xes")
        or word.endswith("zes")
        or word.endswith("ches")
        or word.endswith("shes")
    ):
        return word[:-2]
    if word.endswith("s") and len(word) > 3 and not word.endswith("ss"):
        return word[:-1]
    if word.endswith("ing") and len(word) > 5:
        return word[:-3]
    if word.endswith("ed") and len(word) > 4:
        return word[:-2]
    return word


def preprocess_advanced(text, use_bigrams=True):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = [lemmatize(token) for token in text.split()]
    features = list(tokens)

    if use_bigrams:
        for i in range(len(tokens) - 1):
            features.append(f"{tokens[i]}_{tokens[i + 1]}")

    return features


class BankingIntentModel:
    def __init__(
        self,
        idf_dict,
        vocabulary,
        word_weights,
        class_totals,
        class_docs,
        total_docs,
        label_names,
        train_shape,
        test_shape,
    ):
        self.idf_dict = idf_dict
        self.vocabulary = vocabulary
        self.word_weights = word_weights
        self.class_totals = class_totals
        self.class_docs = class_docs
        self.total_docs = total_docs
        self.label_names = label_names
        self.train_shape = train_shape
        self.test_shape = test_shape

    @property
    def vocab_size(self):
        return len(self.vocabulary)

    def score(self, tokens):
        if not tokens:
            return []

        tf_dict = Counter(tokens)
        doc_len = len(tokens)
        query_tfidf = {}
        for word, count in tf_dict.items():
            tf = count / float(doc_len)
            query_tfidf[word] = tf * self.idf_dict.get(word, 0)

        scores = []
        for label in self.class_docs:
            log_prior = math.log(self.class_docs[label] / self.total_docs)
            log_likelihood = 0.0

            for word, query_weight in query_tfidf.items():
                if word in self.vocabulary:
                    numerator = self.word_weights.get(label, {}).get(word, 0.0) + 0.5
                    denominator = self.class_totals[label] + 0.5 * self.vocab_size
                    log_likelihood += query_weight * math.log(numerator / denominator)

            scores.append((label, log_prior + log_likelihood))

        return sorted(scores, key=lambda item: item[1], reverse=True)

    def predict(self, text, top_k=5):
        tokens = preprocess_advanced(text)
        scores = self.score(tokens)
        if not scores:
            return {
                "query": text,
                "tokens": tokens,
                "prediction": None,
                "top_intents": [],
            }

        max_score = scores[0][1]
        exp_scores = [math.exp(score - max_score) for _, score in scores[:top_k]]
        total = sum(exp_scores) or 1.0

        top_intents = []
        for (label, score), exp_score in zip(scores[:top_k], exp_scores):
            top_intents.append(
                {
                    "label": int(label),
                    "label_text": self.label_names.get(int(label), f"label_{label}"),
                    "confidence": exp_score / total,
                    "score": score,
                }
            )

        return {
            "query": text,
            "tokens": tokens,
            "prediction": top_intents[0],
            "top_intents": top_intents,
        }


def train_model():
    df_train = pd.read_parquet(DATASET_ROOT + SPLITS["train"])
    df_test = pd.read_parquet(DATASET_ROOT + SPLITS["test"])

    train_data = [
        (preprocess_advanced(row["text"]), int(row["label"]))
        for _, row in df_train.iterrows()
    ]

    num_documents = len(train_data)
    doc_freq = Counter()
    for tokens, _ in train_data:
        for word in set(tokens):
            doc_freq[word] += 1

    idf_dict = defaultdict(float)
    for word, count in doc_freq.items():
        idf_dict[word] = math.log(num_documents / float(count))

    vocabulary = set()
    word_weights = defaultdict(lambda: defaultdict(float))
    class_totals = Counter()
    class_docs = Counter()

    for tokens, label in train_data:
        class_docs[label] += 1
        tf_dict = Counter(tokens)
        doc_len = len(tokens)

        for word, count in tf_dict.items():
            tfidf_score = (count / float(doc_len)) * idf_dict.get(word, 0)
            word_weights[label][word] += tfidf_score
            class_totals[label] += tfidf_score
            vocabulary.add(word)

    label_names = (
        df_train[["label", "label_text"]]
        .drop_duplicates()
        .sort_values("label")
        .set_index("label")["label_text"]
        .to_dict()
    )

    return BankingIntentModel(
        idf_dict=dict(idf_dict),
        vocabulary=vocabulary,
        word_weights={label: dict(words) for label, words in word_weights.items()},
        class_totals=class_totals,
        class_docs=class_docs,
        total_docs=len(train_data),
        label_names={int(label): name for label, name in label_names.items()},
        train_shape=tuple(df_train.shape),
        test_shape=tuple(df_test.shape),
    )


def load_model():
    if CACHE_PATH.exists():
        with CACHE_PATH.open("rb") as handle:
            return pickle.load(handle)

    model = train_model()
    with CACHE_PATH.open("wb") as handle:
        pickle.dump(model, handle)
    return model
