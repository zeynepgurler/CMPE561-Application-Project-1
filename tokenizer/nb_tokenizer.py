"""
Author: Zeynep Gürler
Date: 14.11.2025

This code includes a NaiveBayesTokenizer class, which trains a tokenizer
for Boun Treebank and ITU Web Treebank.  

"""

from __future__ import annotations

from typing import List, Tuple, Dict, Set, Optional, Iterable
from collections import Counter
import unicodedata
import math
import os


def load_suffix_lexicon(path: Optional[str]) -> Set[str]:
    """
    Load a suffix lexicon from a text file.
    If path is None or file does not exist, returns an empty set.
    """
    suffixes: Set[str] = set()
    if path is None:
        return suffixes
    if not os.path.exists(path):
        print(f"[NaiveBayesTokenizer] WARNING: suffix lexicon file not found: {path}")
        return suffixes

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            suffixes.add(line)
    return suffixes


def get_left_token(text: str, boundary_idx: int) -> str:
    """
    Given a boundary index in text, return the left token surface
    (from previous whitespace to boundary_idx)
    """
    i = boundary_idx - 1
    # Skip spaces going left
    while i >= 0 and text[i].isspace():
        i -= 1
    if i < 0:
        return ""
    end = i + 1
    # Move left until whitespace
    while i >= 0 and not text[i].isspace():
        i -= 1
    start = i + 1
    
    return text[start:end]


def get_right_token(text: str, boundary_idx: int) -> str:
    """
    Given a boundary index in text, return the right token surface
    (from boundary_idx to next whitespace)
    """
    n = len(text)
    i = boundary_idx
    # Skip spaces going right
    while i < n and text[i].isspace():
        i += 1
    if i >= n:
        return ""
    start = i
    # Move right until whitespace
    while i < n and not text[i].isspace():
        i += 1
    end = i
    
    return text[start:end]


def char_category(c: str) -> str:
    """
    Map a character to a coarse category: BOS/EOS, WS, Letter: L, Digit: D, Punctuation: P, Other: O.
    """
    if c == "<BOS>" or c == "<EOS>":
        return c
    if c.isspace():
        return "WS"
    if c.isalpha():
        return "L"
    if c.isdigit():
        return "D"
    cat = unicodedata.category(c)
    if cat.startswith("P"):
        return "P"
    return "O"


class NaiveBayesTokenizer:
    """
    Naive Bayes based token boundary classifier.

    - Training data: list of (text, gold_boundaries) pairs.
    gold_boundaries is a list of integer indices in [0, len(text)]
    where a token boundary should appear.

    - Features: character window, apostrophe/quote features,
    and optional suffix-based features.
    """

    def __init__(self, alpha: float = 1.0, suffix_lexicon_path: Optional[str] = None, use_apostrophe_features: bool = True, use_suffix_features: bool = True) -> None:
        """
            alpha:
                Additive smoothing parameter for Naive Bayes.
            suffix_lexicon_path:
                Optional path to a suffix lexicon file.
            use_apostrophe_features:
                If True, enable apostrophe/quote related features.
            use_suffix_features:
                If True, enable suffix lexicon based features.
        """
        self.alpha = alpha
        self.use_apostrophe_features = use_apostrophe_features
        self.use_suffix_features = use_suffix_features

        # Load suffix lexicon if provided
        self.suffixes: Set[str] = load_suffix_lexicon(suffix_lexicon_path)

        # feature_counts[label][feature_name] -> count
        self.feature_counts: Dict[int, Counter] = {
            0: Counter(),  # no-boundary
            1: Counter(),  # boundary
        }
        # Number of boundary/non-boundary positions
        self.class_counts: Counter = Counter()
        # Global feature vocabulary
        self.vocab: Set[str] = set()

        self.trained: bool = False
    
    def extract_boundary_features(self, text: str, boundary_idx: int) -> Dict[str, int]:
        """
        Extract features for a potential token boundary at position boundary_idx.

        Features are binary and returned as {feature_name: 1}.
        """
        feats: Dict[str, int] = {}
        n = len(text)

        # ---------- Char window ----------
        prev_c = text[boundary_idx - 1] if boundary_idx - 1 >= 0 else "<BOS>"
        next_c = text[boundary_idx] if boundary_idx < n else "<EOS>"

        feats[f"prev_char={prev_c}"] = 1
        feats[f"next_char={next_c}"] = 1

        prev_cat = char_category(prev_c)
        next_cat = char_category(next_c)
        feats[f"prev_cat={prev_cat}"] = 1
        feats[f"next_cat={next_cat}"] = 1

        # bigram of surrounding chars
        feats[f"prev_next={prev_c}|{next_c}"] = 1

        # Apostrophe/quote/punctuation features 
        if self.use_apostrophe_features:
            prev_is_apo = prev_c in {"'", "’", "‘"}
            next_is_apo = next_c in {"'", "’", "‘"}
            prev_is_quote = prev_c in {'"', "“", "”"}
            next_is_quote = next_c in {'"', "“", "”"}

            if prev_is_apo:
                feats["prev_is_apostrophe"] = 1
            if next_is_apo:
                feats["next_is_apostrophe"] = 1
            if prev_is_quote:
                feats["prev_is_quote"] = 1
            if next_is_quote:
                feats["next_is_quote"] = 1

            # Proper-name pattern around apostrophe:
            # Capitalized word + ' + suffix
            left_tok = get_left_token(text, boundary_idx)
            right_tok = get_right_token(text, boundary_idx)

            if left_tok and right_tok:
                has_capital_start = left_tok[:1].isupper()
                window = text[max(0, boundary_idx - 2): boundary_idx + 2]
                if has_capital_start and ("'" in window or "’" in window):
                    feats["around_apostrophe_after_capital"] = 1

            # Quote directly before a capitalized word:
            # e.g. "Hayır
            if prev_is_quote and right_tok and right_tok[:1].isupper():
                feats["quote_before_capital"] = 1

        # Suffix-based features 
        if self.use_suffix_features and self.suffixes:
            left_tok = get_left_token(text, boundary_idx)
            if left_tok:
                lower_left = left_tok.lower()
                max_suf_len = max(len(s) for s in self.suffixes) if self.suffixes else 0

                # check for explicit suffix matches
                for l in range(2, max_suf_len + 1):
                    if len(lower_left) < l:
                        continue
                    cand = lower_left[-l:]
                    if cand in self.suffixes:
                        feats[f"left_endswith_suf={cand}"] = 1

                # back-off patterns
                if lower_left.endswith(("dir", "dır", "dur", "dür")):
                    feats["left_endswith_dir_variant"] = 1
                if lower_left.endswith(("dı", "di", "du", "dü")):
                    feats["left_endswith_past_d"] = 1
                if lower_left.endswith(("tı", "ti", "tu", "tü")):
                    feats["left_endswith_past_t"] = 1
                if lower_left.endswith("yor"):
                    feats["left_endswith_yor"] = 1
            if right_tok:
                lower_right = right_tok.lower()
                if lower_right in self.suffixes:
                    feats[f"right_is_suffix={lower_right}"] = 1

        return feats

    def log_prob(self, label: int, feats: Dict[str, int]) -> float:
        """
        Compute log P(label | feats).
        Naive Bayes with multinomial features and additive smoothing.
        """
        if not self.trained:
            raise RuntimeError("NaiveBayesTokenizer is not trained yet.")

        total_feat_count = sum(self.feature_counts[label].values())
        V = len(self.vocab)
        total_classes = sum(self.class_counts.values())

        # log prior
        logp = math.log(
            (self.class_counts[label] + self.alpha) /
            (total_classes + 2 * self.alpha)
        )

        # log likelihood factors
        for f_name in feats.keys():
            c = self.feature_counts[label][f_name]
            logp += math.log(
                (c + self.alpha) /
                (total_feat_count + self.alpha * V)
            )

        return logp

    def fit(self, instances: Iterable[Tuple[str, List[int]]]) -> None:
        """
        Train the Naive Bayes model.

        Args:
            instances:
                Iterable of (text, gold_boundaries) pairs.
                gold_boundaries is a list of ints where 
                token boundaries should be placed.
        """
        for text, gold_boundaries in instances:
            n = len(text)
            gold_set = set(gold_boundaries)

            # iterate over all possible boundary positions
            for i in range(n + 1):
                # you can optionally skip BOS/EOS here if you want
                label = 1 if i in gold_set else 0
                feats = self.extract_boundary_features(text, i)

                self.class_counts[label] += 1
                for f_name in feats.keys():
                    self.feature_counts[label][f_name] += 1
                    self.vocab.add(f_name)

        self.trained = True

    def predict_boundaries(self, text: str) -> List[int]:
        """
        Predict token boundary positions for a given text.
        Returns a list of indices in [0, len(text)].
        """
        n = len(text)
        boundaries: List[int] = []

        for i in range(n + 1):
            feats = self.extract_boundary_features(text, i)
            logp_boundary = self.log_prob(1, feats)
            logp_noboundary = self.log_prob(0, feats)

            if logp_boundary >= logp_noboundary:
                boundaries.append(i)

        return boundaries

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize a string based on predicted boundaries.
        """
        boundaries = self.predict_boundaries(text)
        tokens: List[str] = []
        prev = 0
        for b in boundaries:
            if b <= prev:
                continue
            span = text[prev:b]

            # Remove whitespace from the span
            cleaned = span.strip()

            if cleaned: 
                tokens.append(cleaned)
            prev = b
        return tokens
