from typing import List, Dict, Tuple
from collections import Counter, defaultdict
import math
from ..features.token_features import token_boundary_features


class NaiveBayesTokenizer:
    """Character-boundary model: decide if a boundary at each position.
    Train from text + gold tokenization by converting to boundary labels.
    """
    def __init__(self):
        self.feature_counts = defaultdict(lambda: Counter()) # feat->class->count
        self.class_counts = Counter()
        self.classes = [0, 1] # 1 = boundary, 0 = no-boundary
        self.vocab = set()
        self.alpha = 1.0

    def _extract_samples(self, text: str, tokens: List[str]) -> List[Tuple[int, Dict[str, int]]]:
        # Convert gold tokens to boundary indices
        idx = 0
        boundaries = set()
        for tok in tokens:
            idx += len(tok)
            boundaries.add(idx)
            idx += 1 # for the single space between tokens in detok; adjust if needed
        samples = []
        for i in range(len(text)):
            feats = token_boundary_features(text, i)
            y = 1 if i in boundaries else 0
            samples.append((y, feats))
        return samples


    def fit(self, texts: List[str], gold_tokens: List[List[str]]):
        for text, toks in zip(texts, gold_tokens):
            for y, feats in self._extract_samples(text, toks):
                self.class_counts[y] += 1
                for f in feats:
                    self.feature_counts[f][y] += 1
                    self.vocab.add(f)
        return self


    def _logprob(self, feats: Dict[str, int], y: int) -> float:
        total_y = sum(fc[y] for fc in self.feature_counts.values())
        logp = math.log((self.class_counts[y] + self.alpha) / (sum(self.class_counts.values()) + 2*self.alpha))
        for f in feats:
            c = self.feature_counts[f][y]
            logp += math.log((c + self.alpha) / (total_y + self.alpha * (len(self.vocab) or 1)))
        return logp


    def tokenize(self, text: str) -> List[str]:
        # Greedy: split where P(y=1) > P(y=0)
        parts, cur = [], []
        for i, ch in enumerate(text):
            cur.append(ch)
            feats = token_boundary_features(text, i)
            if self._logprob(feats, 1) > self._logprob(feats, 0):
                parts.append("".join(cur).strip())
                cur = []
        if cur:
            parts.append("".join(cur).strip())
        return [p for p in parts if p]