from typing import List
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from ..features.sentence_features import sentence_boundary_features


class LRSentenceSplitter:
    def __init__(self):
        self.vec = DictVectorizer(sparse=True)
        self.clf = LogisticRegression(max_iter=500)
        self.fitted = False


    def _samples(self, tokens: List[str], gold_boundaries: List[int]):
        X, y = [], []
        for i in range(len(tokens)):
            feats = sentence_boundary_features(tokens, i)
            X.append(feats)
            y.append(1 if i in gold_boundaries else 0)
        return X, y


    def fit(self, token_sequences: List[List[str]], gold_boundaries_list: List[List[int]]):
        X_all, y_all = [], []
        for toks, bounds in zip(token_sequences, gold_boundaries_list):
            X, y = self._samples(toks, bounds)
            X_all.extend(X); y_all.extend(y)
        Xv = self.vec.fit_transform(X_all)
        self.clf.fit(Xv, y_all)
        self.fitted = True
        return self


    def split(self, tokens: List[str]) -> List[List[str]]:
        assert self.fitted, "Call fit() before split()"
        X = [sentence_boundary_features(tokens, i) for i in range(len(tokens))]
        Xv = self.vec.transform(X)
        preds = self.clf.predict(Xv)
        sents, cur = [], []
        for i, tok in enumerate(tokens):
            cur.append(tok)
            if preds[i] == 1:
                sents.append(cur); cur = []
        if cur: sents.append(cur)
        return sents