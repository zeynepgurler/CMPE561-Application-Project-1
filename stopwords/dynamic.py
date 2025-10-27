from typing import Iterable, List, Tuple
from collections import Counter


def dynamic_stopwords(corpus_tokens: Iterable[Iterable[str]], top_k: int = 100) -> List[str]:
    """Very simple dynamic list: pick most frequent tokens as stopwords.
    Replace/extend with TF-IDF based selection later.
    """
    cnt = Counter()
    for toks in corpus_tokens:
        cnt.update([t.lower() for t in toks])
    return [w for w, _ in cnt.most_common(top_k)]


def filter_dynamic(tokens: Iterable[str], dynamic_list: List[str]) -> List[str]:
    s = set(dynamic_list)
    return [t for t in tokens if t.lower() not in s]