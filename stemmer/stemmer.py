from typing import List
from ..utils.io import read_tsv


class SimpleTurkishStemmer:
    """Heuristic stemmer using longest-suffix stripping with a suffix lexicon.
    Handles both inflectional and a subset of derivational suffixes.
    """
    def __init__(self, suffix_path: str):
        rows = read_tsv(suffix_path)
        self.suffixes = sorted([r[1] for r in rows if len(r) >= 2], key=len, reverse=True)


    def stem(self, token: str) -> str:
        t = token
        for suf in self.suffixes:
            if t.endswith(suf) and len(t) - len(suf) >= 2:
                t = t[: -len(suf)]
                break
        return t


    def stem_sentence(self, tokens: List[str]) -> List[str]:
        return [self.stem(t) for t in tokens]