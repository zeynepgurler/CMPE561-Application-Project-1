from typing import List
from ..utils.text import is_abbrev


END_PUNCT = {".", "?", "!"}


class RuleSentenceSplitter:
    def split(self, tokens: List[str]) -> List[List[str]]:
        sents, cur = [], []
        for i, tok in enumerate(tokens):
            cur.append(tok)
            if tok[-1:] in END_PUNCT and not is_abbrev(tok.lower()):
                sents.append(cur); cur = []
        if cur: sents.append(cur)
        return sents