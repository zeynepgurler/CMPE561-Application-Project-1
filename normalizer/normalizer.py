from typing import Dict
import unicodedata
import regex as re
from ..utils.io import read_tsv


class Normalizer:
    """Rule + lexicon based normalizer for Turkish.
    Steps:
    - NFKC normalize
    - Lowercase with Turkish rules
    - Replace via normalization lexicon
    - Standardize urls, emails, numbers
    """
    def __init__(self, lexicon_path: str):
        self.lexicon = self._load_lexicon(lexicon_path)
        self.url_token = "<URL>"
        self.email_token = "<EMAIL>"
        self.number_token = "<NUM>"
        self.url_re = re.compile(r"https?://\S+")
        self.email_re = re.compile(r"[\w.\-+]+@[\w\-]+\.[A-Za-z]{2,}")
        self.num_re = re.compile(r"(?<!\p{L})[0-9]+([.,][0-9]+)?(?!\p{L})")

    def _load_lexicon(self, path: str) -> Dict[str, str]:
        pairs = read_tsv(path)
        return {src.strip(): tgt.strip() for src, tgt in pairs}


    def _tr_lower(self, s: str) -> str:
        return s.replace("I", "ı").replace("İ", "i").lower()

    def normalize(self, text: str) -> str:
        s = unicodedata.normalize("NFKC", text)
        s = self.url_re.sub(self.url_token, s)
        s = self.email_re.sub(self.email_token, s)
        s = self.num_re.sub(self.number_token, s)
        toks = s.split()
        normed = [self.lexicon.get(self._tr_lower(t), t) for t in toks]
        return " ".join(normed)