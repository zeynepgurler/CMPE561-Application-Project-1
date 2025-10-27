import regex as re
from ..utils.text import URL_RE, EMAIL_RE, HASHTAG_RE, MENTION_RE


class RuleTokenizer:
    URL_TOK = "<URL>"; EMAIL_TOK = "<EMAIL>"


    def __init__(self):
        self.pattern = re.compile(
            r"(" +
            r"https?://\S+|" +
            r"[\w.\-+]+@[\w\-]+\.[A-Za-z]{2,}|" +
            r"#[\p{L}0-9_]+|@#[\p{L}0-9_]+|" +
            r"\p{L}+|" +
            r"[0-9]+|" +
            r"[\p{Punct}]" +
            r")",
            re.UNICODE
        )

    def tokenize(self, text: str) -> list[str]:
        toks = [m.group(0) for m in self.pattern.finditer(text)]
        return toks