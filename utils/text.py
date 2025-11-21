import regex as re


URL_RE = re.compile(r"https?://\S+")
EMAIL_RE = re.compile(r"[\w.\-+]+@[\w\-]+\.[A-Za-z]{2,}")
HASHTAG_RE = re.compile(r"#[\p{L}0-9_]+", re.UNICODE)
MENTION_RE = re.compile(r"@[\p{L}0-9_]+", re.UNICODE)
WHITESPACE_RE = re.compile(r"\s+")

