from typing import Iterable, Set, List


DEFAULT_STOPWORDS: Set[str] = {
"ve", "ile", "de", "da", "bu", "şu", "o", "için", "gibi", "ama", "fakat", "veya",
}


def filter_static(tokens: Iterable[str], extra: Set[str] | None = None) -> List[str]:
    sw = set(DEFAULT_STOPWORDS)
    if extra: sw |= set(extra)
    return [t for t in tokens if t.lower() not in sw]