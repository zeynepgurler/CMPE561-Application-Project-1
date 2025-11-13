from nltk.corpus import stopwords
from typing import Iterable, Set, List


#DEFAULT_STOPWORDS: Set[str] = {
#"ve", "ile", "de", "da", "bu", "şu", "o", "için", "gibi", "ama", "fakat", "veya",
#}

tr_stopwords = stopwords.words('turkish')
print(tr_stopwords[:100])

def filter_static(tokens: Iterable[str], extra: Set[str] | None = None) -> List[str]:
    sw = set(tr_stopwords)
    if extra: sw |= set(extra)
    return [t for t in tokens if t.lower() not in sw]