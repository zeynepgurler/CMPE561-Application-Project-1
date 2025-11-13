from typing import Iterable, List
from collections import Counter
import conllu
import string


def extract_tokens(path):
    with open(path, "r",encoding="utf-8") as f:
        sentences = conllu.parse(f.read())  # SentenceList      

    tokens = []                                             # token['form'] is the relevant feature

    for sentence in sentences:
        for token in sentence:
            try: 
                int(token['form'])                          # Handles number tokens
                continue
            except:
                if isinstance(token["id"], tuple):          # Handles complex tokens
                    form = token["form"]
                    tokens.append(form)
                elif token['upos'] == "AUX":
                    continue
                else:
                    if token["form"] not in string.punctuation and token["form"] not in ["...", '."']:     # Handles punctuation
                        tokens.append(token['form'].lower())
    return tokens, set(tokens)


def dynamic_stopwords(corpus_tokens: Iterable, top_k: int = 100) -> List[str]:
    """Very simple dynamic list: pick most frequent tokens as stopwords.
    Replace/extend with TF-IDF based selection later.
    """
    cnt = Counter(corpus_tokens)
    return [w for w, _ in cnt.most_common(top_k)]


def filter_dynamic(tokens: Iterable[str], dynamic_list: List[str]) -> List[str]:
    s = set(dynamic_list)
    return [t for t in tokens if t.lower() not in s]


train = "UD_Turkish-BOUN_v2.11_unrestricted-main/train-unr.conllu"
test = "UD_Turkish-BOUN_v2.11_unrestricted-main/test-unr.conllu"

tokens_tr, gold_tr = extract_tokens(train)
token_tst, gold_tst = extract_tokens(test)

print(gold_tr)
print(len(gold_tr))

dynamic = dynamic_stopwords(tokens_tr)
print(dynamic)
filtered_tokens = filter_dynamic(gold_tr, dynamic)

print(len(filtered_tokens))

for stopword in dynamic:
    print(stopword, "\t\t", stopword in filtered_tokens)

