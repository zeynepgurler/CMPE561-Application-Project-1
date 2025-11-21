from typing import List, Set
import re
import conllu
from eval.evaluate import eval_sentence_splitting_raw

train = "../UD_Turkish-BOUN_v2.11_unrestricted-main/train-unr.conllu"
test = "../UD_Turkish-BOUN_v2.11_unrestricted-main/test-unr.conllu"


def loader(path: str, length: int):
    with open(path, "r", encoding="utf-8") as f:
        sentences = conllu.parse(f.read())

    snt_list = []
    for sent in sentences[0:length]:
        snt_list.append(sent.metadata["text"])
    return snt_list


END_PUNCT = re.compile(r"(?:\.{3}|…|[!?]{1,3}|(?<!\d)\.(?!\d))")
SINGLE_LETTER_ABBREV_RE = r"[A-Za-z]\."
EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
URL_RE = re.compile(r"(https?://[^\s'\"()\[\]\{\},;:!?]+|www\.[^\s'\"()\[\]\{\},;:!?]+)", re.IGNORECASE)
NUMERIC_RE = re.compile(r"\b\d+(?:\.\d+)+\b")  
ACRONYM_RE = re.compile(r"(?:[A-Z]\.){2,}")   
SKIP_CHARS = " \t"


ABBREV_SET = {"dr.", "sn.", "örn.", "alb.", "prof.", "bkz.", "mr.", "ms.", "vb.", "vs.", "ar-ge", "doç.", "dzl.",
              "ed.", "ekon.", "ens.", "fak.", "fel.", "fiz.", "fizy.", "gn.", "geom.", "gr.", "haz.", "hek.", "huk",
              "is.", "jeol.", "kim.", "koor.", "kr.", "krş.", "ltd.", "man.", "mat.", "mec.", "müz.", "no.", "ör.",
              "rus.", "rum.", "sf.", "sp.", "sos", "t.c.", "tar.", "tek.", "tel.", "telg.", "tic.", "tğm.", "tiy.",
              "tls.", "vd.", "vet.", "ünl.", "yy.", "zool.", "zm.", "sa.", "adr.", "alm.", "av.", "ecz.", "öğr.",
              "şti.", "a.ş.", "tıp.", "müh.", "u.s.a.", "hzn.", "ph.d.", "m.sc.", "st.", "ave."}



class RuleSentenceSplitter:
    def __init__(self, abbrev_set: Set[str] = None):
        self.abbrev_set = abbrev_set if abbrev_set else set()

    def is_abbrev(self, token: str) -> bool:
        token_lower = token.lower()
        return (token_lower in self.abbrev_set
                or re.fullmatch(SINGLE_LETTER_ABBREV_RE, token)
                or re.fullmatch(ACRONYM_RE, token))

    def split(self, doc: str) -> List[str]:
        sentences = []
        start = 0
        text = doc.replace('\n', ' ')   # normalize newlines to spaces
        placeholder_map = {}

        def placeholder(match):
            key = f"__PH{len(placeholder_map)}__"
            placeholder_map[key] = match.group()
            return key

        text = URL_RE.sub(placeholder, text)
        text = EMAIL_RE.sub(placeholder, text)
        text = NUMERIC_RE.sub(placeholder, text)

        for match in END_PUNCT.finditer(text):
            end = match.end()
            candidate = text[start:end].strip(SKIP_CHARS)
            if not candidate:
                start = end
                continue

            tokens = candidate.split()
            last_token = tokens[-1].strip("'\"“”‘’()") if tokens else ""

            if self.is_abbrev(last_token):
                continue
           
            next_char_idx = end     # Next non-space character
            while next_char_idx < len(text) and text[next_char_idx] in SKIP_CHARS:
                next_char_idx += 1
            next_char = text[next_char_idx] if next_char_idx < len(text) else ""

            punct = match.group()
            split_here = False

            if punct in ("!", "?", "!!", "???", "!?"):
                split_here = True
            elif punct in ("...", "…"):
                quote_count = candidate.count('"') + candidate.count("'")
                if quote_count % 2 == 0 and (not next_char or next_char.isupper()):
                    split_here = True
            elif punct == "." and not self.is_abbrev(last_token):
                split_here = True

            if split_here:
                end_idx = end               
                while end_idx < len(text) and text[end_idx] in "'\"“”‘’)]": # Include trailing quotes/parentheses
                    end_idx += 1

                sentence = text[start:end_idx].strip(SKIP_CHARS)
                if sentence:
                    sentences.append(sentence)
                start = end_idx

        remaining = text[start:].strip(SKIP_CHARS)  # Restore placeholders
        if remaining:
            sentences.append(remaining)

        restored = []  # Restore placeholders
        for sent in sentences:
            for key, value in placeholder_map.items():
                sent = sent.replace(key, value)
            restored.append(sent)

        return restored


gold_sents = loader(train, 100)
raw_text = " ".join(gold_sents)

splitter = RuleSentenceSplitter(ABBREV_SET)
pred_sents = splitter.split(raw_text)

precision, recall, f1 = eval_sentence_splitting_raw(pred_sents, gold_sents)
print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

#Precision: 0.8556, Recall: 0.7700, F1: 0.8105
