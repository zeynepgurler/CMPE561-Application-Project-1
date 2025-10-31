"""
Author: Zeynep Gürler
Date: 31.10.2025

This code is for co-loading BOUN Treebank and ITU Web Treebank. These datasets have different formats. 
UnifiedBounItuTreebankLoader example:
        Domain     : boun
        SentenceID : None
        RAW        : 1936 yılındayız.
        GOLD       : 1936 yılındayız.
        TOKENS     : ['1936', 'yılında', 'yız', '.']
"""


from typing import List, Dict, Iterable, Optional, Tuple
from dataclasses import dataclass
from itertools import islice
import regex as re


# ----------------------------
# Utils
# ----------------------------

# Minimal TR-friendly detokenizer (useful for UD/BOUN where punctuation is split)
_PUNCT_NO_SPACE_BEFORE = set(".,;:!?%)”’›»…")
_PUNCT_NO_SPACE_AFTER  = set("(%“‘‹«")

def detok(tokens: List[str]) -> str:
    """Reconstruct a human-readable sentence from UD-style tokens."""
    if not tokens:
        return ""
    out: List[str] = [tokens[0]]
    for t in tokens[1:]:
        prev = out[-1]
        if t.startswith(("'", "’")) or t in _PUNCT_NO_SPACE_BEFORE:
            out[-1] = prev + t
        elif prev in _PUNCT_NO_SPACE_AFTER:
            out[-1] = prev + t
        else:
            out.append(" " + t)
    return "".join(out)

def join_simple(tokens: List[str]) -> str:
    """Simple join for ITU Web where apostrophe forms are already single tokens."""
    return " ".join(tokens)

# Consistent masking so URL/email/number differences don't dominate evaluation.
RE_URL   = re.compile(r"(?i)\bhttps?://\S+")
RE_EMAIL = re.compile(r"(?i)\b[\w.\-+%]+@[\w.\-]+\.[A-Za-z]{2,}\b")
RE_NUM   = re.compile(r"(?<!\p{L})\d+(?:[.,]\d+)?(?!\p{L})")

def mask_specials(s: str) -> str:
    s = RE_URL.sub("<URL>", s)
    s = RE_EMAIL.sub("<EMAIL>", s)
    s = RE_NUM.sub("<NUM>", s)
    return s


# ----------------------------
# Data container
# ----------------------------
@dataclass
class Example:
    raw_text: str           # input string for the normalizer (BOUN: raw == gold)
    gold_text: str          # gold-normalized sentence (ITU GOLD, BOUN: same as raw)
    gold_tokens: List[str]  # gold token sequence
    domain: str             # "boun" | "iwt"
    sent_id: Optional[str]  # sentence id if available


# ----------------------------
# Unified Loader
# ----------------------------
class UnifiedBounItuTreebankLoader:
    """
    Add sources with add_boun(...) or add_iwt(...).
    Iterate over all examples with iterate().
    """
    def __init__(self, boun_detok: bool = True):
        self.sources: List[Tuple[str, str]] = []
        self.boun_detok = boun_detok

    def add_boun(self, conllu_path: str) -> None:
        """Register a BOUN UD-CoNLL-U file."""
        self.sources.append(("boun", conllu_path))

    def add_iwt(self, with_sentence_begin_path: str) -> None:
        """Register an ITU Web file in '...withSentenceBegin' format."""
        self.sources.append(("iwt", with_sentence_begin_path))

    def iterate(self) -> Iterable[Example]:
        """Yield unified Example items from all registered sources."""
        for stype, path in self.sources:
            if stype == "boun":
                yield from self.read_boun_conllu(path)
            else:
                yield from self.read_iwt_with_sentence_begin(path)

    # --------- BOUN (UD CoNLL-U) ----------
    def read_boun_conllu(self, path: str) -> Iterable[Example]:
        """
        CoNLL-U reader (no external parser).
        - Skips multi-word token ranges (e.g., '3-4') and empty nodes ('2.1').
        - Reconstructs sentence text from tokens if # text is missing.
        """
        sent_meta: Dict[str, str] = {}
        gold_tokens: List[str] = []

        with open(path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.rstrip("\n")
                if not line:
                    # end of sentence
                    if gold_tokens:
                        txt = sent_meta.get("text")
                        if not txt:
                            txt = detok(gold_tokens) if self.boun_detok else " ".join(gold_tokens)
                        yield Example(
                            raw_text=txt,
                            gold_text=txt,          # BOUN has no separate gold normalization
                            gold_tokens=gold_tokens[:],
                            domain="boun",
                            sent_id=sent_meta.get("sent_id"),
                        )
                        gold_tokens.clear()
                        sent_meta.clear()
                    continue

                if line.startswith("#"):
                    # metadata lines like: "# sent_id = ..." or "# text = ..."
                    if " = " in line:
                        k, v = line[2:].split(" = ", 1)
                        sent_meta[k.strip()] = v.strip()
                    continue

                parts = line.split("\t")
                if len(parts) < 2:
                    continue
                tok_id = parts[0]
                if "-" in tok_id or "." in tok_id:
                    # skip multi-word ranges and empty nodes
                    continue
                form = parts[1]
                gold_tokens.append(form)

            # EOF flush
            if gold_tokens:
                txt = sent_meta.get("text")
                if not txt:
                    txt = detok(gold_tokens) if self.boun_detok else " ".join(gold_tokens)
                yield Example(
                    raw_text=txt,
                    gold_text=txt,
                    gold_tokens=gold_tokens[:],
                    domain="boun",
                    sent_id=sent_meta.get("sent_id"),
                )

    # --------- ITU Web (withSentenceBegin) ----------
    def read_iwt_with_sentence_begin(self, path: str) -> Iterable[Example]:
        """
        Reads lines where each sentence is formed by rows:
          - "<sentBegin>\\tRAW\\tGOLD" starts a new sentence,
          - "RAW\\tGOLD" continues the sentence.
        Blank lines also finalize a sentence.
        """
        raw_sent: List[str] = []
        gold_sent: List[str] = []

        with open(path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.rstrip("\n")
                if not line or line.startswith("#"):
                    # finalize current sentence on blank/comment
                    if raw_sent:
                        yield Example(
                            raw_text=join_simple(raw_sent),
                            gold_text=join_simple(gold_sent),
                            gold_tokens=gold_sent[:],
                            domain="iwt",
                            sent_id=None,
                        )
                        raw_sent.clear()
                        gold_sent.clear()
                    continue

                parts = line.split("\t")
                if len(parts) == 1:
                    parts = re.split(r"\s+", line.strip())
                parts = [p for p in parts if p != ""]
                if not parts:
                    continue

                if len(parts) == 3 and parts[0] == "<sentBegin>":
                    # start a new sentence
                    if raw_sent:
                        yield Example(
                            raw_text=join_simple(raw_sent),
                            gold_text=join_simple(gold_sent),
                            gold_tokens=gold_sent[:],
                            domain="iwt",
                            sent_id=None,
                        )
                        raw_sent.clear()
                        gold_sent.clear()
                    raw_tok, gold_tok = parts[1], parts[2]
                    raw_sent.append(raw_tok)
                    gold_sent.append(gold_tok)
                elif len(parts) >= 2:
                    raw_tok, gold_tok = parts[-2], parts[-1]  # be tolerant to extra cols
                    raw_sent.append(raw_tok)
                    gold_sent.append(gold_tok)

            # EOF flush
            if raw_sent:
                yield Example(
                    raw_text=join_simple(raw_sent),
                    gold_text=join_simple(gold_sent),
                    gold_tokens=gold_sent[:],
                    domain="iwt",
                    sent_id=None,
                )


# ----------------------------
# Minimal evaluator (optional)
# ----------------------------
def evaluate_normalizer(normalize_fn, examples: Iterable[Example], mask: bool = True) -> Dict[str, float]:
    """
    Simple evaluation:
      - sentence exact match (string level)
      - token accuracy (by whitespace split; good enough for ITU where apostrophes are single tokens)
    """
    total_sent = 0
    exact_sent = 0
    total_tok = 0
    correct_tok = 0

    for ex in examples:
        pred = normalize_fn(ex.raw_text)
        gold = ex.gold_text
        if mask:
            pred = mask_specials(pred)
            gold = mask_specials(gold)

        if pred == gold:
            exact_sent += 1
        total_sent += 1

        pred_toks = pred.split()
        gold_toks = gold.split()
        m = min(len(pred_toks), len(gold_toks))
        correct_tok += sum(1 for i in range(m) if pred_toks[i] == gold_toks[i])
        total_tok += len(gold_toks)

    return {
        "sent_exact": (exact_sent / total_sent) if total_sent else 0.0,
        "token_acc": (correct_tok / total_tok) if total_tok else 0.0,
    }

def show_examples(ex_iter, n=3, title=""):
    print(f"\n*** {title} (first {n}) ***\n")
    for ex in islice(ex_iter, n):
        print("Domain     :", ex.domain)
        print("SentenceID :", ex.sent_id)
        print("RAW        :", ex.raw_text)
        print("GOLD       :", ex.gold_text)
        print("TOKENS     :", repr(ex.gold_tokens))  # repr -> satırların birleşmesini engeller
        print("-" * 60)

def main():
    ul = UnifiedBounItuTreebankLoader(boun_detok=True)

    # ---- YOUR PATHS ----
    # BOUN (UD CoNLL-U)
    ul.add_boun("../UD_Turkish-BOUN-master/tr_boun-ud-train.conllu")
    # ITU Web (RAW→GOLD withSentenceBegin)
    ul.add_iwt("../IWTandTestSmall/IWT_normalizationerrorsNoUpperCase.withSentenceBegin")

    # Split by domain using single pass
    boun_buf = []
    iwt_buf = []

    for ex in ul.iterate():
        if ex.domain == "boun" and len(boun_buf) < 3:
            boun_buf.append(ex)
        elif ex.domain == "iwt" and len(iwt_buf) < 3:
            iwt_buf.append(ex)
        if len(boun_buf) >= 3 and len(iwt_buf) >= 3:
            break

    show_examples(iter(boun_buf), 3, title="BOUN examples")
    show_examples(iter(iwt_buf), 3, title="ITU examples")

if __name__ == "__main__":
    main()

