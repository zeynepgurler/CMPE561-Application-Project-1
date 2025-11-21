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

# Set the path as src, so we can use relative paths
from pathlib import Path
import sys

current = Path(__file__).resolve()
for parent in current.parents:
    if (parent / "src").is_dir():
        project_root = parent
        break
else:
    project_root = current.parent  

# Add the project root path into sys
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from typing import List, Dict, Iterable, Optional, Tuple, Set
from dataclasses import dataclass
from itertools import islice
import regex as re

from src.tokenizer.utils import build_text_and_boundaries_from_example, merge_boun_mwes

# ----------------------------
# Utils
# ----------------------------

# Minimal TR-friendly detokenizer (useful for UD/BOUN where punctuation is split)
_PUNCT_NO_SPACE_BEFORE = set(".,;:!?%)”’›»…")
_PUNCT_NO_SPACE_AFTER  = set("(%“‘‹«")

def detok(tokens: List[str]) -> str:
    """Reconstruct a sentence from UD-style tokens. We need the reconstruct sentences so we can use their detokenized versions when training a NB tokenizer"""
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
    """Join for IWT where apostrophe forms are already single tokens."""
    return " ".join(tokens)

# Mask special tokens to not falsely normalize
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

    If boun_mwe_aware=True, BOUN gold tokens will merge certain
    UD-style MWE groups (flat/fixed/mwe/compound) into a single
    gold token (e.g. "Türk Hava Yolları"). Because we need to 
    learn to tokenize MWEs as well. 
    """
    def __init__(self, boun_detok: bool = True, boun_mwe_aware: bool = True):
        self.sources: List[Tuple[str, str]] = []
        self.boun_detok = boun_detok
        self.boun_mwe_aware = boun_mwe_aware

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
            elif stype == "iwt":
                yield from self.read_iwt_with_sentence_begin(path)
            else:
                raise ValueError(f"Unknown source type: {stype}")

    # --------- BOUN (UD CoNLL-U) ----------
    def read_boun_conllu(self, path: str) -> Iterable[Example]:
        """
        CoNLL-U reader
        - Skips multi-word token ranges (e.g., '3-4') and empty nodes ('2.1').
        - Reconstructs sentence text from tokens if # text is missing.
        - If self.boun_mwe_aware is True, merges some MWE groups
          (flat/fixed/mwe/compound) into single gold tokens.
        """
        sent_meta: Dict[str, str] = {}
        sent_tokens: List[dict] = []  

        with open(path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.rstrip("\n")
                if not line:
                    # end of sentence
                    if sent_tokens:
                        # Decide gold_tokens (MWE-aware or plain)
                        if self.boun_mwe_aware:
                            gold_tokens = merge_boun_mwes(sent_tokens)
                        else:
                            gold_tokens = [t["form"] for t in sent_tokens]

                        txt = sent_meta.get("text")
                        if not txt:
                            txt = detok(gold_tokens) if self.boun_detok else " ".join(gold_tokens)

                        yield Example(
                            raw_text=txt,
                            gold_text=txt,
                            gold_tokens=gold_tokens, # [:]
                            domain="boun",
                            sent_id=sent_meta.get("sent_id"),
                        )
                        sent_tokens.clear()
                        sent_meta.clear()
                    continue

                if line.startswith("#"):
                    # metadata lines like: "# sent_id = ..." or "# text = ..."
                    if " = " in line:
                        k, v = line[2:].split(" = ", 1)
                        sent_meta[k.strip()] = v.strip()
                    continue

                parts = line.split("\t")
                if len(parts) < 8:
                    continue
                tok_id = parts[0]
                if "-" in tok_id or "." in tok_id:
                    # skip multi-word ranges and empty nodes
                    continue

                tid = int(tok_id)
                form = parts[1]
                head = parts[6]
                deprel = parts[7]

                # some HEADs are "0" (root)
                try:
                    head_int = int(head)
                except ValueError:
                    head_int = 0

                sent_tokens.append({
                    "id": tid,
                    "form": form,
                    "head": head_int,
                    "deprel": deprel,
                })

            # EOF flush
            if sent_tokens:
                if self.boun_mwe_aware:
                    gold_tokens = merge_boun_mwes(sent_tokens)
                else:
                    gold_tokens = [t["form"] for t in sent_tokens]

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
                            gold_tokens=gold_sent, #[:],
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

    def iterate_for_tokenization(self, domains: Optional[List[str]] = None, max_sentences: Optional[int] = None) -> Iterable[Tuple[str, Set[int], Example]]:
        """
        Iterate over examples in a form that is directly usable for
        tokenization training on real sentences (detokenized).

        For each sentence, it returns:
            - text: the sentence string (raw_text or gold_text, depending on domain)
            - boundaries: set of gold token boundary positions on that text
            - example: the original Example object
        """
        count = 0
        for ex in self.iterate():
            if domains is not None and ex.domain not in domains:
                continue
            if not ex.gold_tokens:
                continue

            text, boundaries = build_text_and_boundaries_from_example(ex)
            if not boundaries:
                # alignment failed, skip this sentence
                continue

            yield text, boundaries, ex

            count += 1
            if max_sentences is not None and count >= max_sentences:
                break

