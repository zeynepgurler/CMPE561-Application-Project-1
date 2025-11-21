from typing import List, Tuple, Dict, Optional
from collections import Counter, defaultdict
import unicodedata
import regex as re
import json


class SimpleLexiconBuilder:
    """
    Build a normalization lexicon from ITU Web Treebank (withSentenceBegin)
    and optionally prefer targets that also appear in BOUN UD (CoNLL-U).

    - Turkish-aware lowercase keys
    - Skips identities and special patterns (URL/email/number/placeholders)
    - For each key, picks target with highest (freq + boun_bonus_if_seen)
    - Keep code simple and readable
    """

    def __init__(self, max_len: int = 40):
        self.max_len = max_len
        self.pair_counter: Counter[Tuple[str, str]] = Counter()

        # --- BOUN support ---
        self.boun_vocab: set[str] = set()          # case-sensitive forms from BOUN
        self.boun_freq: Counter[str] = Counter()   # optional: counts per token

        # Patterns to ignore or detect
        self.re_ws = re.compile(r"\s+")
        self.re_all_punct = re.compile(r"^\p{P}+$")
        self.re_url = re.compile(r"(?i)\bhttps?://\S+")
        self.re_email = re.compile(r"(?i)\b[\w.\-+%]+@[\w.\-]+\.[A-Za-z]{2,}\b")
        self.re_num = re.compile(r"(?<!\p{L})\d+(?:[.,]\d+)?(?!\p{L})")

        # Placeholders to skip
        self.skip_tokens = {"<URL>", "<EMAIL>", "<NUM>", "<PCT>", "<MONEY>", "<DATE>", "<TIME>", "<PHONE>"}

    # ----------------- tiny utilities -----------------

    def tr_lower(self, s: str) -> str:
        """Turkish-aware lowercase (I→ı, İ→i, then lower)."""
        return s.replace("I", "ı").replace("İ", "i").lower()

    def nfkc(self, s: str) -> str:
        """Unicode normalization to NFKC."""
        return unicodedata.normalize("NFKC", s)

    def clean(self, s: str) -> str:
        """Normalize unicode and collapse whitespace."""
        s = self.nfkc(s)
        s = self.re_ws.sub(" ", s).strip()
        return s

    def is_ignorable(self, s: str) -> bool:
        """Skip placeholders, URLs/emails/numbers, pure punctuation, empties, very long tokens."""
        if not s:
            return True
        if len(s) > self.max_len:
            return True
        if s in self.skip_tokens:
            return True
        if self.re_all_punct.fullmatch(s):
            return True
        if self.re_url.fullmatch(s) or self.re_email.fullmatch(s) or self.re_num.fullmatch(s):
            return True
        return False

    # ----------------- ITU reader -----------------

    def add_iwt_file(self, path: str) -> None:
        """
        Read an ITU '...withSentenceBegin' file and collect (raw, gold) token-level pairs.
        Lines:
          <sentBegin>\traw\tgold  -> new sentence
          raw\tgold               -> continue sentence
        """
        raw_sent: List[str] = []
        gold_sent: List[str] = []

        def flush_sentence():
            for r, g in zip(raw_sent, gold_sent):
                self.consider_pair(r, g)
            raw_sent.clear()
            gold_sent.clear()

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")

                if not line or line.startswith("#"):
                    if raw_sent and len(raw_sent) == len(gold_sent):
                        flush_sentence()
                    continue

                parts = line.split("\t")
                if len(parts) == 1:
                    parts = re.split(r"\s+", line.strip())
                parts = [p for p in parts if p != ""]
                if not parts:
                    continue

                if len(parts) == 3 and parts[0] == "<sentBegin>":
                    if raw_sent and len(raw_sent) == len(gold_sent):
                        flush_sentence()
                    raw_sent.clear()
                    gold_sent.clear()
                    raw_sent.append(parts[1])
                    gold_sent.append(parts[2])
                elif len(parts) >= 2:
                    raw_sent.append(parts[-2])
                    gold_sent.append(parts[-1])

            if raw_sent and len(raw_sent) == len(gold_sent):
                flush_sentence()

    # ----------------- BOUN reader -----------------

    def add_boun_conllu(self, path: str) -> None:
        """
        Read a BOUN UD CoNLL-U file and collect *surface* token vocabulary.
        Prefer '# text = ...' line (already detokenized; MWT merged).
        If '# text' missing, reconstruct from FORM + SpaceAfter=No.
        Punctuation at token edges is stripped for vocab purposes.
        """
        # small helpers kept local for clarity
        punct_strip = re.compile(r"^\p{P}+|\p{P}+$")  # strip leading punctuation

        def emit_from_surface_text(txt: str):
            # split by whitespace, strip edge punctuation, clean, filter
            for t in txt.split():
                t = punct_strip.sub("", t)
                t = self.clean(t)
                if not t or self.is_ignorable(t):
                    continue
                self.boun_vocab.add(t)
                self.boun_freq[t] += 1

        sent_text = None           # from "# text = ..."
        sent_forms = []            # (form, space_after_no: bool)

        def flush_sentence():
            # prefer #text
            if sent_text:
                emit_from_surface_text(sent_text)
            else:
                # reconstruct naive surface using SpaceAfter=No
                chunks = []
                for form, no_space in sent_forms:
                    if not chunks:
                        chunks.append(form)
                    else:
                        if no_space:
                            chunks[-1] = chunks[-1] + form
                        else:
                            chunks.append(" " + form)
                reconstructed = "".join(chunks)
                emit_from_surface_text(reconstructed)

        with open(path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.rstrip("\n")

                if not line:
                    # end of sentence
                    if sent_forms or sent_text:
                        flush_sentence()
                    sent_text = None
                    sent_forms.clear()
                    continue

                if line.startswith("#"):
                    # metadata: capture '# text = ...' if present
                    if line.startswith("# text = "):
                        sent_text = line[len("# text = "):].strip()
                    continue

                parts = line.split("\t")
                if len(parts) < 8:
                    continue

                tok_id = parts[0]
                form   = self.clean(parts[1])
                misc   = parts[9] if len(parts) >= 10 else "_"

                # skip multi-word ranges and empty nodes
                if "-" in tok_id or "." in tok_id:
                    continue
                if not form or self.is_ignorable(form):
                    continue

                no_space = False
                if misc and misc != "_":
                    # detect SpaceAfter=No
                    # (works even if there are multiple MISC items, e.g. "SpaceAfter=No|...=...")
                    if "SpaceAfter=No" in misc:
                        no_space = True

                sent_forms.append((form, no_space))

            # EOF flush
            if sent_forms or sent_text:
                flush_sentence()

    # ----------------- core steps -----------------

    def consider_pair(self, raw_tok: str, gold_tok: str) -> None:
        """Apply cleaning and filtering, then count the (key, target) pairs from ITU."""
        raw_tok = self.clean(raw_tok)
        gold_tok = self.clean(gold_tok)

        if not raw_tok or not gold_tok:
            return
        if self.is_ignorable(raw_tok) or self.is_ignorable(gold_tok):
            return
        if raw_tok == gold_tok:
            return  # identity - no need in lexicon

        key = self.tr_lower(raw_tok)  # case-insensitive key
        tgt = gold_tok                # keep target casing
        self.pair_counter[(key, tgt)] += 1

    """
    def build(self, min_freq: int = 1, boun_bonus: int = 1, prefer_boun: bool = True) -> Dict[str, str]:
        
        Aggregate pairs by key and choose the target with the highest score.
        score(target) = freq + (boun_bonus if target in BOUN vocab and prefer_boun else 0)
        
        # group key -> {tgt: freq}
        grouped: Dict[str, Dict[str, int]] = defaultdict(dict)
        for (key, tgt), freq in self.pair_counter.items():
            if freq >= min_freq:
                grouped[key][tgt] = grouped[key].get(tgt, 0) + freq

        lex: Dict[str, str] = {}
        for key, tgts in grouped.items():
            best_tgt: Optional[str] = None
            best_score = -1
            for tgt, freq in tgts.items():
                score = freq
                if prefer_boun and tgt in self.boun_vocab:
                    score += boun_bonus
                # tie-breaker: higher freq wins; then lexicographic for determinism
                if score > best_score or (score == best_score and tgt < (best_tgt or "")):
                    best_score = score
                    best_tgt = tgt
            if best_tgt is not None:
                lex[key] = best_tgt
        return lex
    """
    def build(self, min_freq: int = 1, boun_bonus: int = 1, prefer_boun: bool = True, expand_with_boun_identities: bool = False, min_boun_freq: int = 1) -> Dict[str, str]:
        """
        score(target) = freq + (boun_bonus if target in BOUN and prefer_boun else 0)

        If expand_with_boun_identities=True:
        - For any key not seen in ITU Web, backfill from BOUN:
            key = tr_lower(boun_form) -> target = canonical BOUN casing
        - Canonical casing chosen by max BOUN frequency per lowercase bucket.
        """
        # ITU: group key -> {tgt: freq}
        grouped: Dict[str, Dict[str, int]] = defaultdict(dict)
        for (key, tgt), freq in self.pair_counter.items():
            if freq >= min_freq:
                grouped[key][tgt] = grouped[key].get(tgt, 0) + freq

        # ITU: resolve best target per key (with optional BOUN bonus)
        lex: Dict[str, str] = {}
        for key, tgts in grouped.items():
            best_tgt: Optional[str] = None
            best_score = -1
            for tgt, freq in tgts.items():
                score = freq
                if prefer_boun and tgt in self.boun_vocab:
                    score += boun_bonus
                if score > best_score or (score == best_score and tgt < (best_tgt or "")):
                    best_score = score
                    best_tgt = tgt
            if best_tgt is not None:
                lex[key] = best_tgt

        # 3) OPTIONAL: expand with BOUN identities (only for keys not in ITU)
        if expand_with_boun_identities and self.boun_freq:
            # build lowercase -> best-cased target by BOUN frequency
            bucket: Dict[str, Tuple[str, int]] = {}  # lc -> (best_form, freq)
            for form, f in self.boun_freq.items():
                if f < min_boun_freq:
                    continue
                lc = self.tr_lower(form)
                # skip ignorable & too-long forms (already filtered when filling boun_freq)
                cur = bucket.get(lc)
                if (cur is None) or (f > cur[1]) or (f == cur[1] and form < cur[0]):
                    bucket[lc] = (form, f)

            # backfill: only add keys that ITU did NOT provide
            added = 0
            for lc, (best_form, _) in bucket.items():
                if lc not in lex:
                    lex[lc] = best_form
                    added += 1
            # (isteğe bağlı) küçük bir log:
            # print(f"[build] BOUN backfill added {added} entries (min_boun_freq={min_boun_freq})")

        return lex

    def save_tsv(self, lex: Dict[str, str], path: str) -> None:
        """Write lexicon as 'source<TAB>target' lines (keys are TR-lowered already)."""
        with open(path, "w", encoding="utf-8") as f:
            for src in sorted(lex.keys()):
                f.write(f"{src}\t{lex[src]}\n")


# =========================
# ===== Usage example =====
# =========================
if __name__ == "__main__":
    builder = SimpleLexiconBuilder(max_len=40)

    # 1) ITU Web -> raw to gold pairs
    builder.add_iwt_file("data/IWTandTestSmall/IWT_normalizationerrorsNoUpperCase.withSentenceBegin")

    # 2) BOUN UD -> vocabulary (to prefer cleaner targets)
    builder.add_boun_conllu("data/UD_Turkish-BOUN_v2.11_unrestricted-main/train-unr.conllu")

    safe_vocab = set(builder.boun_vocab) # DO-NOT-TOUCH list
    boun_freq  = dict(builder.boun_freq)

    with open("normalization_resources/boun_safe_vocab.txt", "w", encoding="utf-8") as f:
        for tok in safe_vocab:
            f.write(tok + "\n")

    payload = {tok: int(freq) for tok, freq in boun_freq.items()}
    with open("normalization_resources/boun_freq.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)

    # Build lexicon
    #    - min_freq: how many times should the word appear for acception
    #    - boun_bonus: if the word appears in Boun (clean treebank), give a bonus point
    #    - prefer_boun: if True, prefer the Boun ones
    
    #lex = builder.build(min_freq=1, boun_bonus=1, prefer_boun=True)
    lex = builder.build(min_freq=1, boun_bonus=1, prefer_boun=True, expand_with_boun_identities=False, min_boun_freq=2)

    builder.save_tsv(lex, "normalization_resources/lexicon.tsv")
    print("Saved", len(lex), "entries to lexicon.tsv")
