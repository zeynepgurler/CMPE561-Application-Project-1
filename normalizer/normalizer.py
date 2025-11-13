"""
Simple Normalizer with masking, diacritic toggle, and ED<=1 fallback

Pipeline:
  1) Unicode NFKC
  2) Token-level normalization:
      - lexicon (case-insensitive; TR-aware lower)
      - diacritic toggle (try one position) validated by lexicon
      - edit-distance <= 1 fallback to nearest lexicon key
  3) Apostrophe & punctuation spacing fix
  4) Collapse spaces

  Masking replaces URL/EMAIL/NUM with placeholders and keeps them as-is.
"""

from typing import Dict, List, Optional
from collections import defaultdict
import unicodedata
import regex as re

# --- Turkish-aware lower/upper helpers (keep your existing versions if you already have them) ---
TR_UP = str.maketrans("iıüşöçg", "İIÜŞÖÇĞ")
TR_LOW = str.maketrans("İIÜŞÖÇĞ", "iıüşöçg")

def tr_lower(s: str) -> str:
    return s.translate(TR_LOW).lower()

def tr_upper_first(s: str) -> str:
    if not s:
        return s
    for i, ch in enumerate(s):
        if ch.isalpha():
            head = ch.translate(TR_UP).upper()
            return s[:i] + head + s[i+1:]
    return s


class UniversalSimpleNormalizer:
    """
    Universal Simple Normalizer (Normalization-focused)
    - Unicode pipeline: NFKC -> sanitize smart punctuation/zero-width -> NFC
    - Reversible masking (ID-based placeholders): URL/EMAIL/CUR/PERC/PHONE/DATE/USER/HASHTAG/EMOJI/EMOTICON/NUM
    - Token-level normalization via lexicon (case-insensitive, TR-aware), diacritic probing, ED<=1 fallback
    - Safe-vocab to keep certain tokens intact, frequency prior to boost confidence
    - Basic punctuation spacing fixes and whitespace compaction
    """

    def __init__(
        self,
        lexicon_path: str,
        use_masking: bool = True,
        use_diacritics: bool = True,
        use_edit_distance: bool = True,
        # Clean-corpus priors (e.g., BOUN)
        safe_vocab: Optional[set] = None,            # case-sensitive forms to never touch
        boun_freq: Optional[Dict[str, int]] = None,  # candidate target -> frequency
        freq_ok: int = 1,                            # minimal freq to trust a candidate
        allow_on_noisy: bool = True,                 # allow lexicon fixes when raw looks noisy
        force_downcase_after_first: bool = True,     # downcase tokens after the first if raw started lowercase
    ):
        # Toggles
        self.use_masking = use_masking
        self.use_diacritics = use_diacritics
        self.use_edit_distance = use_edit_distance
        self.force_downcase_after_first = force_downcase_after_first
        self.max_diacritic_flips_short = 2   # up to 2 flips for short tokens
        self.short_token_len = 6             # len<=6 => short
        self.use_repeat_shrink = True        # shrink 3+ char repetitions to 2

        # precompiled repeat regex
        self.re_repeat = re.compile(r"(.)\1{2,}", flags=re.IGNORECASE)

        # Priors
        self.safe_vocab = safe_vocab or set()
        self.safe_vocab_lc = {tr_lower(w) for w in self.safe_vocab}
        self.boun_freq = boun_freq or {}
        self.freq_ok = freq_ok
        self.allow_on_noisy = allow_on_noisy

        # --- Regexes for spacing & punctuation ---
        self.re_ws = re.compile(r"\s+")
        self.re_ap = re.compile(r"\s*'\s*")
        self.re_punct_before = re.compile(r"\s+([.,!?;:])")
        self.re_punct_after  = re.compile(r"([.,!?;:])(\S)")
        self.re_all_punct = re.compile(r"^\p{P}+$")

        # --- Core patterns ---
        self.re_url   = re.compile(r"(?i)\bhttps?://\S+")
        self.re_email = re.compile(r"(?i)\b[\w.\-+%]+@[\w.\-]+\.[A-Za-z]{2,}\b")
        self.re_num   = re.compile(r"(?<!\p{L})\p{N}+(?:[.,]\p{N}+)?(?!\p{L})")

        # --- Social/Real-world masking patterns ---
        # Mentions 
        self.re_mention = re.compile(r"(?<!\w)@(?:[\p{L}\p{N}_]{2,32})", flags=re.UNICODE)
        # Hashtags
        self.re_hashtag = re.compile(r"(?<!\w)#(?:[\p{L}\p{N}_]{2,64})", flags=re.UNICODE)
        # ASCII emoticons 
        self.re_emoticon = re.compile(
            r"(?:(?<!\S)(?:"
            r":-?\)|:-?D|:-?\(|:'\(|;-?\)|:-?P|:-?/|:-?\\|:-?\*|:-?\||:-?O|<3|xD|XD|=\)|=\(|:3"
            r")(?!\S))"
        )
        # Unicode emoji 
        self.re_emoji = re.compile(r"\p{Extended_Pictographic}", flags=re.UNICODE)
        # Phone numbers 
        self.re_phone = re.compile(
            r"(?<!\p{L})(?:\+?\d{1,3}[\s-]?)?(?:\(?\d{3}\)?[\s-]?)\d{3}[\s-]?\d{2}[\s-]?\d{2}(?!\p{L})"
        )
        # Percent values 
        self.re_percent = re.compile(r"(?<!\p{L})\p{N}+(?:[.,]\p{N}+)?\s?%(?!\p{L})")
        # Currency values 
        self.re_currency = re.compile(
            r"(?<!\p{L})(?:[€$₺£]\s?\p{N}+(?:[.,]\p{N}+)*|\p{N}+(?:[.,]\p{N}+)*\s?[€$₺£])(?!\p{L})"
        )
        # Dates: DD/MM/YYYY, DD.MM.YYYY, YYYY-MM-DD, and short yy
        self.re_date = re.compile(
            r"(?<!\p{L})(?:\d{1,2}[./-]\d{1,2}[./-]\d{2,4}|\d{4}[./-]\d{1,2}[./-]\d{1,2})(?!\p{L})"
        )

        # Diacritics map 
        self.dmap = {
            "c": ["c", "ç"], "ç": ["c", "ç"],
            "g": ["g", "ğ"], "ğ": ["g", "ğ"],
            "s": ["s", "ş"], "ş": ["s", "ş"],
            "o": ["o", "ö"], "ö": ["o", "ö"],
            "u": ["u", "ü"], "ü": ["u", "ü"],
            "i": ["i", "ı"], "ı": ["i", "ı"],
        }

        # Unicode sanitize map (smart quotes, dashes, ellipsis, NBSP/zero-width, BOM)
        self.safe_map = {
            "\u2019": "'", "\u2018": "'", "\u201C": '"', "\u201D": '"',
            "\u2013": "-", "\u2014": "-",
            "\u2026": "...",
            "\u00A0": " ", "\u200B": "", "\u200C": "", "\u200D": "", "\uFEFF": "",
        }

        # Reversible masking map (ID -> original string)
        self._mask_map: Dict[str, str] = {}

        # Lexicon: keys are TR-lowercased, values are target forms (cased)
        self.lexicon = self._load_lexicon(lexicon_path)
        self.lex_keys = list(self.lexicon.keys())

        # build length buckets once for ED<=1 search
        self.length_buckets = defaultdict(list)
        for k in self.lexicon.keys():
            self.length_buckets[len(k)].append(k)

        # Placeholders (ID-based forms will be <TAG1>, <TAG2>, ...)
        self.URL, self.EMAIL, self.NUM = "<URL>", "<EMAIL>", "<NUM>"
        self.USER, self.HASHTAG = "<USER>", "<HASHTAG>"
        self.EMOTICON, self.EMOJI = "<EMOTICON>", "<EMOJI>"
        self.PHONE, self.PERC, self.CUR, self.DATE = "<PHONE>", "<PERC>", "<CUR>", "<DATE>"

    # ---------- Public API ----------
    def normalize(self, text: str) -> str:
        """Main normalization pipeline."""
        # 1) Unicode normalization & sanitize
        s = self._normalize_unicode(text)

        # 2) Reversible masking (Level 2)
        if self.use_masking:
            s = self._mask(s)

        if self.use_repeat_shrink:
            s = self._shrink_repetitions(s)

        # 3) Token-level normalization
        raw_tokens = s.split()
        out_tokens: List[str] = [self._normalize_token(raw_tokens[0])] if raw_tokens else []
        for i in range(1, len(raw_tokens)):
            out_tokens.append(self._normalize_token(raw_tokens[i]))

        # 4) Simple sentence casing: capitalize first token; optionally downcase others
        if out_tokens:
            out_tokens[0] = tr_upper_first(out_tokens[0])
            for i in range(1, len(out_tokens)):
                raw_tok = raw_tokens[i]
                if self.force_downcase_after_first and self._raw_starts_lower(raw_tok):
                    out_tokens[i] = tr_lower(out_tokens[i])

        # 5) Spacing & compaction
        s = " ".join(out_tokens)
        s = self._fix_spacing(s)
        s = self._compact_ws(s)
        return s

    def unmask(self, s: str) -> str:
        """Replace ID-based placeholders with their original strings."""
        if not self._mask_map:
            return s
        # Replace in insertion order
        for key, original in self._mask_map.items():
            s = s.replace(key, original)
        return s

    # ---------- Token pipeline with confidence ----------
    def _normalize_token(self, tok: str) -> str:
        """Normalize a single token using lexicon/diacritics/ED<=1, with priors."""
        if not tok or self.re_all_punct.fullmatch(tok):
            return tok

        # Never touch tokens in safe vocabulary
        if (tok in self.safe_vocab) or (tr_lower(tok) in self.safe_vocab_lc):
            return tok

        key = tr_lower(tok)

        # 1 - Exact lexicon
        hit = self.lexicon.get(key)
        if hit is not None:
            return hit

        if self.use_diacritics:
            # 2 - Single-position diacritic variant (must hit lexicon)
            """
            cand_key = self._one_diacritic_variant_in_lex(key)
            if cand_key is not None:
                cand = self.lexicon[cand_key]
                return cand
            """
            
            # 2 - Diacritics: allow up to 2 flips for short tokens, 1 for others
            max_flips = self.max_diacritic_flips_short if len(key) <= self.short_token_len else 1
            cand_key = self._multi_diacritic_variant_in_lex(key, max_flips=max_flips)
            if cand_key is not None:
                return self.lexicon[cand_key]

        # 3 - ED<=1 nearest in lexicon (validated by confidence)
        if self.use_edit_distance:
            near = self._nearest_ed1(key)
            if near is not None:
                cand = self.lexicon[near]
                if self._confident(tok, cand, from_lexicon=True):
                    return cand

        # 4) No change
        return tok

    # ---------- Confidence & heuristics ----------
    def _confident(self, raw_tok: str, cand: str, from_lexicon: bool) -> bool:
        """Decide whether to accept a candidate replacement."""
        if cand == raw_tok:
            return True
        if self.boun_freq.get(cand, 0) >= self.freq_ok:
            return True
        if self.allow_on_noisy and from_lexicon and self._looks_noisy(raw_tok, cand):
            return True
        return False

    def _looks_noisy(self, raw_tok: str, cand: str) -> bool:
        """Heuristics for noisy tokens: elongations, simple diacritic mismatches, etc."""
        if re.search(r"(.)\1{2,}", raw_tok, flags=re.IGNORECASE):
            return True
        def strip_diac(s: str) -> str:
            return (s
                .replace("ç", "c").replace("ğ", "g").replace("ş", "s")
                .replace("ö", "o").replace("ü", "u").replace("ı", "i")
                .replace("Ç", "C").replace("Ğ", "G").replace("Ş", "S")
                .replace("Ö", "O").replace("Ü", "U").replace("İ", "I"))
        if strip_diac(raw_tok).lower() == strip_diac(cand).lower() and raw_tok.lower() != cand.lower():
            return True
        return False

    # ---------- Helpers ----------
    def _multi_diacritic_variant_in_lex(self, key: str, max_flips: int) -> Optional[str]:
        """Try flipping up to `max_flips` diacritic positions; return first lexicon hit."""
        from itertools import combinations, product

        idx = [i for i, ch in enumerate(key) if ch in self.dmap]
        if not idx:
            return None

        max_flips = max(1, max_flips)
        for flips in range(1, max_flips + 1):
            if flips > len(idx):
                break
            for pos_combo in combinations(idx, flips):
                # Build alternatives for selected positions
                alts_lists = []
                for i in pos_combo:
                    # only alternatives different from original
                    alts_lists.append([a for a in self.dmap[key[i]] if a != key[i]])
                for alts in product(*alts_lists):
                    chars = list(key)
                    for i, a in zip(pos_combo, alts):
                        chars[i] = a
                    cand_key = "".join(chars)
                    if cand_key in self.lexicon:
                        return cand_key
        return None

    def _one_diacritic_variant_in_lex(self, key: str) -> Optional[str]:
        """Try flipping a single diacritic position; return a lexicon key if found."""
        chars = list(key)
        for i, ch in enumerate(chars):
            if ch in self.dmap:
                for alt in self.dmap[ch]:
                    if alt == ch:
                        continue
                    chars[i] = alt
                    ck = "".join(chars)
                    if ck in self.lexicon:
                        return ck
                chars[i] = ch
        return None
    
    def _shrink_repetitions(self, s: str) -> str:
        """Shrink character runs of length >=3 to length 2 (e.g., 'çoook' -> 'çook')."""
        return self.re_repeat.sub(lambda m: m.group(1) * 2, s)

    def _nearest_ed1(self, key: str) -> Optional[str]:
        """Bucketed ED<=1 search with deterministic scoring among candidates."""
        L = len(key)
        # only lengths L-1, L, L+1 are relevant
        pool = self.length_buckets.get(L, []) + \
            self.length_buckets.get(L - 1, []) + \
            self.length_buckets.get(L + 1, [])
        if not pool:
            return None

        cands = []
        for k in pool:
            # quick length guard already applied
            if self._ed_leq_one(key, k):
                cands.append(k)

        if not cands:
            return None

        # scoring: prefer lower edit distance, higher corpus freq, richer diacritics, shorter target
        def score(k):
            tgt = self.lexicon[k]
            ed = 0 if key == k else 1
            freq = self.boun_freq.get(tgt, 0)
            dia = sum(ch in "çğıöşüÇĞİÖŞÜ" for ch in tgt)
            return (ed, -freq, -dia, len(tgt), k)  # last 'k' tie-breaker for determinism

        return min(cands, key=score)

    def _ed_leq_one(self, a: str, b: str) -> bool:
        """True iff Levenshtein(a,b) <= 1 (optimized for small distances)."""
        if a == b:
            return True
        la, lb = len(a), len(b)
        if abs(la - lb) > 1:
            return False
        if la == lb:
            return sum(1 for x, y in zip(a, b) if x != y) <= 1
        if la > lb:
            a, b = b, a
            la, lb = lb, la
        i = j = edits = 0
        while i < la and j < lb:
            if a[i] == b[j]:
                i += 1; j += 1
            else:
                edits += 1
                if edits > 1:
                    return False
                j += 1  # skip one char in longer string
        return True

    def _mask(self, s: str) -> str:
        """
        Reversible masking in a safe order (to avoid overlaps):
          1) URL, EMAIL
          2) Currency, Percent, Phone, Date
          3) Mentions (@user), Hashtags (#tag)
          4) Emoji (unicode), Emoticon (ASCII)
          5) Numbers (plain numeric literals)
        Resulting placeholders are ID-based (<TAG1>, <TAG2>, ...), recorded in _mask_map.
        """
        self._mask_map = {}

        def _repl(tag: str):
            def _r(m):
                i = len(self._mask_map) + 1
                key = f"<{tag}{i}>"
                self._mask_map[key] = m.group(0)
                return key
            return _r

        # 1) Highly fragile structures
        s = self.re_url.sub(_repl("URL"), s)
        s = self.re_email.sub(_repl("EMAIL"), s)

        # 2) Real-world numeric entities (before plain numbers)
        s = self.re_currency.sub(_repl("CUR"), s)
        s = self.re_percent.sub(_repl("PERC"), s)
        s = self.re_phone.sub(_repl("PHONE"), s)
        s = self.re_date.sub(_repl("DATE"), s)

        # 3) Social patterns
        s = self.re_mention.sub(_repl("USER"), s)
        s = self.re_hashtag.sub(_repl("HASHTAG"), s)

        # 4) Emoji(Unicode) and Emoticon(ASCII)
        s = self.re_emoji.sub(_repl("EMOJI"), s)
        s = self.re_emoticon.sub(_repl("EMOTICON"), s)

        # 5) Plain numbers (leftovers)
        s = self.re_num.sub(_repl("NUM"), s)

        return s

    def _normalize_unicode(self, s: str) -> str:
        """NFKC -> sanitize smart punctuation/zero-width -> NFC for stable Unicode."""
        s = unicodedata.normalize("NFKC", s)
        s = self._pre_sanitize(s)
        return unicodedata.normalize("NFC", s)

    def _pre_sanitize(self, s: str) -> str:
        """Map smart quotes/dashes/ellipsis; remove NBSP/ZW* and BOM."""
        return "".join(self.safe_map.get(ch, ch) for ch in s)

    def _compact_ws(self, s: str) -> str:
        """Collapse multiple spaces/tabs/newlines to a single space and trim ends."""
        return self.re_ws.sub(" ", s).strip()

    def _fix_spacing(self, s: str) -> str:
        """Tighten apostrophes, remove space before punctuation, ensure space after."""
        s = self.re_ap.sub("'", s)
        #s = self.re_punct_before.sub(r"\1", s)
        
        # Do not add space if punctuation is followed by a double quote
        s = re.sub(r"([.,!?;:])(?!\")(\S)", r"\1 \2", s)
        return s

    def _raw_starts_lower(self, raw_tok: str) -> bool:
        """Check if the first alphabetic char in the raw token was lowercase."""
        for ch in raw_tok:
            if ch.isalpha():
                return ch.islower()
        return False
    
    def _load_lexicon(self, path: str) -> Dict[str, str]:
        """Load TSV lexicon file: 'source<TAB>target'. Keys are TR-lowercased.
        NOTE: Pass src/tgt through the SAME unicode pipeline as input text to avoid NFC/NFKC mismatches."""
        mapping: Dict[str, str] = {}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "\t" not in line:
                    continue
                src, tgt = line.split("\t", 1)
                # make lexicon bytes compatible with normalize() pipeline
                src = self._normalize_unicode(src.strip())
                tgt = self._normalize_unicode(tgt.strip())
                mapping[tr_lower(src)] = tgt
        return mapping
