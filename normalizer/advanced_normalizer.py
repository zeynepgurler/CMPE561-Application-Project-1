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

from src.normalizer.normalizer import UniversalSimpleNormalizer
from typing import Dict, List, Optional, Iterable, Tuple
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

class UniversalAdvancedNormalizer(UniversalSimpleNormalizer):
    """
    Universal Advanced Normalizer (General-domain)
    Extends UniversalSimpleNormalizer with:
      - IWD + LV hooks (morphological analyzer / language validator)
      - Proper Noun detection (gazetteer + ratio heuristic + apostrophe handling)
      - Replacement rules (slang/leet/logogram), optional repeat shrink to 1
      - Diacritization hook (char-level deasciify) with LV validation
      - Partial Vowel Restoration hook (constrained), with LV validation
      - Accent Normalization (speechy -> written) via lightweight rules + generator validation
      - Precision-first Spelling Correction (ED≤2) gated by LV + frequency prior
    All advanced pieces are optional to keep it robust across domains.
    """

    def __init__(
        self,
        *args,
        morph_analyzer=None,          # expects .analyze(token) -> list[analyses]
        morph_generator=None,         # expects .generate(lemma, tags) -> list[str]
        disambiguator=None,           # expects .disambiguate(tokens, analyses) -> best
        diacritizer=None,             # expects .diacritize(token) -> str
        vowel_restorer=None,          # expects .restore(token) -> str
        proper_noun_gazetteers: Optional[set] = None,
        proper_ratio_thresh: float = 1.5,
        use_accent_norm: bool = True,
        use_vowel_restoration: bool = True,
        use_repeat_shrink_to_one: bool = False,  # if True: "çoook" -> "çok"; else -> "çook"
        use_slang: bool = False,                 # general-domain default: OFF
        lv_mode: str = "soft",                   # "soft" | "off"
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        # Hooks
        self.morph_analyzer = morph_analyzer
        self.morph_generator = morph_generator
        self.disambiguator = disambiguator
        self.diacritizer = diacritizer
        self.vowel_restorer = vowel_restorer

        # Resources/toggles
        self.proper_sets = proper_noun_gazetteers or []
        self.proper_ratio_thresh = proper_ratio_thresh
        self.use_accent_norm = use_accent_norm
        self.use_vowel_restoration = use_vowel_restoration
        self.use_repeat_shrink_to_one = use_repeat_shrink_to_one
        self.use_slang = use_slang
        self.lv_mode = lv_mode

        # Lightweight replacements (general-domain safe)
        self.repl_map: Dict[str, str] = {
            # common leet/logograms/slang (non-aggressive)
            "2m": "iyim", "mrb": "merhaba", "slm": "selam",
            "nbr": "ne haber", "kib": "kendine iyi bak",
            # symbols to letters (only when surrounded by letters)
            # handled via regex below
        }
        self.re_around_symbol = re.compile(r"(?<=\p{L})[@$](?=\p{L})", re.IGNORECASE)

        self.accent_rules = [
            # Şimdiki zaman
            (re.compile(r"(?i)\b(\p{L}{3,}?[ıiuü])yom\b"), r"\1yorum"),
            (re.compile(r"(?i)\b(\p{L}{3,}?[ıiuü])yon\b"), r"\1yorsun"),
            (re.compile(r"(?i)\b(\p{L}{3,}?[ıiuü])yosun\b"), r"\1yorsun"),
            (re.compile(r"(?i)\b(\p{L}{3,}?[ıiuü])yo\b"), r"\1yor"),
            (re.compile(r"(?i)\b(\p{L}{3,}?[ıiuü])yoz\b"), r"\1yoruz"),
            (re.compile(r"(?i)\b(\p{L}{3,}?[ıiuü])yosunuz\b"), r"\1yorsunuz"),
            (re.compile(r"(?i)\b(\p{L}{3,}?[ıiuü])yolar\b"), r"\1yorlar"),
            (re.compile(r"(?i)\b(\p{L}{2,}?)yolar\b"), r"\1yorlar"),

            # Gelecek zaman (olumlu)
            (re.compile(r"(?i)\b(\p{L}{3,}?)cem\b"), r"\1ceğim"),
            (re.compile(r"(?i)\b(\p{L}{3,}?)cam\b"), r"\1cağım"),
            (re.compile(r"(?i)\b(\p{L}{3,}?)cen\b"), r"\1ceksin"),
            (re.compile(r"(?i)\b(\p{L}{3,}?)can\b"), r"\1caksın"),
            (re.compile(r"(?i)\b(\p{L}{3,}?)cek\b"), r"\1ecek"),
            (re.compile(r"(?i)\b(\p{L}{3,}?)cak\b"), r"\1acak"),
            (re.compile(r"(?i)\b(\p{L}{3,}?)cez\b"), r"\1ceğiz"),
            (re.compile(r"(?i)\b(\p{L}{3,}?)caz\b"), r"\1cağız"),
            (re.compile(r"(?i)\b(\p{L}{3,}?)cekler\b"), r"\1ecekler"),
            (re.compile(r"(?i)\b(\p{L}{3,}?)caklar\b"), r"\1acaklar"),

            # Gelecek zaman (olumsuz)
            (re.compile(r"(?i)\b(\p{L}{3,}?)miycem\b"), r"\1meyeceğim"),
            (re.compile(r"(?i)\b(\p{L}{3,}?)micem\b"), r"\1mayacağım"),
            (re.compile(r"(?i)\b(\p{L}{3,}?)miycen\b"), r"\1meyeceksin"),
            (re.compile(r"(?i)\b(\p{L}{3,}?)mican\b"), r"\1mayacaksın"),
            (re.compile(r"(?i)\b(\p{L}{3,}?)miycek\b"), r"\1meyecek"),
            (re.compile(r"(?i)\b(\p{L}{3,}?)micak\b"), r"\1mayacak"),
            (re.compile(r"(?i)\b(\p{L}{3,}?)miycez\b"), r"\1meyeceğiz"),
            (re.compile(r"(?i)\b(\p{L}{3,}?)micaz\b"), r"\1mayacağız"),

            (re.compile(r"(?i)\b(\p{L}+?)yiz\b"), r"\1yiz"),
        ]

        # Question particle squashed: gidiyonmu -> gidiyor musun (very limited heuristic)
        self.re_q_particle = re.compile(r"(?i)^(.+?)(mi|mı|mu|mü)(n|sun|sunuz|siniz)?$")

        # Proper noun apostrophe (’)
        self.re_proper_apostrophe = re.compile(r"(?i)^([A-ZÇĞİÖŞÜ][\p{L}]+)(lar|ler)?(dan|den|e|a|i|ı|u|ü|in|ın|un|ün|de|da|te|ta|tir|tır|tur|dür|dır|dur)?$")

        # Core splitter for punct-preserving lexicon lookups
        self._re_tok_core = re.compile(r"^(\p{P}+)?([\p{L}\p{N}]+)(\p{P}+)?$", re.UNICODE)
 
    def normalize(self, text: str) -> str:
        # Unicode normalize + sanitize
        s = self.normalize_unicode(text)

        # Masking (IDs); then light repetition shrink
        if self.use_masking:
            s = self._mask(s)
        s = self.shrink_repetitions_to(s, 1 if self.use_repeat_shrink_to_one else 2)

        # Token pipeline (cascaded)
        raw_tokens = s.split()
        out_tokens: List[str] = []
        for i, tok in enumerate(raw_tokens):
            norm = self.normalize_token_cascade(tok)
            out_tokens.append(norm)

        # 4) Sentence casing like the base class
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

    # -------------------- Cascade --------------------
    def normalize_token_cascade(self, tok: str) -> str:
        # Early outs: punctuation-only or safe-vocab
        if (not tok) or self.re_all_punct.fullmatch(tok):
            return tok
        if (tok in self.safe_vocab) or (tr_lower(tok) in self.safe_vocab_lc):
            return tok

        # F0) Lexicon first, on CORE only, put lead/trail back.
        #     Gate by (freq_ok OR looks_noisy) to avoid over-normalization on clean text.
        lead, core, trail = self._split_core(tok)
        if core:
            key = tr_lower(core)
            hit = self.lexicon.get(key)
            if hit is not None:
                if self.boun_freq.get(hit, 0) >= self.freq_ok or self.looks_noisy(core, hit):
                    return lead + hit + trail

        x = tok

        # A) Replacement rules (slang/leet/logogram, *very* conservative)
        y = self.apply_replacements(x) if self.use_slang else x
        if y != x and self.lv_ok(y, context="repl"):
            return y
        x = y

        # B) Proper noun detection & apostrophe handling
        y = self.proper_noun_fix(x)
        if y != x and self.lv_ok(y, context="proper"):
            return y
        x = y  # continue with possibly improved token

        # C) Diacritization (hook -> fallback variants), gate by LV
        y = self.diacritize_seq(x)
        if y != x and self.lv_ok(y, context="diacritics"):
            return y
        x = y

        # D) Partial vowel restoration (hook -> heuristic), gate by LV
        if self.use_vowel_restoration and self.looks_noisy(x, x):
            y = self.restore_vowels(x)
            if y != x and self.lv_ok(y, context="vowel"):
                return y
            x = y

        # E) Accent normalization (speechy → written), validate via generator/analyzer
        if self.use_accent_norm and self._is_all_lower(x) and self.looks_noisy(x, x):
            y = self.accent_normalize(x)
            if y != x and self.lv_ok(y, context="accent") and self.boun_freq.get(y, 0) >= self.freq_ok:
                return y
            x = y

        # F) Base class lexicon/ED<=1 (already has boun_freq & diacritics)
        y = super().normalize_token(x)
        if y != x and self.lv_ok(y, context="lexicon"):
            return y
        x = y

        # G) Precision-first spelling correction ED<=2 (LV + freq prior)
        y = self.spell_correct_ed2(x)
        if y != x and self.lv_ok(y, context="ed2"):
            return y

        return y

    # -------------------- Layers --------------------
    def apply_replacements(self, tok: str) -> str:
        t = tok
        # Symbol-in-letter replacements: $→s, @→a (only between letters)
        def sym(m):
            ch = m.group(0)
            return "s" if ch == "$" else "a"
        t = self.re_around_symbol.sub(sym, t)

        # Small slang map (lowercased key match)
        low = tr_lower(t)
        if low in self.repl_map:
            t = self.repl_map[low]
        return t

    def proper_noun_fix(self, tok: str) -> str:
        """Detect probable proper nouns using gazetteers + ratio heuristic, add apostrophe if suffix-like ending exists."""
        if not self.proper_sets:
            return tok

        low = tr_lower(tok)
        hits = sum(1 for s in self.proper_sets if low in {tr_lower(x) for x in s})
        misses = len(self.proper_sets) - hits
        ratio = (hits + 1) / (misses + 1)  # smoothed

        candidate = tok
        if ratio >= self.proper_ratio_thresh:
            # ProperCase the first alpha (keep the rest as is)
            candidate = tr_upper_first(tok)

            # Add apostrophe if looks like a suffix-attached form
            m = self.re_proper_apostrophe.match(candidate)
            if m:
                stem, plural, suf = m.groups()
                if suf:  # e.g., Ahmetten -> Ahmet’ten
                    # naive split: move suffix after apostrophe
                    stem2 = stem + ("lar" if plural == "lar" else "ler" if plural == "ler" else "")
                    tail = suf
                    candidate = stem2 + "’" + tail
        return candidate

    def diacritize_seq(self, tok: str) -> str:
        if self.diacritizer is not None:
            try:
                out = self.diacritizer.diacritize(tok)
                return out or tok
            except Exception:
                pass
        # fallback: try base multi-flip search but gate with LV
        key = tr_lower(tok)
        max_flips = self.max_diacritic_flips_short if len(key) <= self.short_token_len else 1
        cand_key = self.multi_diacritic_variant_in_lex(key, max_flips=max_flips)
        if cand_key is not None:
            cand = self.lexicon[cand_key]
            return cand
        return tok

    def restore_vowels(self, tok: str) -> str:
        if self.vowel_restorer is not None:
            try:
                out = self.vowel_restorer.restore(tok)
                return out or tok
            except Exception:
                pass
        # conservative heuristic: collapse long consonant runs by inserting 'e' (very limited)
        # only apply if analyzer later approves
        t = re.sub(r"(?i)(\b\p{L}{2,}?)([bcçdfgğhjklmnprsştvyz]{3,})(\p{L}*\b)", r"\1e\2\3", tok)
        return t

    def accent_normalize(self, tok: str) -> str:
        t = tok
        for pat, repl in self.accent_rules:
            tt = pat.sub(repl, t)
            if tt != t:
                # if we have generator/analyzer, prefer a valid surface
                if self.lv_ok(tt, context="accent"):
                    return tt
            t = tt
        # Question particle heuristic (only if no apostrophe present)
        if "’" not in t and self.re_q_particle.match(t):
            # gidiyormusun -> gidiyor musun  
            m = self.re_q_particle.match(t)
            if m:
                root, mi, suf = m.groups()
                candidate = (root.rstrip() + " " + mi.lower() + ("" if not suf else " " + suf.lower())).strip()
                if self.lv_ok(candidate, context="accent"):
                    return candidate
        return t

    def spell_correct_ed2(self, tok: str) -> str:
        """Try ED<=2 correction with LV + priors (freq, diacritics).
        VERY conservative: only attempt on noisy tokens and accept only high-freq candidates."""
        if not self.looks_noisy(tok, tok):
            return tok
        key = tr_lower(tok)
        pool = (
            self.length_buckets.get(len(key)-2, []) +
            self.length_buckets.get(len(key)-1, []) +
            self.length_buckets.get(len(key), []) +
            self.length_buckets.get(len(key)+1, []) +
            self.length_buckets.get(len(key)+2, [])
        )
        best = tok
        best_score = None

        for k in pool:
            if k == key:
                continue
            if self.ed_leq_two(key, k):
                cand = self.lexicon[k]

                # require LV OK and frequency strong enough
                if not self.lv_ok(cand, context="ed2"):
                    continue
                if self.boun_freq.get(cand, 0) < self.freq_ok:
                    continue

                # score: lower ED, higher freq, richer diacritics, shorter
                ed = self.ed_est(key, k)
                freq = self.boun_freq.get(cand, 0)
                dia = sum(ch in "çğıöşüÇĞİÖŞÜ" for ch in cand)
                score = (ed, -freq, -dia, len(cand), k)
                if best_score is None or score < best_score:
                    best_score = score
                    best = cand
        return best

    # -------------------- Utilities --------------------
    def shrink_repetitions_to(self, s: str, n: int) -> str:
        """Shrink runs of >=(n+1) to length n."""
        if n <= 1:
            return re.sub(r"(.)\1{1,}", r"\1", s)
        else:
            s = re.sub(r"(.)\1{%d,}" % (n,), lambda m: m.group(1) * n, s)
        
            # matches: any letter repeated twice at end of a word
            s = re.sub(r"([A-Za-zÇĞİÖŞÜçğıöşü])\1\b", r"\1", s)

            return s

    def lv_ok(self, token: str, context: str = "general") -> bool:
        """Language Validator:
        - lv_mode='off' or analyzer=None => always True
        - lv_mode='soft' => require analysis if available, but allow low-risk contexts
        """
        if not token or token.strip() == "":
            return False
        if self.morph_analyzer is None or getattr(self, "lv_mode", "soft") == "off":
            return True
        try:
            analyses = self.morph_analyzer.analyze(token)
            if analyses:
                return True
            if getattr(self, "lv_mode", "soft") == "soft" and context in ("lexicon", "diacritics", "proper"):
                return True
            return False
        except Exception:
            return True  # fail-open for robustness

    @staticmethod
    def ed_est(a: str, b: str) -> int:
        """Very small-distance edit estimator (0/1/2+)."""
        if a == b:
            return 0
        # quick checks
        if len(a) == len(b):
            diff = sum(x != y for x, y in zip(a, b))
            if diff <= 1:
                return diff
        # fallback: <=2 check
        return 1 if UniversalAdvancedNormalizer.ed_leq_one(a, b) else (2 if UniversalAdvancedNormalizer.ed_leq_two(a, b) else 3)

    @staticmethod
    def ed_leq_one(a: str, b: str) -> bool:
        # re-use base logic if available
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
                j += 1
        return True

    @staticmethod
    def ed_leq_two(a: str, b: str) -> bool:
        """True iff Levenshtein(a,b) <= 2. Banded DP (Ukkonen) with k=2."""
        la, lb = len(a), len(b)
        k = 2
        if abs(la - lb) > k:
            return False
        if a == b:
            return True
        prev = list(range(lb + 1))
        for i in range(1, la + 1):
            curr = [k + 1] * (lb + 1)
            j_start = max(1, i - k)
            j_end   = min(lb, i + k)
            curr[0] = i
            for j in range(j_start, j_end + 1):
                cost = 0 if a[i - 1] == b[j - 1] else 1
                deletion     = prev[j] + 1
                insertion    = curr[j - 1] + 1
                substitution = prev[j - 1] + cost
                curr[j] = min(deletion, insertion, substitution)
            prev = curr
            band = prev[max(1, i - k): min(lb, i + k) + 1]
            if band and min(band) > k:
                return False
        return prev[lb] <= k

    def _split_core(self, tok: str):
        m = self._re_tok_core.match(tok)
        if not m:
            return "", tok, ""
        return m.group(1) or "", m.group(2), m.group(3) or ""

    @staticmethod
    def _is_all_lower(s: str) -> bool:
        for ch in s:
            if ch.isalpha():
                if not ch.islower():
                    return False
        return True

