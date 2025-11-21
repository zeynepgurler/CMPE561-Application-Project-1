from __future__ import annotations
import regex as re
from typing import List, Optional


class RuleBasedUnifiedTokenizer:
    """
    Rule-based tokenizer tuned mainly for BOUN Treebank,
    but also supports ITU Web Treebank with the domain parameter.

    domain="boun":
        - Makes punctuation marks into separate tokens.
        - '.' within a number and ',' characters (e.g., 3.14, 1.25).
        - Heuristically splits some copula/time suffixes:

        * ağlayacaktı -> ağlayacak + tı
        * şöyleydi    -> şöyle + ydi
        * girebilirdik -> girebilir + dik
        * böyledir    -> böyle + dir (if not a verb)
        - Leaves apostrophe forms (SASA's, Meclis', Türkiye'ye) as a SINGLE token.
        - Combines multi-word expressions that match EXACTLY on the token array with externally supplied MWE lists into a single token.

    domain="iwt" (ITU Web Treebank):
        - Preserves patterns such as @mention[@user], @smiley[:)], @vocative[ahaha], @hashtag[#tag], @keyword[RT] as a SINGLE token.
        - Characters such as [, ], :, ', ( within these patterns are never made into separate tokens.
        - Morphological split DOES NOT.
        - MWE merge (Wikipedia/proper-name MWE) OFF (Emre Aydın, fotoğraf çekmek, bir kere vs. birleşmez).
    """

    VOWELS = "aeıioöuüAEIİOÖUÜ"

    # Yalnızca sınırlı kopula ekleri (BOUN için)
    COPULA_SUFFIXES = [
        "dır", "dir", "dur", "dür",
        "tır", "tir", "tur", "tür",
    ]

    ELLIPSIS_PLACEHOLDER = "<ELLIPSIS>"

    def __init__(
        self,
        mwe_path: Optional[str] = None,
        proper_mwe_path: Optional[str] = None,
    ) -> None:
        """
        mwe path: Text file path with one MWE per line
                          exp: "ya da", "geri döndükten", "paldür küldür" 
        proper_mwe_path: Text file path with a multi-word proper name per line.
                          exp: "Ahmet Hamdi Tanpınar", "Beş Şehir"
        """
        self.general_mwes: List[str] = []
        self.proper_mwes: List[str] = []

        if mwe_path:
            with open(mwe_path, encoding="utf-8") as f:
                for line in f:
                    term = line.strip()
                    if not term:
                        continue
                    if " " in term:
                        self.general_mwes.append(term)

        if proper_mwe_path:
            with open(proper_mwe_path, encoding="utf-8") as f:
                for line in f:
                    term = line.strip()
                    if not term:
                        continue
                    if " " in term:
                        self.proper_mwes.append(term)

        # Tüm MWE'leri tek listede topla (BOUN için kullanılacak)
        self.all_mwes: List[str] = self.general_mwes + self.proper_mwes
        # Sliding window için token sayısına göre uzun olan önce denensin
        self.all_mwes.sort(key=lambda x: len(x.split()), reverse=True)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def tokenize(self, text: str, domain: str = "boun") -> List[str]:
        """
        Ana tokenizasyon fonksiyonu.

        domain:
          - "boun" : BOUN kuralları (morfolojik split + MWE merge)
          - "iwt"  : ITU Web kuralları (tag koruma, no morph split, no MWE)
        """
        if not text.strip():
            return []

        tag_map = {}
        if domain == "iwt":
            # ITU tweetlerindeki özel tag'leri koru
            text, tag_map = self._protect_iwt_tags(text)

        # 1) Noktalama etrafına boşluk ekle
        text = self._add_spaces(text)

        # 2) Basit whitespace split
        rough_tokens = text.split()

        # 3) Apostrof kuralı (şu an BOUN için de IWT için de split yok)
        tokens_after_apo: List[str] = []
        for tok in rough_tokens:
            tokens_after_apo.extend(self._apostrophe_rule(tok))

        # 4) Kopula / zaman eklerinin sınırlı ayrılması (sadece BOUN)
        morph_tokens: List[str] = []
        for tok in tokens_after_apo:
            if domain == "boun":
                morph_tokens.extend(self._split_morph(tok))
            else:
                # ITU: hiçbir morfolojik split yapma
                morph_tokens.append(tok)

        # 5) MWE merge
        if domain == "boun":
            merged_tokens = self._merge_mwes(morph_tokens)
        else:
            # ITU: MWE merge kapalı (Wikipedia/proper-name MWE'leri kullanma)
            merged_tokens = morph_tokens

        # 6) ITU tag'lerini geri aç
        if domain == "iwt":
            merged_tokens = self._restore_iwt_tags(merged_tokens, tag_map)

        return merged_tokens

    # ------------------------------------------------------------------ #
    # 1) Noktalama işleme
    # ------------------------------------------------------------------ #
    def _add_spaces(self, text: str) -> str:
        # Önce üç nokta'yı placeholder ile koru ki nokta kuralı bozmasın
        text = text.replace("...", f" {self.ELLIPSIS_PLACEHOLDER} ")

        # ?, !, ;, parantez vb.
        text = re.sub(r"([?!;(){}\[\]])", r" \1 ", text)
        # sayının parçası olmayan virgüller
        text = re.sub(r"(?<!\d),(?!\d)", r" , ", text)
        # sayının parçası olmayan iki nokta
        text = re.sub(r"(?<!\d):(?!\d)", r" : ", text)
        # sayının parçası olmayan noktalar
        text = re.sub(r"(?<!\d)\.(?!\d)", r" . ", text)
        # tırnaklar
        text = re.sub(r"([\"“”«»])", r" \1 ", text)

        # whitespace normalizasyonu
        text = re.sub(r"\s+", " ", text).strip()

        # Ellipsis placeholder'ı tekrar gerçek '...' ile değiştir
        text = text.replace(self.ELLIPSIS_PLACEHOLDER, "...")

        return text

    # ------------------------------------------------------------------ #
    # 2) ITU tag koruma
    # ------------------------------------------------------------------ #
    def _protect_iwt_tags(self, text: str):
        """
        ITU Web tweetlerindeki özel pattern'leri tek token olarak korur:
          @mention[@user]
          @smiley[:)]
          @vocative[ahaha]
          @hashtag[#tag]
          @keyword[RT]
        Bunları placeholder'a çevirir, mapping döner.
        """
        pattern = re.compile(
            r"@(mention|smiley|vocative|hashtag|keyword)\[[^\]]*\]"
        )

        tag_map = {}

        def repl(m):
            idx = len(tag_map)
            placeholder = f"<IWT_TAG_{idx}>"
            tag_map[placeholder] = m.group(0)
            return placeholder

        new_text = pattern.sub(repl, text)
        return new_text, tag_map

    def _restore_iwt_tags(self, tokens: List[str], tag_map):
        if not tag_map:
            return tokens
        return [tag_map.get(tok, tok) for tok in tokens]

    # ------------------------------------------------------------------ #
    # 3) Apostrof kuralı
    # ------------------------------------------------------------------ #
    def _apostrophe_rule(self, token: str) -> List[str]:
        """
        BOUN ve ITU için şu an apostroflu kelimeleri bölmüyoruz:
          SASA'nın    -> ["SASA'nın"]
          Meclis'in   -> ["Meclis'in"]
          Türkiye'ye  -> ["Türkiye'ye"]
        """
        return [token]

    # ------------------------------------------------------------------ #
    # 4) Morfolojik split (BOUN)
    # ------------------------------------------------------------------ #
    def _split_morph(self, token: str) -> List[str]:
        """
        evaluate_tokenizer ile uyumlu wrapper.
        """
        return self._split_limited_copula(token)

    def _split_limited_copula(self, token: str) -> List[str]:
        """
        Heuristik olarak şu durumlarda split yapar (sadece BOUN'da kullanılır):
          - (.*)(acak|ecek)(tı|ti|tu|tü)
                ağlayacaktı  -> ağlayacak + tı
          - (.*)(ydi|ymiş|yken)
                şöyleydi     -> şöyle + ydi
          - (.*bil)(ir)(dik)
                girebilirdik -> girebilir + dik
          - Sonunda COPULA_SUFFIXES varsa ve fiil gibi görünmüyorsa:
                böyledir     -> böyle + dir
        """
        lower = token.lower()

        # 1) -acak/-ecek + tı/ti/tu/tü
        m = re.match(r"^(.*?)(acak|ecek)(tı|ti|tu|tü)$", lower)
        if m:
            stem_len = len(m.group(1)) + len(m.group(2))
            stem = token[:stem_len]
            suf = token[stem_len:]
            return [stem, suf]

        # 2) -ydi/-ymiş/-yken
        m2 = re.match(r"^(.*)(ydi|ymiş|yken)$", lower)
        if m2:
            stem = token[:len(m2.group(1))]
            suf = token[len(m2.group(1)):]
            if len(stem) > 1:
                return [stem, suf]

        # 3) -bil + ir + dik (girebilirdik, yapabilirdik vb.)
        m3 = re.match(r"^(.*bil)(ir)(dik)$", lower)
        if m3:
            stem_len = len(m3.group(1)) + len(m3.group(2))
            stem = token[:stem_len]
            suf = token[stem_len:]
            if len(stem) > 2:
                return [stem, suf]

        # 4) COPULA_SUFFIXES
        for suf in self.COPULA_SUFFIXES:
            if lower.endswith(suf):
                # Fiil gibi görünüyorsa dokunma
                if self._looks_like_verb(lower):
                    return [token]

                stem = token[:-len(suf)]
                if len(stem) < 2:
                    return [token]
                if not any(c in self.VOWELS for c in stem):
                    return [token]
                return [stem, token[-len(suf):]]

        # Hiçbir kural tetiklenmediyse
        return [token]

    def _looks_like_verb(self, lower: str) -> bool:
        """
        Çok kaba bir fiil heuristiği.
        Eğer kelime tipik fiil son ekleriyle bitiyorsa, kopula split yapma.
        """
        if lower.endswith(("dı", "di", "du", "dü",
                           "tı", "ti", "tu", "tü")):
            return True
        if lower.endswith(("yor", "muş", "miş", "acak", "ecek")):
            return True
        return False

    # ------------------------------------------------------------------ #
    # 5) MWE merge (BOUN için)
    # ------------------------------------------------------------------ #
    def _merge_mwes(self, tokens: List[str]) -> List[str]:
        """
        Token listesi üzerinde, self.all_mwes içindeki ifadelerle
        TAM token dizisi eşleşmesini bulup merge eder.

        Örn:
          tokens   = ['Ahmet', 'Hamdi', 'Tanpınar', ',', 'Beş', 'Şehir']
          all_mwes = ['Ahmet Hamdi Tanpınar', 'Beş Şehir']

          -> ['Ahmet Hamdi Tanpınar', ',', 'Beş Şehir']
        """
        if not self.all_mwes:
            return tokens

        mwe_set = set(self.all_mwes)
        max_len = max(len(mwe.split()) for mwe in self.all_mwes)

        merged_tokens: List[str] = []
        i = 0
        n = len(tokens)

        while i < n:
            merged = None

            max_window = min(max_len, n - i)
            for L in range(max_window, 1, -1):  # en az 2 kelimelik MWE
                span = tokens[i:i + L]
                phrase = " ".join(span)
                if phrase in mwe_set:
                    merged = phrase
                    i += L
                    break

            if merged is not None:
                merged_tokens.append(merged)
            else:
                merged_tokens.append(tokens[i])
                i += 1

        return merged_tokens
