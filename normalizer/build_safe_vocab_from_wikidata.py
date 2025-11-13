import re, sys, unicodedata
from collections import Counter

# ==== Ayarlar ====
SQL_PATH   = "C:/Users/90551/Desktop/NLP/application_project1/datasets/trwiki-latest-page.sql/trwiki-latest-page.sql"   # elinizdeki dump
OUT_TXT    = "safe_vocab.txt"
MIN_COUNT  = 2         # en az 2 kez geçen biçimleri tut
MAX_LEN    = 40
MIN_LEN    = 2
MAX_ACRON  = 8         # ACRONYM uzunluk eşiği
NS_FILTER  = {0}       # yalnızca ana aduzayı (0)
# MySQL dump'ta page tuple alan sırası genelde:
# (page_id, page_namespace, page_title, page_restrictions, page_is_redirect, ...)
TITLE_IDX  = 2
NS_IDX     = 1

TR_UP  = str.maketrans("iıüşöçg", "İIÜŞÖÇĞ")
TR_LOW = str.maketrans("İIÜŞÖÇĞ", "iıüşöçg")
def tr_lower(s: str) -> str:
    return s.translate(TR_LOW).lower()
def tr_upper_first(s: str) -> str:
    if not s: return s
    for i, ch in enumerate(s):
        if ch.isalpha():
            head = ch.translate(TR_UP).upper()
            return s[:i] + head + s[i+1:]
    return s

SAFE_MAP = {
    "\u2019": "'", "\u2018": "'", "\u201C": '"', "\u201D": '"',
    "\u2013": "-", "\u2014": "-",
    "\u2026": "...",
    "\u00A0": " ", "\u200B": "", "\u200C": "", "\u200D": "", "\uFEFF": "",
}
def sanitize_unicode(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = "".join(SAFE_MAP.get(ch, ch) for ch in s)
    return unicodedata.normalize("NFC", s)

# Parantez içindeki tuple’ları güvenle ayır (tek tırnak kaçışlarını gözet)
def split_mysql_values_block(block: str):
    # block: "(...),(...),(...);" şeklindeki bölüm
    i, n = 0, len(block)
    tup = []
    in_str = False
    curr = []
    escapes = False
    res = []

    while i < n:
        ch = block[i]
        if in_str:
            curr.append(ch)
            if escapes:
                escapes = False
            elif ch == "\\":
                escapes = True
            elif ch == "'":
                in_str = False
            i += 1
            continue

        if ch == "'":
            in_str = True
            curr.append(ch)
        elif ch == "(":
            curr = ["("]
        elif ch == ")":
            curr.append(")")
            tup_str = "".join(curr)
            res.append(tup_str)
            curr = []
        else:
            if curr:
                curr.append(ch)
        i += 1
    return res

# Bir tuple string’ini alanlara ayır
def split_tuple_fields(tup_str: str):
    # "(1,0,'Ankara','',0,...)" -> liste
    assert tup_str[0] == "(" and tup_str[-1] == ")"
    s = tup_str[1:-1]
    out = []
    i, n = 0, len(s)
    buf = []
    in_str = False
    esc = False
    while i < n:
        ch = s[i]
        if in_str:
            buf.append(ch)
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == "'":
                in_str = False
            i += 1
            continue
        if ch == "'":
            in_str = True
            buf.append(ch)
        elif ch == ",":
            out.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
        i += 1
    if buf:
        out.append("".join(buf).strip())
    return out

# Tek tırnaklı MySQL alanını çıkar
def unquote_mysql(s: str):
    s = s.strip()
    if len(s) >= 2 and s[0] == "'" and s[-1] == "'":
        inner = s[1:-1]
        # MySQL dump'ta \' kaçışları olabilir
        inner = inner.replace("\\'", "'").replace("\\\\", "\\")
        return inner
    return s

# Token’ı “safe vocab” adayı yapma kriterleri
ALPHA_TR = re.compile(r"^[A-Za-zÇĞİÖŞÜçğıöşü’']+$")
def is_acronym(token: str) -> bool:
    # Tamamen büyük harfler (TR dahil) ve uzunluk kısıtı
    letters = [ch for ch in token if ch.isalpha()]
    return (
        len(token) <= MAX_ACRON and
        letters and all(ch.isupper() for ch in letters)
    )

def is_proper_case(token: str) -> bool:
    # İlk alfa büyük, geri kalan karışık olabilir; tamamen büyük olmasın
    seen_alpha = False
    first_alpha_upper = False
    all_upper = True
    for ch in token:
        if ch.isalpha():
            if not seen_alpha:
                first_alpha_upper = ch.isupper()
                seen_alpha = True
            if not ch.isupper():
                all_upper = False
    return seen_alpha and first_alpha_upper and not all_upper

def tokenize_title(title: str):
    # Basit ayrıştırma: boşluk + apostrof varyantları
    # "Ankara'nın Tarihi" -> ["Ankara'nın", "Tarihi"]
    return [t for t in re.split(r"\s+", title) if t]

def main():
    title_counter = Counter()

    insert_re = re.compile(r"INSERT INTO\s+`?page`?\s+VALUES\s*", re.IGNORECASE)
    with open(SQL_PATH, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not insert_re.search(line):
                continue
            # Bu satır genelde çok uzun olur; tüm parantez bloklarını çıkar
            blocks = split_mysql_values_block(line)
            for tup in blocks:
                fields = split_tuple_fields(tup)
                if len(fields) <= max(TITLE_IDX, NS_IDX):
                    continue
                try:
                    ns  = int(fields[NS_IDX])
                except Exception:
                    continue
                if ns not in NS_FILTER:
                    continue
                raw_title = unquote_mysql(fields[TITLE_IDX])
                raw_title = raw_title.replace("_", " ")
                title = sanitize_unicode(raw_title)

                # Başlığı token’lara ayır
                for tok in tokenize_title(title):
                    # sadece harf + (Türkçe) apostrof
                    if not ALPHA_TR.match(tok):
                        continue
                    if not (MIN_LEN <= len(tok) <= MAX_LEN):
                        continue
                    # Aday kuralları: Proper case veya ACRONYM
                    if is_proper_case(tok) or is_acronym(tok):
                        title_counter[tok] += 1

    # Frekans filtresi
    kept = [w for w, c in title_counter.items() if c >= MIN_COUNT]

    # Basit temizlik: fazlalık apostrof tekilleştirme (düz tırnak → Türkçe kesme işareti isterseniz burada dönüştürün)
    kept = sorted(set(sanitize_unicode(w) for w in kept))

    with open(OUT_TXT, "w", encoding="utf-8") as out:
        for w in kept:
            out.write(w + "\n")

    print(f"Done. Wrote {len(kept)} items to {OUT_TXT}")

if __name__ == "__main__":
    main()
