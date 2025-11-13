# -*- coding: utf-8 -*-
"""
TR Wikipedia dump'larından (page.sql.gz + categorylinks.sql.gz)
title \t lang \t categories(pipe-separated) çıkarır.
Temiz Mod (Seçenek 2):
 - Yalnızca main namespace (ns=0)
 - Kategorisi olmayan sayfaları at
 - Disambiguation (anlam ayrımı) sayfalarını at
 - Kategorilerde hafif normalize (küçük harf, boşluk/alt çizgi düzeltme,
   TR apostrof 'deki/'daki türü eklere mini temizlik, bazı çoğul eklerini
   (sadece çok kelimeli kategorilerin en sonunda) kaldırma)
Çıktı: pages_tr_clean.tsv
Kullanım:
    python build_pages_tr_clean.py trwiki-latest-page.sql.gz trwiki-latest-categorylinks.sql.gz
"""

import gzip, re, csv, sys, unicodedata
from pathlib import Path
from typing import Dict, List

def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s) if s is not None else ""

def parse_insert_lines(path: str, table: str):
    """Yalnızca ilgili tabloya ait INSERT satırlarını stream eder."""
    openf = gzip.open if str(path).endswith(".gz") else open
    with openf(path, "rt", encoding="utf-8", errors="ignore") as f:
        prefix = f"INSERT INTO `{table}` VALUES"
        for line in f:
            if line.startswith(prefix):
                yield line

def split_values_clause(insert_line: str) -> List[str]:
    """INSERT ... VALUES (...),(...); satırındaki (...) bloklarını kaba ama sağlam biçimde ayırır."""
    values = insert_line.split("VALUES", 1)[1].strip().rstrip(";")
    return re.findall(r"\((.*?)\)", values)

def robust_csv_row_to_cols(row: str) -> List[str]:
    """Parantez içindeki kolonları virgülle ayırırken tırnakları/escape'leri gözeten küçük parser."""
    cols = []
    buf, in_str, esc = "", False, False
    for ch in row:
        if in_str:
            buf += ch
            if ch == "'" and not esc:
                in_str = False
            esc = (ch == "\\")
        else:
            if ch == "'":
                in_str, buf = True, buf + ch
            elif ch == ",":
                cols.append(buf); buf = ""
            else:
                buf += ch
    cols.append(buf)
    return cols

def decode_sql_string(s: str) -> str:
    """'...'(SQL) içeriğini Python string'e çevirir; alt çizgileri boşluk yapar."""
    s = s.strip()
    if s.startswith("'") and s.endswith("'"):
        s = s[1:-1]
        # backslash escape'lerini mümkün olduğunca çöz
        s = s.encode("utf-8", "backslashreplace").decode("unicode_escape")
    return s.replace("_", " ")

# --- Disambiguation (anlam ayrımı) saptama ---

RE_DISAMBIG_TITLE = re.compile(r"\((anlam ayrımı|disambiguation)\)", re.IGNORECASE)

def looks_disambiguation(title: str, cats: List[str]) -> bool:
    if RE_DISAMBIG_TITLE.search(title):
        return True
    for c in cats:
        lc = c.lower()
        # TR + EN yaygın disambiguation kategori kalıpları
        if "anlam ayrımı" in lc or "disambiguation pages" in lc:
            return True
    return False

# --- Kategori hafif normalizasyonu (Seçenek 2) ---

RE_DEKI = re.compile(r"(?i)(')?d[ae]ki\b")    # Türkiye'deki -> Türkiye
RE_NDEKI = re.compile(r"(?i)n[d]eki\b")       # yerleşimindeki -> yerleşimi  (çok yaygın değil, etkisi küçük)

def clean_category(cat: str) -> str:
    """
    Çok hafif temizlik:
      - lowercase + NFC
      - alt çizgi -> boşluk
      - 'deki/'daki eklerini düşür (Türkiye'deki -> Türkiye)
      - yalnızca ÇOK KELİMELİ kategorilerin sonunda 'ları/leri' kaldır
        (tek kelimeli kategorilere dokunma: 'iller' gibi anahtarları bozmayalım)
    """
    c = nfc(cat).replace("_", " ").strip()
    if not c:
        return c
    c = c.lower()

    # "'deki" / "'daki" gibi ekleri kaldır
    c = RE_DEKI.sub("", c)
    c = RE_NDEKI.sub("", c)
    c = re.sub(r"\s{2,}", " ", c).strip()

    # Sadece birden fazla kelime varsa ve en sondaysa 'ları/leri' kaldır
    if " " in c:
        c = re.sub(r"(ları|leri)$", "", c, flags=re.IGNORECASE).strip()

    return c

def extract_page_map(page_sql_gz: str) -> Dict[int, str]:
    """
    page tablosundan: id (ns=0 olanlar) -> title
    """
    id2title = {}
    for ins in parse_insert_lines(page_sql_gz, "page"):
        for row in split_values_clause(ins):
            cols = robust_csv_row_to_cols(row)
            try:
                pid   = int(cols[0])
                ns    = int(cols[1])
                title = decode_sql_string(cols[2])
                title = nfc(title)
                if ns == 0 and title:  # main namespace
                    id2title[pid] = title
            except Exception:
                continue
    return id2title

def extract_categorylinks(cl_sql_gz: str) -> Dict[int, List[str]]:
    """
    categorylinks tablosundan: page_id -> [category,...]
    """
    links = {}
    for ins in parse_insert_lines(cl_sql_gz, "categorylinks"):
        for row in split_values_clause(ins):
            cols = robust_csv_row_to_cols(row)
            try:
                cl_from = int(cols[0])
                cl_to   = decode_sql_string(cols[1])
                cat = nfc(cl_to).replace("_", " ").strip()
                if cat:
                    links.setdefault(cl_from, []).append(cat)
            except Exception:
                continue
    return links

def main():
    if len(sys.argv) < 3:
        print("Kullanım: python build_pages_tr_clean.py trwiki-latest-page.sql.gz trwiki-latest-categorylinks.sql.gz")
        sys.exit(1)

    page_sql, cl_sql = sys.argv[1], sys.argv[2]
    print("[1/4] page.sql.gz okunuyor...")
    id2title = extract_page_map(page_sql)
    print(f"  -> main namespace sayfa: {len(id2title):,}")

    print("[2/4] categorylinks.sql.gz okunuyor...")
    clmap = extract_categorylinks(cl_sql)
    print(f"  -> kategori eşleşmesi olan sayfa: {len(clmap):,}")

    print("[3/4] filtreleme + temizleme...")
    out_path = "pages_tr_clean.tsv"
    kept = 0
    with open(out_path, "w", encoding="utf-8", newline="") as w:
        wr = csv.writer(w, delimiter="\t")
        for pid, title in id2title.items():
            cats = clmap.get(pid, [])
            if not cats:
                continue

            # disambiguation'ları ele
            if looks_disambiguation(title, cats):
                continue

            # kategori temizlik
            norm_cats = []
            for c in cats:
                cc = clean_category(c)
                if cc:
                    norm_cats.append(cc)
            if not norm_cats:
                continue

            # tekrarları at
            norm_cats = sorted(set(norm_cats))
            wr.writerow([title, "tr", "|".join(norm_cats)])
            kept += 1

    print(f"[4/4] yazıldı: {out_path} (satır: {kept:,})")
    print("Bitti ✓")

if __name__ == "__main__":
    main()
