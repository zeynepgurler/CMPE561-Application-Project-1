"""
Build Turkish proper-noun gazetteers from multiple sources:
- Wikipedia titles dump (all titles in ns0)
- Wikidata CSVs (column name 'label' or configurable)
- JRC-Names (TSV/CSV) [optional]
- GeoNames TR dump [optional]

Outputs:
- proper_titles.txt  : multi-word titles (title-level)
- proper_tokens.txt  : single tokens (token-level proper candidates)

Run:
python build_gazetteers.py --wiki ns0_titles.txt \
                           --wikidata people.csv cities.csv orgs.csv \
                           --jrc jrcnames.txt \
                           --geonames TR.txt \
                           --min-count 1
"""

import argparse
import csv
import os
import re
import unicodedata
from collections import Counter
from typing import Iterable, List, Optional

# ----------------------------
# Turkish-aware helpers
# ----------------------------
TR_UP  = str.maketrans("iıüşöçg", "İIÜŞÖÇĞ")
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

ALPHA_TR = re.compile(r"^[A-Za-zÇĞİÖŞÜçğıöşü’'().,-]+$")  # başlıktaki bazı noktalama işaretlerine izin

def is_acronym(token: str, max_len: int = 8) -> bool:
    letters = [ch for ch in token if ch.isalpha()]
    return (letters and all(ch.isupper() for ch in letters) and len(token) <= max_len)

def is_proper_case(token: str) -> bool:
    # İlk alfabetik karakter büyük ve tamamı büyük olmayan
    seen_alpha = False
    first_upper = False
    all_upper = True
    for ch in token:
        if ch.isalpha():
            if not seen_alpha:
                first_upper = ch.isupper()
                seen_alpha = True
            if not ch.isupper():
                all_upper = False
    return seen_alpha and first_upper and not all_upper

def tokenize_title(title: str) -> List[str]:
    return [t for t in re.split(r"\s+", title.strip()) if t]

def normalize_title_line(line: str) -> Optional[str]:
    # Wikipedia lines: may have underscores for spaces
    s = line.strip().replace("_", " ")
    s = sanitize_unicode(s)
    # Drop empty or non-letter-only titles
    if not any(ch.isalpha() for ch in s):
        return None
    # Keep a reasonable length
    if len(s) > 200:
        return None
    return s

# ----------------------------
# Loaders
# ----------------------------
def load_wiki_titles(path: str) -> Counter:
    titles = Counter()
    if not path:
        return titles
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = normalize_title_line(line)
            if not s:
                continue
            # Wiki titles sometimes include namespace prefixes — ns0 dump'ta olmamalı ama yine güvenli olalım
            if not any(ch.isalpha() for ch in s):
                continue
            titles[s] += 1
    return titles

def load_wikidata_labels(paths: List[str], label_col: str = "label") -> Counter:
    titles = Counter()
    for p in paths:
        if not p:
            continue
        if not os.path.exists(p):
            continue
        # auto-detect delimiter
        delim = ","
        if p.lower().endswith(".tsv"):
            delim = "\t"
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f, delimiter=delim)
            # if column not present, try first column
            cols = reader.fieldnames or []
            use_col = label_col if label_col in cols else (cols[0] if cols else None)
            if not use_col:
                continue
            for row in reader:
                val = row.get(use_col, "").strip()
                if not val:
                    continue
                s = sanitize_unicode(val)
                if any(ch.isalpha() for ch in s):
                    titles[s] += 1
    return titles

def load_jrc_names(path: str) -> Counter:
    """
    JRC-Names: genelde TSV; formatları değişebilir.
    Basit yaklaşım: satırdaki ilk metin alanını 'ad' say ve alfabesi olanları ekle.
    """
    titles = Counter()
    if not path or not os.path.exists(path):
        return titles
    delim = "\t" if path.lower().endswith(".tsv") else ","
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split(delim)
            if not parts:
                continue
            name = sanitize_unicode(parts[0])
            if any(ch.isalpha() for ch in name):
                titles[name] += 1
    return titles

def load_geonames_tr(path: str) -> Counter:
    """
    GeoNames TR.txt formatı: tab-separated, 1. kolon geonameid, 2. kolon name, 3. kolon asciiname, 4. alternatenames...
    Biz 2. ve (opsiyonel) 4. kolondan TR görünümlü isimleri çekmeye çalışırız.
    """
    titles = Counter()
    if not path or not os.path.exists(path):
        return titles
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            name = sanitize_unicode(parts[1])
            if any(ch.isalpha() for ch in name):
                titles[name] += 1
            # alternatenames virgülle ayrılır (opsiyonel, çok gürültülü olabilir)
            if len(parts) >= 4 and parts[3]:
                for alt in parts[3].split(","):
                    alt = sanitize_unicode(alt)
                    if any(ch.isalpha() for ch in alt):
                        titles[alt] += 1
    return titles

# ----------------------------
# Merge + filter
# ----------------------------
def build_gazetteers(
    wiki_path: Optional[str],
    wikidata_paths: List[str],
    jrc_path: Optional[str],
    geonames_path: Optional[str],
    out_titles: str,
    out_tokens: str,
    min_count: int = 1,
    min_token_len: int = 2,
    max_token_len: int = 40,
    max_acronym_len: int = 8
):
    titles = Counter()

    if wiki_path:
        t = load_wiki_titles(wiki_path)
        titles.update(t)
        print(f"[wiki] titles +={len(t)}")

    if wikidata_paths:
        t = load_wikidata_labels(wikidata_paths, label_col="label")
        titles.update(t)
        print(f"[wikidata] titles +={len(t)}")

    if jrc_path:
        t = load_jrc_names(jrc_path)
        titles.update(t)
        print(f"[jrc] titles +={len(t)}")

    if geonames_path:
        t = load_geonames_tr(geonames_path)
        titles.update(t)
        print(f"[geonames] titles +={len(t)}")

    # Filter titles
    titles_out = []
    for s, c in titles.items():
        if c < min_count:
            continue
        if not ALPHA_TR.match(s):
            # başlık içinde çok fazla sembol varsa at
            continue
        titles_out.append(sanitize_unicode(s))

    titles_out = sorted(set(titles_out))

    # Token-level candidates
    token_counts = Counter()
    for s in titles_out:
        for tok in tokenize_title(s):
            if not (min_token_len <= len(tok) <= max_token_len):
                continue
            if not re.fullmatch(r"[A-Za-zÇĞİÖŞÜçğıöşü’']+", tok):
                continue
            if is_proper_case(tok) or is_acronym(tok, max_len=max_acronym_len):
                token_counts[tok] += 1

    tokens_out = sorted({sanitize_unicode(w) for w, c in token_counts.items() if c >= min_count})

    # Write
    with open(out_titles, "w", encoding="utf-8") as fo:
        for w in titles_out:
            fo.write(w + "\n")
    with open(out_tokens, "w", encoding="utf-8") as fo:
        for w in tokens_out:
            fo.write(w + "\n")

    print(f"Titles kept: {len(titles_out)}  -> {out_titles}")
    print(f"Tokens kept: {len(tokens_out)}  -> {out_tokens}")

# ----------------------------
# CLI
# ----------------------------
def parse_args():
    ap = argparse.ArgumentParser("Build TR gazetteers (titles & tokens) from multiple sources.")
    ap.add_argument("--wiki", help="Wikipedia all-titles-in-ns0 (unzipped) file path", default="C:/Users/90551/Desktop/NLP/application_project1/datasets/trwiki-all-titles/trwiki-all-titles")
    ap.add_argument("--wikidata", nargs="*", default=[], help="Wikidata CSV/TSV file paths (must include column 'label')")
    ap.add_argument("--jrc", help="JRC-Names TSV/CSV path (optional)")
    ap.add_argument("--geonames", help="GeoNames TR.txt (optional)")
    ap.add_argument("--out-titles", default="proper_titles.txt")
    ap.add_argument("--out-tokens", default="proper_tokens.txt")
    ap.add_argument("--min-count", type=int, default=1)
    ap.add_argument("--min-token-len", type=int, default=2)
    ap.add_argument("--max-token-len", type=int, default=40)
    ap.add_argument("--max-acronym-len", type=int, default=8)
    return ap.parse_args()

def main():
    args = parse_args()
    build_gazetteers(
        wiki_path=args.wiki,
        wikidata_paths=args.wikidata,
        jrc_path=args.jrc,
        geonames_path=args.geonames,
        out_titles=args.out_titles,
        out_tokens=args.out_tokens,
        min_count=args.min_count,
        min_token_len=args.min_token_len,
        max_token_len=args.max_token_len,
        max_acronym_len=args.max_acronym_len
    )

if __name__ == "__main__":
    main()
