"""
Author: Zeynep Gürler
Date: 02.11.2025

This code process the frequency list of TS V2 Corpus so we can ge rid of the punctuations in the freq list.
"""

import pandas as pd
import csv

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


INPUT_TXT = "../datasets/tsv2_word_freq.txt"     
OUTPUT_TSV = "normalizer/resources/tsv2_word_freq_clean.tsv"

# 1) CSV alan boyutu limitini artır
try:
    csv.field_size_limit(sys.maxsize)
except OverflowError:
    csv.field_size_limit(10**9)

def find_header_row(path: str) -> int:
    """'Number\t(Word|Lemma)\tFrequency' satırının kaçıncı satırda olduğunu bulur (0-based skiprows için)."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if "\t" in line:
                parts = [p.strip().lower() for p in line.strip().split("\t")]
                if len(parts) >= 3 and parts[0].startswith("number") and parts[-1].startswith("frequency"):
                    # ikinci sütun Word veya Lemma olmalı
                    if "word" in parts[1] or "lemma" in parts[1]:
                        return i
    # fallback: çoğu TS Corpus dosyasında ilk 3 satır header/preamble oluyor
    return 3

def load_freq_table(path: str) -> pd.DataFrame:
    hdr_idx = find_header_row(path)
    # hdr_idx satırına kadar atla, sonraki satırı header kabul et
    df = pd.read_csv(
        path,
        sep="\t",
        engine="python",
        skiprows=hdr_idx,   # header satırı dahil EDİLECEK (aşağıda header=None ile manuel isim verirsek)
        header=0,           # skiprows sonrası ilk satır header
        quoting=csv.QUOTE_NONE,
        encoding="utf-8",
        on_bad_lines="skip"
    )
    # Kolon adlarını normalize et
    df.columns = [c.strip().lower() for c in df.columns]
    # 'word' veya 'lemma' kolonunu bul
    word_col = "word" if "word" in df.columns else ("lemma" if "lemma" in df.columns else df.columns[1])
    freq_col = "frequency" if "frequency" in df.columns else df.columns[-1]
    # Sadece bu iki kolonu bırak
    df = df[[word_col, freq_col]].copy()
    df.rename(columns={word_col: "word", freq_col: "frequency"}, inplace=True)
    # Sayıya çevir
    df["frequency"] = pd.to_numeric(df["frequency"], errors="coerce").fillna(0).astype(int)
    return df

def filter_to_words(df: pd.DataFrame) -> pd.DataFrame:
    # Türkçe harfle başlayanları tut (noktalama, sayı, URL vs. dışarı)
    pat = r"^[A-Za-zÇĞİÖŞÜçğıöşüİı]"
    m = df["word"].astype(str).str.match(pat, na=False)
    return df[m].copy()

def main():
    df = load_freq_table(INPUT_TXT)
    df = filter_to_words(df)
    # frekansı en az 2 olanlar (UI’da vermiştik; burada da emniyet için uygula)
    df = df[df["frequency"] >= 2]
    # boşluk temizliği
    df["word"] = df["word"].astype(str).str.strip()
    # kaydet
    df.to_csv(OUTPUT_TSV, sep="\t", index=False)
    print(f"✅ Temiz liste kaydedildi: {OUTPUT_TSV} (rows={len(df):,})")

if __name__ == "__main__":
    main()