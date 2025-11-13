import csv, re, sys
from collections import Counter, defaultdict
from pathlib import Path

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

# ====== KULLANICI AYARLARI ======
TWEETS_TSV = "../datasets/tweetS.txt"        # TweetS export (tab-separated, headers)
TSV2_FREQ  = "normalizer/resources/tsv2_word_freq_clean.tsv"       # TS Corpus v2 frekans listesi (word,freq) ya da (lemma,freq)
IS_LEMMA_FREQ = False                   # TS v2 dosyan 'lemma,freq' ise True yap

OUT_LEXICON = "normalizer/resources/tweet_normalization_lexicon.tsv"

ALPHA = 1.0   # frekans ağırlığı
BETA  = 2.0   # edit distance cezası
MAX_CANDIDATES = 6   # her noisy için en çok şu kadar aday tut

# ====== Yardımcılar ======
def turk_casefold(s: str) -> str:
    # Python casefold iyi ama dotted/dotless I için kontrol faydalı
    return s.casefold()

def de_elongate(w: str) -> str:
    # 3+ tekrarları 2'ye indir (cooool -> cool)
    return re.sub(r'(.)\1{2,}', r'\1\1', w)

ASCII2DIACRITIC = {
    'c': ['c', 'ç'],
    'g': ['g', 'ğ'],
    'o': ['o', 'ö'],
    'u': ['u', 'ü'],
    's': ['s', 'ş'],
    'i': ['i', 'ı'],  # dikkat: her yerde doğru olmayabilir; yine de aday üretmek için faydalı
    'I': ['ı', 'i'],  # güvenlik
}

def diacritic_variants(w: str, max_variants=64) -> set:
    # Basit varyant üretici: riskli combinatorial patlamayı sınırlayalım
    w = turk_casefold(w)
    pools = []
    for ch in w:
        pools.append(ASCII2DIACRITIC.get(ch, [ch]))
    # artan maliyeti sınırlamak için backtracking ile max_variants kadar üret
    out = set()
    def backtrack(i, cur):
        if len(out) >= max_variants:
            return
        if i == len(pools):
            out.add("".join(cur))
            return
        for opt in pools[i]:
            cur.append(opt)
            backtrack(i+1, cur)
            cur.pop()
    backtrack(0, [])
    return out

def simple_levenshtein(a: str, b: str) -> int:
    # küçük ve hızlı bir Levenshtein
    a, b = turk_casefold(a), turk_casefold(b)
    if a == b: return 0
    if not a:  return len(b)
    if not b:  return len(a)
    prev = list(range(len(b)+1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur.append(min(
                prev[j] + 1,     # sil
                cur[j-1] + 1,    # ekle
                prev[j-1] + cost # değiştir
            ))
        prev = cur
    return prev[-1]

def read_tweets_noisy_counts(path: str) -> Counter:
    # CQP export: "Query item" kolonundan noisy form frekansları
    cnt = Counter()
    with open(path, "r", encoding="utf-8", newline="") as f:
        rd = csv.DictReader(f, delimiter="\t")
        # başlıkları normalize edelim
        headers = [h.strip().lower().replace(" ", "_") for h in rd.fieldnames]
        rd.fieldnames = headers
        key = "query_item"
        for row in rd:
            q = (row.get(key) or "").strip()
            if not q:
                continue
            cnt[turk_casefold(q)] += 1
    return cnt

def read_tsv2_freq(path: str, is_lemma=False):
    # Esnek okuma: kolon adları word/lemma ve freq olsun
    vocab = {}
    with open(path, "r", encoding="utf-8") as f:
        # virgul veya tab olabilir
        sample = f.read(4096)
        f.seek(0)
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t;")
        rd = csv.DictReader(f, dialect=dialect)
        # kolonları normalize
        rd.fieldnames = [c.strip().lower() for c in rd.fieldnames]
        word_key = "lemma" if is_lemma else "word"
        if word_key not in rd.fieldnames:
            # fallback: ilk kolon
            word_key = rd.fieldnames[0]
        freq_key = "freq" if "freq" in rd.fieldnames else rd.fieldnames[1]
        for row in rd:
            w = (row.get(word_key) or "").strip()
            if not w: 
                continue
            freq = row.get(freq_key)
            try:
                fval = int(float(freq))
            except:
                fval = 0
            vocab[turk_casefold(w)] = vocab.get(turk_casefold(w), 0) + fval
    return vocab

def generate_candidates(noisy: str, vocab: set) -> set:
    # 1) elongation kırp
    base = de_elongate(noisy)
    cand = set([base])

    # 2) diacritics restore
    cand |= diacritic_variants(base)

    # 3) basit harf dönüşümleri (q->k, w->v, x->ks gibi sosyal medya eserleri)
    subs = [
        ("aa", "a"), ("kk", "k"),
        ("q", "k"), ("w", "v"), ("x", "ks"),
        ("sh", "ş"), ("ch", "ç"),
        ("ıi", "ı"), ("ii", "i"),
    ]
    for a, b in subs:
        if a in base:
            cand.add(base.replace(a, b))

    # 4) yalnızca vocab’ta olanları bırak
    return {c for c in cand if c in vocab}

def rank_candidates(noisy: str, candidates: set, vocab_freq: dict) -> list:
    # Skor: ALPHA * global_freq  -  BETA * edit_distance
    scored = []
    for c in candidates:
        f = vocab_freq.get(c, 0)
        d = simple_levenshtein(noisy, c)
        score = ALPHA * f - BETA * d
        scored.append((score, c, f, d))
    scored.sort(reverse=True)
    return scored

def main():
    print("Reading TweetS noisy counts…")
    noisy_counts = read_tweets_noisy_counts(TWEETS_TSV)
    print(f"  noisy types: {len(noisy_counts):,}  tokens: {sum(noisy_counts.values()):,}")

    print("Reading TS v2 frequencies…")
    vocab_freq = read_tsv2_freq(TSV2_FREQ, IS_LEMMA_FREQ)
    vocab = set(vocab_freq.keys())
    print(f"  vocab size: {len(vocab):,}")

    rows_out = []
    for noisy, ncnt in noisy_counts.most_common():
        cands = generate_candidates(noisy, vocab)
        if not cands:
            # hiçbir aday çıkmadıysa, de-elongate + diacritics’ten gelen en yakın 3 kelimeyi
            # sözlüğe sokmamak daha güvenli; yine de debug için boş geçebiliriz
            continue
        ranked = rank_candidates(noisy, cands, vocab_freq)[:MAX_CANDIDATES]
        best = ranked[0]
        best_score, best_cand, best_freq, best_dist = best
        all_cands = ";".join([c for _, c, _, _ in ranked])
        rows_out.append({
            "noisy_form": noisy,
            "best_candidate": best_cand,
            "candidates": all_cands,
            "noisy_count": ncnt,
            "best_freq": best_freq,
            "edit_distance": best_dist,
            "score": round(best_score, 2),
        })

    if not rows_out:
        print("No candidates produced. Check TSV2_FREQ path/format and TweetS export headers.")
        sys.exit(1)

    # yaz
    with open(OUT_LEXICON, "w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=[
            "noisy_form","best_candidate","candidates",
            "noisy_count","best_freq","edit_distance","score"
        ], delimiter="\t")
        wr.writeheader()
        for row in rows_out:
            wr.writerow(row)

    print("Saved:", OUT_LEXICON, f"({len(rows_out)} entries)")

if __name__ == "__main__":
    main()
