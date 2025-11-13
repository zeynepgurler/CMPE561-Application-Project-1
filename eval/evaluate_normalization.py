"""
Author: Zeynep Gürler
Date: 01.11.2025

Evaluation pipeline for text normalization.

What it does:
- Normalizes each example with Normalizer
- Computes metrics: Sentence Exact Match, Token Accuracy,
  and Edit Precision/Recall/F1 (change-level)
- Prints a few mismatched examples with inline DIFF:
    RAW : safak gelcek şiştiiiiiiiiii
    PRED: Şafak gelecek şiştii
    GOLD: şafak gelecek şişti
    DIFF: [Şafak -> şafak] gelecek [şiştii -> şişti]
"""

from typing import Iterable, List, Tuple, Dict, Set
import unicodedata
import regex as re
import json

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


from src.dataset import UnifiedBounItuTreebankLoader, Example
from src.normalizer.normalizer import UniversalSimpleNormalizer
from src.normalizer.advanced_normalizer import UniversalAdvancedNormalizer

from zemberek.start_zemberek import ZemberekAnalyzer


# ==============================
# Tokenization and masking (eval)
# ==============================

def _tokize(s: str) -> List[str]:
    """Very simple whitespace tokenization."""
    return s.split()

RE_URL   = re.compile(r"(?i)\bhttps?://\S+")
RE_EMAIL = re.compile(r"(?i)\b[\w.\-+%]+@[\w.\-]+\.[A-Za-z]{2,}\b")
RE_NUM   = re.compile(r"(?<!\p{L})\d+(?:[.,]\d+)?(?!\p{L})")

def mask_eval(s: str) -> str:
    """Mask URL/EMAIL/NUM so they don't dominate metrics."""
    s = RE_URL.sub("<URL>", s)
    s = RE_EMAIL.sub("<EMAIL>", s)
    s = RE_NUM.sub("<NUM>", s)
    return s


# ==============================
# Pretty inline DIFF
# ==============================

def inline_diff(pred: str, gold: str) -> str:
    """
    Show only wrong tokens as [pred -> gold], print correct tokens as-is.
    Aligns by index; if lengths differ, prints '' on the missing side.
    """
    pt = _tokize(pred)
    gt = _tokize(gold)
    out = []
    m = max(len(pt), len(gt))
    for i in range(m):
        p = pt[i] if i < len(pt) else "∅"
        g = gt[i] if i < len(gt) else "∅"
        out.append(g if p == g else f"[{p} -> {g}]")
    return " ".join(out)

def print_case(raw: str, pred: str, gold: str) -> None:
    print(f"RAW : {raw}")
    print(f"PRED: {pred}")
    print(f"GOLD: {gold}")
    print(f"DIFF: {inline_diff(pred, gold)}")


# ==============================
# Edit-level PRF counters
# ==============================

def edit_counts(raw_t: List[str], pred_t: List[str], gold_t: List[str]) -> Tuple[int, int, int]:
    """
    Count TP/FP/FN at token positions:
      TP: raw!=gold and pred==gold        (correct change)
      FP: raw==gold and pred!=raw         (unnecessary change / over-normalization)
      FN: raw!=gold and pred!=gold        (missed or wrong change)
    """
    m = min(len(raw_t), len(pred_t), len(gold_t))
    TP = FP = FN = 0
    for i in range(m):
        r, p, g = raw_t[i], pred_t[i], gold_t[i]
        need = (r != g)
        changed = (p != r)
        correct = (p == g)
        if need and correct:
            TP += 1
        elif not need and changed:
            FP += 1
        elif need and not correct:
            FN += 1
    return TP, FP, FN


# ==============================
# ITU tag strip
# ==============================

RE_ITU_TAG = re.compile(r'@([A-Za-z_]+)\[([^\]]*)\]')

def strip_itu_tags(s: str, drop_tags: set[str] = frozenset()) -> str:
    """
    @tag[inner] -> inner (drop the tag)
    drop_tags: {'url','email'}
    """
    def _repl(m):
        tag = m.group(1).lower()
        inner = m.group(2)
        if tag in drop_tags:
            return ""
        return inner
    return RE_ITU_TAG.sub(_repl, s)


# ==============================
# Gazetteer
# ==============================


def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)

def tr_lower(s: str) -> str:
    # senin aynı yardımcı fonksiyonunla tutarlı olsun
    return s.translate(str.maketrans("İIÜŞÖÇĞ", "iıüşöçg")).lower()

def load_gazetteer_by_label(path: str) -> Dict[str, Set[str]]:
    """
    TSV: name<TAB>LABEL  (name lowercase ise bile normalize ediyoruz)
    Dönen: {"PERS": set(...), "LOC": set(...), ...}
    """
    buckets: Dict[str, Set[str]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "\t" not in line:
                continue
            name, label = line.split("\t", 1)
            name = tr_lower(_nfc(name.strip()))
            label = label.strip().upper()
            if not name or not label:
                continue
            buckets.setdefault(label, set()).add(name)
    return buckets


# ==============================
# Morphological analyzer
# ==============================

class NoOpAnalyzer:
    def analyze(self, token: str):
        # Her token'ı "geçerli" say (general-domain güvenli)
        return [("DUMMY",)]
    
morph = ZemberekAnalyzer(jar_path="../zemberek/zemberek-full.jar")

# ==============================
# Core evaluation
# ==============================

def evaluate_dataset(norm, examples: Iterable["Example"], mask: bool = True, max_err: int = 5, strip: bool = False) -> Dict[str, float]:
    """
    Evaluate a normalizer over examples.
    Returns dict with sent_exact, token_acc, edit_prec, edit_rec, edit_f1.
    Prints up to max_err mismatches with inline DIFF.
    """
    sent_ok = sent_tot = 0
    tok_ok = tok_tot = 0
    TP = FP = FN = 0
    shown = 0

    for ex in examples:
        pred = norm.normalize(ex.raw_text)
        pred = norm.unmask(pred)
        gold = ex.gold_text
        if strip:
            gold = strip_itu_tags(gold)
        raw  = ex.raw_text

        if mask:
            pred_m = mask_eval(pred)
            gold_m = mask_eval(gold)
            raw_m  = mask_eval(raw)
        else:
            pred_m, gold_m, raw_m = pred, gold, raw

        # Sentence exact match
        if pred_m == gold_m:
            sent_ok += 1
        sent_tot += 1

        # Token accuracy
        pt, gt = _tokize(pred_m), _tokize(gold_m)
        m = min(len(pt), len(gt))
        tok_ok += sum(1 for i in range(m) if pt[i] == gt[i])
        tok_tot += len(gt)

        # Edit PRF
        tp, fp, fn = edit_counts(_tokize(raw_m), pt, gt)
        TP += tp; FP += fp; FN += fn

        # Show a few mismatches
        if shown < max_err and pred_m != gold_m:
            print("-" * 68)
            print_case(raw_m, pred_m, gold_m)
            shown += 1

    prec = TP / (TP + FP) if (TP + FP) else 0.0
    rec  = TP / (TP + FN) if (TP + FN) else 0.0
    f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0

    return {
        "sent_exact": sent_ok / sent_tot if sent_tot else 0.0,
        "token_acc":  tok_ok  / tok_tot if tok_tot  else 0.0,
        "edit_prec":  prec,
        "edit_rec":   rec,
        "edit_f1":    f1,
    }


def evaluate_split(norm, data: List["Example"], title: str, mask: bool = True, max_err: int = 5, strip: bool = False) -> Dict[str, float]:
    """Evaluate a subset and pretty-print a summary."""
    print(f"\n=== {title} ===")
    stats = evaluate_dataset(norm, data, mask=mask, max_err=max_err, strip=strip)
    print(f"\nSummary ({title})")
    print(f"- Sentence Exact : {stats['sent_exact']:.3f}")
    print(f"- Token Accuracy : {stats['token_acc']:.3f}")
    print(f"- Edit Precision : {stats['edit_prec']:.3f}")
    print(f"- Edit Recall    : {stats['edit_rec']:.3f}")
    print(f"- Edit F1        : {stats['edit_f1']:.3f}")
    return stats


# ==============================
# Demo main
# ==============================

def main():
    # Load data
    ul = UnifiedBounItuTreebankLoader(boun_detok=True)

    ul.add_boun("../datasets/UD_Turkish-BOUN_v2.11_unrestricted-main/test-unr.conllu")
    ul.add_iwt("../datasets/IWTandTestSmall/SmallTest.withSentenceBegin")

    data = list(ul.iterate())
    boun = [ex for ex in data if ex.domain == "boun"]
    iwt  = [ex for ex in data if ex.domain == "iwt"]

    with open("normalization_resources/safe_vocab.txt", "r", encoding="utf-8") as f:
        safe_vocab = {line.strip() for line in f if line.strip()}

    with open("normalization_resources/boun_freq.json", "r", encoding="utf-8") as f:
        boun_freq = json.load(f)

    # Build normalizer
    norm = UniversalSimpleNormalizer(
        lexicon_path="normalization_resources/lexicon.tsv",
        use_masking=True,
        use_diacritics=True,
        use_edit_distance=True,
        safe_vocab=safe_vocab,
        boun_freq=boun_freq,
        freq_ok=5,            
        allow_on_noisy=True
    )

    gaz = load_gazetteer_by_label("normalization_resources/proper_gazetteer_tr.tsv")

    # proper_noun_gazetteers parametresi "list[set]" bekliyor.
    # Birden fazla bağımsız set verirsen, sınıf “kaç sette var/kaç sette yok” oranıyla “özel ad” sinyali üretir.
    proper_sets = [
        gaz.get("PERS", set()),
        gaz.get("LOC", set()),
        gaz.get("ORG", set()),
        gaz.get("PRODUCT", set()),
        gaz.get("WORK", set()),
        gaz.get("EVENT", set()),
        gaz.get("MISC", set()),
    ]

    # ---------------- Normalizer: SIMPLE (baseline) ----------------
    simple = UniversalSimpleNormalizer(
        lexicon_path="normalization_resources/lexicon.tsv",
        use_masking=True,
        use_diacritics=True,
        use_edit_distance=True,
        safe_vocab=safe_vocab,
        boun_freq=boun_freq,
        freq_ok=5,
        allow_on_noisy=True
    )

    # ---------------- Normalizer: ADVANCED (general-domain) ----------------
    # Hook’ları vermesen de general-domain güvenli: NoOpAnalyzer sadece LV için “pass” verir.
    analyzer = NoOpAnalyzer()

    advanced = UniversalAdvancedNormalizer(
        lexicon_path="normalization_resources/lexicon.tsv",
        safe_vocab=safe_vocab,
        boun_freq=boun_freq,
        freq_ok=25,
        allow_on_noisy=True,
        # Advanced toggles
        morph_analyzer=morph,     # şimdilik analizör yoksa
        lv_mode="soft",          # ileride analizör ekleyince yumuşak gate
        use_slang=True,         # genel domain için kapalı
        use_accent_norm=True,   # konservatif
        use_vowel_restoration=False,  # konservatif
    )

    # 3) Evaluate per split (mask specials during eval)
    evaluate_split(advanced, boun, title="BOUN (clean)", mask=True, max_err=5)
    evaluate_split(advanced, iwt,  title="IWT (noisy)",     mask=True, max_err=5, strip=True)
    evaluate_split(advanced, data, title="OVERALL",         mask=True, max_err=5) 


if __name__ == "__main__":
    main()
