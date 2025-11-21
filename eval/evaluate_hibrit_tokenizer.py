# src/eval/evaluate_hibrit_tokenizer.py

from pathlib import Path
import sys
import pickle
from typing import Iterable, Tuple, Set, List, Optional

# ---------------------------------------------------------------------------
# Make sure project root (the directory that contains "src/") is on sys.path
# ---------------------------------------------------------------------------
current = Path(__file__).resolve()
for parent in current.parents:
    if (parent / "src").is_dir():
        project_root = parent
        break
else:
    project_root = current.parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
from src.dataset import UnifiedBounItuTreebankLoader, Example
from src.tokenizer.nb_tokenizer import NaiveBayesTokenizer
# Eğer dosyanın adı rule_based_tokenizer.py ise bu satırı değiştir:
# from src.tokenizer.rule_based_tokenizer import RuleBasedBounTokenizer
from src.tokenizer.rule_tokenizer import RuleBasedUnifiedTokenizer

from src.tokenizer.utils import (
    TokenizationInstance,
    collect_tokenization_data,
    predicted_spans_from_text,
)

# ---------------------------------------------------------------------------
# Small helper: P / R / F1 for a single sentence
# ---------------------------------------------------------------------------

def prf1(gold: Set[int], pred: Set[int]) -> Tuple[float, float, float]:
    """
    Compute precision / recall / F1 for a single sentence
    given gold and predicted boundary sets.
    """
    if not gold and not pred:
        return 1.0, 1.0, 1.0

    tp = len(gold & pred)
    fp = len(pred - gold)
    fn = len(gold - pred)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


# ---------------------------------------------------------------------------
# Hybrid evaluation
# ---------------------------------------------------------------------------

def evaluate_hybrid_on_boun(
    nb_tokenizer: NaiveBayesTokenizer,
    rule_tokenizer: RuleBasedUnifiedTokenizer,
    loader: Optional[UnifiedBounItuTreebankLoader] = None,
    max_eval_sentences: Optional[int] = None,
    show_error_examples: int = 10,
) -> None:
    """
    Evaluate:
      - Naive Bayes tokenizer
      - Rule-based tokenizer
      - OR-hybrid (boundary = NB ∪ Rule)
      - AND-hybrid (boundary = NB ∩ Rule)

    on BOUN sentences using the *same* tokenization instances that were
    used for training the NB tokenizer (via collect_tokenization_data).

    Uses character-level boundary sets derived from gold and predicted
    spans over the sentence text.
    """

    # Eğer dışarıdan loader verilmediyse, default bir loader oluştur
    if loader is None:
        loader = UnifiedBounItuTreebankLoader(
            boun_detok=True,
            boun_mwe_aware=True,
        )
        # TODO: path'i kendi dataset yapına göre değiştir
        loader.add_boun("data/UD_Turkish-BOUN_v2.11_unrestricted-main/test-unr.conllu")

    # NB eğitiminde de kullanılan utility:
    instances: List[TokenizationInstance] = collect_tokenization_data(
        loader,
        domains=["boun"],
        max_sentences=max_eval_sentences,
    )

    if not instances:
        print("No instances collected for evaluation. Check your dataset paths.")
        return

    nb_prec_sum = nb_rec_sum = nb_f1_sum = 0.0
    rb_prec_sum = rb_rec_sum = rb_f1_sum = 0.0
    or_prec_sum = or_rec_sum = or_f1_sum = 0.0
    and_prec_sum = and_rec_sum = and_f1_sum = 0.0

    n_sent = 0
    error_examples = []

    for inst in instances:
        text: str = inst.text
        gold_boundaries: Set[int] = inst.boundaries
        gold_tokens: List[str] = inst.gold_tokens
        domain: str = inst.domain
        sent_id: Optional[str] = inst.sent_id

        # --- Naive Bayes tokenizer prediction ---
        nb_tokens = nb_tokenizer.tokenize(text)
        nb_spans = predicted_spans_from_text(text, nb_tokens)
        nb_boundaries = {end for (_, end) in nb_spans}

        # --- Rule-based tokenizer prediction ---
        rb_tokens = rule_tokenizer.tokenize(text, domain=inst.domain)
        rb_spans = predicted_spans_from_text(text, rb_tokens)
        rb_boundaries = {end for (_, end) in rb_spans}

        # --- Hybrids ---
        or_boundaries   = nb_boundaries | rb_boundaries
        and_boundaries  = nb_boundaries & rb_boundaries

        nb_p, nb_r, nb_f = prf1(gold_boundaries, nb_boundaries)
        rb_p, rb_r, rb_f = prf1(gold_boundaries, rb_boundaries)
        or_p, or_r, or_f = prf1(gold_boundaries, or_boundaries)
        and_p, and_r, and_f = prf1(gold_boundaries, and_boundaries)

        nb_prec_sum  += nb_p
        nb_rec_sum   += nb_r
        nb_f1_sum    += nb_f

        rb_prec_sum  += rb_p
        rb_rec_sum   += rb_r
        rb_f1_sum    += rb_f

        or_prec_sum  += or_p
        or_rec_sum   += or_r
        or_f1_sum    += or_f

        and_prec_sum += and_p
        and_rec_sum  += and_r
        and_f1_sum   += and_f

        # Hata örneği: NB veya Rule gold'dan farklıysa kaydet
        if len(error_examples) < show_error_examples:
            if nb_tokens != gold_tokens or rb_tokens != gold_tokens:
                error_examples.append(
                    (text, domain, sent_id, gold_tokens, nb_tokens, rb_tokens)
                )

        n_sent += 1

    def macro(avg_sum: float) -> float:
        return avg_sum / n_sent if n_sent > 0 else 0.0

    print(f"Evaluated on {n_sent} BOUN sentences.\n")

    print("=== Naive Bayes tokenizer ===")
    print(f"  Precision: {macro(nb_prec_sum):.4f}")
    print(f"  Recall   : {macro(nb_rec_sum):.4f}")
    print(f"  F1       : {macro(nb_f1_sum):.4f}\n")

    print("=== Rule-based tokenizer ===")
    print(f"  Precision: {macro(rb_prec_sum):.4f}")
    print(f"  Recall   : {macro(rb_rec_sum):.4f}")
    print(f"  F1       : {macro(rb_f1_sum):.4f}\n")

    print("=== OR-hybrid (boundary = NB ∪ Rule) ===")
    print(f"  Precision: {macro(or_prec_sum):.4f}")
    print(f"  Recall   : {macro(or_rec_sum):.4f}")
    print(f"  F1       : {macro(or_f1_sum):.4f}\n")

    print("=== AND-hybrid (boundary = NB ∩ Rule) ===")
    print(f"  Precision: {macro(and_prec_sum):.4f}")
    print(f"  Recall   : {macro(and_rec_sum):.4f}")
    print(f"  F1       : {macro(and_f1_sum):.4f}\n")

    # Hata örnekleri
    print("Sample error cases (where NB or Rule ≠ GOLD):")
    print("-" * 60)
    for text, domain, sent_id, gold, nb_pred, rb_pred in error_examples:
        print(f"Domain     : {domain}")
        print(f"SentenceID : {sent_id}")
        print(f"TEXT       : {text}")
        print(f"GOLD TOKS  : {gold}")
        print(f"NB TOKS    : {nb_pred}")
        print(f"RULE TOKS  : {rb_pred}")
        print("-" * 60)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    # 1) Trained NB tokenizer'ı yükle
    with open("naive_bayes_tokenizer_boun.pkl", "rb") as f:
        nb_tok: NaiveBayesTokenizer = pickle.load(f)

    # 2) Rule-based tokenizer'ı oluştur
    #    Burada MWE / proper name path'lerini kendi yapına göre doldur.
    rule_tok = RuleBasedUnifiedTokenizer(
        mwe_path=None,          # örn: "data/mwe/turkish_multiword_terms_wiktionary.txt"
        proper_mwe_path=None,   # örn: "data/mwe/proper_names.txt"
    )

    # 3) Loader (pathleri burada özelleştirebilirsin)
    loader = UnifiedBounItuTreebankLoader(
        boun_detok=True,
        boun_mwe_aware=True,
    )
    # BOUN dev/test path'in:
    loader.add_boun("data/UD_Turkish-BOUN_v2.11_unrestricted-main/dev-unr.conllu")
    #loader.add_iwt("data/IWTandTestSmall/SmallTest.withSentenceBegin")

    # 4) Evaluate
    evaluate_hybrid_on_boun(
        nb_tokenizer=nb_tok,
        rule_tokenizer=rule_tok,
        loader=loader,
        max_eval_sentences=5000,
        show_error_examples=10,
    )


if __name__ == "__main__":
    main()
