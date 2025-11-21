"""
Author: Zeynep Gürler
Date: 16.11.2025

Evaluation pipeline for tokenization
"""

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

from typing import List, Set, Tuple
import pickle

from src.dataset import UnifiedBounItuTreebankLoader
from src.tokenizer.nb_tokenizer import NaiveBayesTokenizer
from src.tokenizer.rule_tokenizer_iwt_boun import RuleBasedUnifiedTokenizer
from src.tokenizer.utils import TokenizationInstance, token_f1_for_sentence, collect_tokenization_data

def show_error_cases(tokenizer: NaiveBayesTokenizer, instances: List[TokenizationInstance], n_examples: int = 5) -> None:
    """
    Print a few mismatched examples (gold vs predicted) for evaluation.
    """
    print("\nSample error cases:\n")
    shown = 0

    for inst in instances:
        if TEST == "nb":
            pred_tokens = tokenizer.tokenize(inst.text)
        if TEST == "rule":
            pred_tokens = tokenizer.tokenize(inst.text, domain=inst.domain)
        
        if pred_tokens != inst.gold_tokens:
            print("Domain     :", inst.domain)
            print("SentenceID :", inst.sent_id)
            print("TEXT       :", inst.text)
            print("GOLD TOKS  :", inst.gold_tokens)
            print("PRED TOKS  :", pred_tokens)
            print("-" * 60)
            shown += 1
            if shown >= n_examples:
                break

    if shown == 0:
        print("No mismatches found in the evaluated subset.")


def evaluate_tokenizer_on_boun_itu(tokenizer: NaiveBayesTokenizer, max_eval_sentences: int | None = 500, domain = None) -> None:
    """
    Evaluate the given tokenizer on BOUN and ITU Web Treebank.
    Print macro-averaged Precision/Recall/F1.
    """
    # Initialize loader and register sources
    loader = UnifiedBounItuTreebankLoader(boun_detok=True, boun_mwe_aware=True)
    #loader.add_iwt("data/IWTandTestSmall/SmallTest.withSentenceBegin")
    loader.add_boun("data/UD_Turkish-BOUN_v2.11_unrestricted-main/test-unr.conllu")

    debug_show_first_instances(loader, tokenizer, n=5)

    # Collect evaluation instances
    instances = collect_tokenization_data(
        loader,
        domains=["iwt"],
        max_sentences=max_eval_sentences,
    )
    if not instances:
        print("No evaluation instances collected. Check dataset paths or loader.")
        return

    total_p = total_r = total_f1 = 0.0
    n = len(instances)

    for inst in instances:
        text = inst.text          # this is the same text used in training
        gold_tokens = inst.gold_tokens
        pred_tokens = tokenizer.tokenize(text, domain=domain)

        p, r, f1 = token_f1_for_sentence(gold_tokens, pred_tokens, text)

        total_p += p
        total_r += r
        total_f1 += f1

    precision = total_p / n
    recall = total_r / n
    f1 = total_f1 / n

    print(f"Evaluated on {n} sentences.")
    print("Tokenization performance (macro-averaged over sentences):")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall   : {recall:.4f}")
    print(f"  F1       : {f1:.4f}")
    # Optional: show a few error cases
    show_error_cases(tokenizer, instances, n_examples=10)

def debug_show_first_instances(loader, tokenizer=None, n=5):
    print("\n=== DEBUG: First few instances for tokenization ===\n")
    count = 0
    for text, boundaries, ex in loader.iterate_for_tokenization(
        domains=["boun", "iwt"],
        max_sentences=n,
    ):
        print(f"[{count}] domain={ex.domain}, sent_id={ex.sent_id}")
        print("TEXT:")
        print(repr(text))
        print("GOLD TOKENS:")
        print(ex.gold_tokens)
        print("GOLD BOUNDARIES:", sorted(list(boundaries)))
        if tokenizer is not None:
            pred = tokenizer.tokenize(text)
            print("PRED TOKENS:")
            print(pred)
        print("-" * 80)
        count += 1


DOMAIN = ["iwt"] 
TEST = "nb" # rule

def main():

    if TEST == "nb":
        # NB tokenizer
        with open("naive_bayes_tokenizer_hibrit.pkl", "rb") as f:
            tokenizer = pickle.load(f)

    if TEST == "rule":
        # rule-based tokenizer
        tokenizer = RuleBasedUnifiedTokenizer(mwe_path="tokenization_resources/wiktionary_turkish_mwe.txt", proper_mwe_path="normalization_resources/tr_gazetteer_large.txt")
    
    # Initialize loader and register sources
    loader = UnifiedBounItuTreebankLoader(boun_detok=True, boun_mwe_aware=True)
    
    if "iwt" in DOMAIN:
        loader.add_iwt("data/IWTandTestSmall/SmallTest.withSentenceBegin")
    if "boun" in DOMAIN:
        loader.add_boun("data/UD_Turkish-BOUN_v2.11_unrestricted-main/test-unr.conllu")

    debug_show_first_instances(loader, tokenizer, n=5)

    # Collect evaluation instances
    instances = collect_tokenization_data(loader, domains=DOMAIN, max_sentences=5000)
    if not instances:
        print("No evaluation instances collected. Check dataset paths or loader.")
        return

    total_p = total_r = total_f1 = 0.0
    n = len(instances)

    for inst in instances:
        text = inst.text          
        gold_tokens = inst.gold_tokens

        if TEST == "rule":
            pred_tokens = tokenizer.tokenize(text, domain=inst.domain)

        if TEST == "nb":
            pred_tokens = tokenizer.tokenize(text)

        p, r, f1 = token_f1_for_sentence(gold_tokens, pred_tokens, text)

        total_p += p
        total_r += r
        total_f1 += f1

    precision = total_p / n
    recall = total_r / n
    f1 = total_f1 / n

    print(f"Evaluated on {n} sentences.")
    print("Tokenization performance (macro-averaged over sentences):")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall   : {recall:.4f}")
    print(f"  F1       : {f1:.4f}")
    # Optional: show a few error cases
    show_error_cases(tokenizer, instances, n_examples=10)


if __name__ == "__main__":
    main()


