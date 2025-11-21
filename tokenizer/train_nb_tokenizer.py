"""
train_tokenizer.py

Train a Naive Bayes–based tokenizer using the unified BOUN + ITU Web Treebank
loader and the NaiveBayesTokenizer class.

Assumptions:
    - You have a module `unified_loader.py` that defines:
        * UnifiedBounItuTreebankLoader
        * Example
        * iterate_for_tokenization(...) method on the loader
    - You have a module `nb_tokenizer.py` that defines:
        * NaiveBayesTokenizer

Adjust import paths and dataset paths below according to your project layout.
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

from src.tokenizer.utils import TokenizationInstance, token_f1_for_sentence, collect_tokenization_data
from src.tokenizer.nb_tokenizer import NaiveBayesTokenizer
from src.dataset import UnifiedBounItuTreebankLoader

# ---------------------------------------------------------------------------
# Training + evaluation pipeline
# ---------------------------------------------------------------------------

def split_train_dev(
    instances: List[TokenizationInstance],
    dev_ratio: float = 0.1,
) -> Tuple[List[TokenizationInstance], List[TokenizationInstance]]:
    """
    Simple train/dev split without shuffling (you can add shuffling if desired).
    """
    n = len(instances)
    dev_size = int(n * dev_ratio)
    if dev_size == 0:
        return instances, []

    train = instances[:-dev_size]
    dev = instances[-dev_size:]
    return train, dev


def train_naive_bayes_tokenizer(
    instances: List[TokenizationInstance],
    alpha: float = 1.0,
    suffix_lexicon_path: str | None = None,
    use_apostrophe_features: bool = True,
    use_suffix_features: bool = True,
) -> NaiveBayesTokenizer:
    """
    Train the NaiveBayesTokenizer using our new API.
    """
    # Convert instances into (text, boundary_list)
    nb_instances = [
        (inst.text, inst.boundaries)
        for inst in instances
    ]

    tokenizer = NaiveBayesTokenizer(
        alpha=alpha,
        suffix_lexicon_path=suffix_lexicon_path,
        use_apostrophe_features=use_apostrophe_features,
        use_suffix_features=use_suffix_features,
    )

    tokenizer.fit(nb_instances)
    return tokenizer


def evaluate_tokenizer(
    tokenizer: NaiveBayesTokenizer,
    instances: List[TokenizationInstance],
    max_eval_sentences: int | None = None,
) -> Tuple[float, float, float]:
    """
    Evaluate the tokenizer on a list of instances and return
    macro-averaged (over sentences) precision, recall, F1.
    """
    if not instances:
        return 0.0, 0.0, 0.0

    total_p = total_r = total_f1 = 0.0
    n = len(instances)

    for inst in instances:
        text = inst.text          # this is the same text used in training
        gold_tokens = inst.gold_tokens
        pred_tokens = tokenizer.tokenize(text)

        p, r, f1 = token_f1_for_sentence(gold_tokens, pred_tokens, text)

        total_p += p
        total_r += r
        total_f1 += f1

    precision = total_p / n
    recall = total_r / n
    f1 = total_f1 / n

    return precision, recall, f1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # 1) Initialize loader and register sources
    loader = UnifiedBounItuTreebankLoader(boun_detok=True, boun_mwe_aware=True)

    # Adjust paths according to your local setup
    loader.add_boun("data/UD_Turkish-BOUN_v2.11_unrestricted-main/train-unr.conllu")
    loader.add_iwt("data/IWTandTestSmall/IWT_normalizationerrorsNoUpperCase.withSentenceBegin")

    # 2) Collect data for tokenization
    instances = collect_tokenization_data(
        loader,
        domains=["iwt"],  # or None to use all domains
        max_sentences=None,       # or set a small number for debugging
    )
    print(f"Collected {len(instances)} sentences for tokenization.")

    if not instances:
        print("No instances collected. Check your dataset paths and loader.")
        return

    # 3) Train/dev split
    train_instances, dev_instances = split_train_dev(instances, dev_ratio=0.1)
    print(f"Training instances: {len(train_instances)}")
    print(f"Dev instances     : {len(dev_instances)}")

    # 4) Train Naive Bayes tokenizer
    suffix_path = "C:/Users/gurle/OneDrive/Masaüstü/NLP/application_project1/tokenization_resources/tr_suffix_lexicon.txt"

    tokenizer = train_naive_bayes_tokenizer(
        train_instances,
        alpha=1.0,
        suffix_lexicon_path=suffix_path,
        use_apostrophe_features=True,
        use_suffix_features=True,
    )

    # 5) Evaluate on dev set
    if dev_instances:
        p, r, f1 = evaluate_tokenizer(
            tokenizer,
            dev_instances,
            max_eval_sentences=None,
        )
        print("\nDev set tokenization performance (span-based macro-average):")
        print(f"  Precision: {p:.4f}")
        print(f"  Recall   : {r:.4f}")
        print(f"  F1       : {f1:.4f}")
    else:
        print("No dev set to evaluate on (dev_ratio too small?).")

    # 6) (Optional) Save the trained tokenizer with pickle
    #    You can uncomment this if you want to persist the model to disk.
    #
    # import pickle
    with open("naive_bayes_tokenizer_hibrit.pkl", "wb") as f:
        pickle.dump(tokenizer, f)
        print("Saved tokenizer to naive_bayes_tokenizer.pkl")


if __name__ == "__main__":
    main()
