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

from typing import Set, Tuple, List
from typing import List, Set, Tuple
from dataclasses import dataclass

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.dataset import UnifiedBounItuTreebankLoader, Example

def build_text_and_boundaries_from_tokens(tokens: List[str]) -> Tuple[str, Set[int]]:
    """
    Reconstruct a raw text from gold tokens and compute the gold token boundaries.

    Example:
        tokens = ["Hello", ",", "world", "!"]
        text = "Hello , world !"
        boundaries = {5, 7, 13, 15}
        (i.e., indices where a token ends in the reconstructed string)

    For tokenization training, we deliberately use a simple
    "tokens + space" scheme so the mapping between characters
    and token boundaries is deterministic and consistent across domains.
    """
    text_parts: List[str] = []
    boundaries: Set[int] = set()
    pos = 0

    for i, tok in enumerate(tokens):
        if i > 0:
            text_parts.append(" ")
            pos += 1  # space
        text_parts.append(tok)
        pos += len(tok)
        boundaries.add(pos)  # token ends at this character index

    text = "".join(text_parts)
    return text, boundaries

def build_text_and_boundaries_from_example(ex: "Example") -> Tuple[str, Set[int]]:
    """
    Use the example's sentence text and its gold tokens to compute
    gold token boundaries on the *real* sentence.

    For BOUN:
        ex.raw_text == ex.gold_text (both like "Fakülteyi bitirenler ... başlıyorlarmış.")
        ex.gold_tokens are UD tokens, e.g. ["Fakülteyi", "bitirenler", ..., ".",]

    For ITU Web:
        raw_text  : noisy / original input
        gold_text : normalized sentence (tokens already match this)
        We prefer to tokenize gold_text, since gold_tokens come from GOLD.
    """
    # Decide which text to tokenize on:
    if ex.domain == "iwt":
        # ITU Web: gold_text and gold_tokens align nicely
        text = ex.gold_text
    else:
        # BOUN (and tweets, if you want): raw_text == the real sentence
        text = ex.raw_text

    boundaries: Set[int] = set()
    idx = 0

    for tok in ex.gold_tokens:
        # Skip whitespace before the token
        while idx < len(text) and text[idx].isspace():
            idx += 1

        # Find the token starting from current index
        start = text.find(tok, idx)
        if start == -1:
            return text, boundaries

        end = start + len(tok)
        boundaries.add(end)
        idx = end

    return text, boundaries

# ---------------------------------------------------------------------------
# Utility functions for token-level evaluation
# ---------------------------------------------------------------------------

def tokens_to_spans(tokens: List[str]) -> Tuple[str, List[Tuple[int, int]]]:
    """
    Convert a gold token list to character spans over the reconstructed text.

    Returns:
        text  : reconstructed string
        spans : list of (start, end) indices for each token
    """
    text, _ = build_text_and_boundaries_from_tokens(tokens)
    spans: List[Tuple[int, int]] = []
    pos = 0

    for i, tok in enumerate(tokens):
        if i > 0:
            pos += 1  # space
        start = pos
        end = start + len(tok)
        spans.append((start, end))
        pos = end

    return text, spans

def align_tokens_to_text(text: str, tokens: List[str]) -> List[Tuple[int, int]]:
    """
    Align a list of tokens to a given text and return (start, end) spans.
    Same logic as in build_text_and_boundaries_from_example, but returns spans.
    """
    spans: List[Tuple[int, int]] = []
    idx = 0

    for tok in tokens:
        while idx < len(text) and text[idx].isspace():
            idx += 1
        start = text.find(tok, idx)
        if start == -1:
            # Alignment failed; stop or skip; for now we stop
            # print(f"[WARN] Could not align token {tok!r} in: {text!r}")
            break
        end = start + len(tok)
        spans.append((start, end))
        idx = end

    return spans

def predicted_spans_from_text(text: str, pred_tokens: List[str]) -> List[Tuple[int, int]]:
    """
    Map predicted tokens back to character spans using a simple greedy search.

    Assumes:
        - tokens appear in order in the text
        - the text was built using the same scheme as in tokens_to_spans / build_text_and_boundaries_from_tokens
    """
    spans: List[Tuple[int, int]] = []
    pos = 0

    for tok in pred_tokens:
        tok = tok.strip()
        if not tok:
            continue
        start = text.find(tok, pos)
        if start == -1:
            # If something went wrong, skip this token to avoid crashing.
            # (In practice this should rarely happen if the tokenizer is consistent.)
            continue
        end = start + len(tok)
        spans.append((start, end))
        pos = end

    return spans


def token_f1_for_sentence(gold_tokens: List[str], pred_tokens: List[str], text: str) -> Tuple[float, float, float]:
    gold_spans = align_tokens_to_text(text, gold_tokens)
    pred_spans = align_tokens_to_text(text, pred_tokens)

    gold_set = set(gold_spans)
    pred_set = set(pred_spans)

    tp = len(gold_set & pred_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - gold_set)

    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)

    return precision, recall, f1


# ---------------------------------------------------------------------------
# Data collection using the loader
# ---------------------------------------------------------------------------

@dataclass
class TokenizationInstance:
    text: str
    boundaries: Set[int]
    gold_tokens: List[str]
    domain: str
    sent_id: str | None


def collect_tokenization_data(
    loader: "UnifiedBounItuTreebankLoader",
    domains: List[str] | None = None,
    max_sentences: int | None = None,
) -> List[TokenizationInstance]:
    """
    Use UnifiedBounItuTreebankLoader.iterate_for_tokenization() to collect
    training instances for tokenization.

    Each instance contains:
        - text:      synthetic training text (tokens + spaces)
        - boundaries: gold boundary positions
        - gold_tokens: gold token list
        - domain / sent_id: metadata from the underlying Example
    """
    instances: List[TokenizationInstance] = []

    for text, boundaries, ex in loader.iterate_for_tokenization(
        domains=domains,
        max_sentences=max_sentences,
    ):
        inst = TokenizationInstance(
            text=text,
            boundaries=set(boundaries),
            gold_tokens=list(ex.gold_tokens),
            domain=ex.domain,
            sent_id=ex.sent_id,
        )
        instances.append(inst)

    return instances

def merge_boun_mwes(sent_tokens: List[dict]) -> List[str]:
        """
        Given a list of UD tokens for a single BOUN sentence,
        merge tokens that form certain MWE groups (flat/fixed/mwe/compound).

        sent_tokens: list of dicts with keys:
            - "id": int
            - "form": str
            - "head": int
            - "deprel": str

        Returns:
            A list of surface forms (strings), where MWE groups are joined
            with spaces, e.g. ["Türk Hava Yolları", "uçuşları", ...].
        """
        MWE_RELS = {"flat", "fixed", "mwe", "compound"}

        # Map: head_id -> set of token_ids that depend on it with MWE_RELS
        groups_by_head = {}

        for tok in sent_tokens:
            tid = tok["id"]
            head = tok["head"]
            deprel = tok["deprel"]
            if deprel in MWE_RELS and head != 0:
                groups_by_head.setdefault(head, set()).add(tid)

        # Build a token_id -> group_head map
        group_head_for = {}
        for head, dep_ids in groups_by_head.items():
            members = set(dep_ids)
            members.add(head)
            for tid in members:
                group_head_for[tid] = head

        out_forms: List[str] = []
        n = len(sent_tokens)

        # Iterate in surface order
        for tok in sent_tokens:
            tid = tok["id"]
            if tid in group_head_for:
                head = group_head_for[tid]
                # Emit group only when we are at the head
                if tid != head:
                    continue
                # Collect all members of this group in surface order
                member_ids = [t["id"] for t in sent_tokens
                            if group_head_for.get(t["id"]) == head]
                member_ids.sort()
                member_forms = [t["form"] for t in sent_tokens
                                if t["id"] in member_ids]
                out_forms.append(" ".join(member_forms))
            else:
                out_forms.append(tok["form"])

        return out_forms
