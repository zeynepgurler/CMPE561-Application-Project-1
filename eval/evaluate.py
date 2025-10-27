from typing import List
from .metrics import precision_recall_f1


def eval_tokenization(pred: List[str], gold: List[str]):
    sp = set(_boundaries_from_tokens(pred))
    sg = set(_boundaries_from_tokens(gold))
    tp = len(sp & sg); fp = len(sp - sg); fn = len(sg - sp)
    return precision_recall_f1(tp, fp, fn)


def _boundaries_from_tokens(tokens: List[str]):
    idx, bounds = 0, []
    for t in tokens:
        idx += len(t)
        bounds.append(idx)
        idx += 1 # space
    return bounds


def eval_sentence_splitting(pred_sents: List[List[str]], gold_sents: List[List[str]]):
    def bounds(toks):
        idx, out = 0, []
        for i, t in enumerate(toks):
            idx += len(t)
            out.append(idx)
            idx += 1
        return out
    pred_b = [b for s in pred_sents for b in bounds(s)]
    gold_b = [b for s in gold_sents for b in bounds(s)]
    tp = len(set(pred_b) & set(gold_b))
    fp = len(set(pred_b) - set(gold_b))
    fn = len(set(gold_b) - set(pred_b))
    return precision_recall_f1(tp, fp, fn)