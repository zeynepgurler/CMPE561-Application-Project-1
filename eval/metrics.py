from typing import List, Tuple


def precision_recall_f1(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = fn == 0 and 1.0 or tp / (tp + fn)
    f1 = 2*p*r/(p+r) if p+r else 0.0
    return p, r, f1