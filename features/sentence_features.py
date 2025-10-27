from typing import Dict
import string


SENT_END = set(".!?")


def sentence_boundary_features(tokens: list[str], i: int) -> Dict[str, int]:
    """Features for deciding whether token i ends a sentence."""
    tok = tokens[i]
    next_tok = tokens[i+1] if i+1 < len(tokens) else "<END>"
    feats = {
        f"tok={tok}": 1,
        f"tok_lastchar={tok[-1] if tok else ''}": 1,
        f"next_capital={int(next_tok and next_tok[0:1].isupper())}": 1,
        f"ends_punct={int(tok[-1:] in SENT_END)}": 1,
        f"is_abbrev={int(tok.lower() in {'dr.', 'sn.', 'örn.', 'prof.', 'bkz.'})}": 1,
    }
    return feats