from typing import Dict


def token_boundary_features(text: str, idx: int) -> Dict[str, int]:
    """Features for deciding whether position idx is a token boundary.
    idx points *between* characters: [.. idx-1][idx ..]
    """
    prev = text[idx-1] if idx > 0 else "^"
    curr = text[idx] if idx < len(text) else "$"
    nxt = text[idx+1] if idx+1 < len(text) else "$"
    feats = {
        f"prev={prev}": 1,
        f"curr={curr}": 1,
        f"next={nxt}": 1,
        f"is_space={int(curr.isspace())}": 1,
        f"is_punct={int(curr in ",.;:!?()[]{}\"')]}\-")}": 1,
        f"is_digit_prev={int(prev.isdigit())}": 1,
        f"is_digit_next={int(nxt.isdigit())}": 1,
        f"is_alpha_prev={int(prev.isalpha())}": 1,
        f"is_alpha_next={int(nxt.isalpha())}": 1,
    }
    return feats