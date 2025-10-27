from src.tokenizer.rule_tokenizer import RuleTokenizer


def test_tokens():
    t = RuleTokenizer()
    toks = t.tokenize("Merhaba dünya!")
    assert "Merhaba" in toks and "dünya" in toks