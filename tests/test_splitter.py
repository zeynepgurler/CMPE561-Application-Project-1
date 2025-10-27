from src.splitter.rule_splitter import RuleSentenceSplitter


def test_split():
    s = RuleSentenceSplitter()
    out = s.split(["Merhaba", "dünya", ".", "Nasılsın", "?"])
    assert len(out) == 2