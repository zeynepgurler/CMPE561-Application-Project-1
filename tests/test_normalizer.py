from src.normalizer.normalizer import Normalizer


def test_basic_norm():
    n = Normalizer("lexicons/normalization_lexicon.tsv")
    out = n.normalize("Kardesim degil 123 https://example.com")
    assert "kardeşim" in out and "<URL>" in out and "<NUM>" in out