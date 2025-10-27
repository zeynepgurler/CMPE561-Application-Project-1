# Example: trains NaiveBayesTokenizer on small toy data
from joblib import dump
from src.tokenizer.nb_tokenizer import NaiveBayesTokenizer


texts = [
    "Merhaba dünya! Nasılsın?",
    "Bugün hava çok güzel.",
]

# Gold tokenization for texts above
gold = [
    ["Merhaba", "dünya", "!", "Nasılsın", "?"],
    ["Bugün", "hava", "çok", "güzel", "."],
]

nb = NaiveBayesTokenizer().fit(texts, gold)
dump(nb, "models/tokenizer_nb.pkl")
print("Saved models/tokenizer_nb.pkl")