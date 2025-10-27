from joblib import dump
from src.splitter.lr_splitter import LRSentenceSplitter


# tokens and gold sentence boundary indices (token-level)
token_sequences = [
    ["Merhaba", "dünya", "!", "Nasılsın", "?"],
    ["Bugün", "hava", "çok", "güzel", "."],
]

# boundaries at tokens 2 and 4 in the first example, 4 in the second
boundaries = [
    [2, 4],
    [4],
]

lr = LRSentenceSplitter().fit(token_sequences, boundaries)
dump(lr, "models/splitter_lr.pkl")
print("Saved models/splitter_lr.pkl")