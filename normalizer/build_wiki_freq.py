# build_freq_prior.py
import sys, json, regex as re
from collections import Counter

def tr_lower(s):  # basit TR lower
    return s.translate(str.maketrans("İIÜŞÖÇĞ", "iıüşöçg")).lower()

counter = Counter()
word_re = re.compile(r"\p{L}+", re.UNICODE)

for path in sys.argv[1:]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            for w in word_re.findall(line):
                counter[tr_lower(w)] += 1

# çok nadirleri kırp (ör. >=2)
freq = {w: int(c) for w, c in counter.items() if c >= 2}
with open("freq_prior_trwiki.json", "w", encoding="utf-8") as f:
    json.dump(freq, f, ensure_ascii=False)
