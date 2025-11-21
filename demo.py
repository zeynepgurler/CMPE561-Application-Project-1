from pathlib import Path
import sys

current = Path(__file__).resolve()
for parent in current.parents:
    if (parent / "src").is_dir():
        project_root = parent
        break
else:
    project_root = current.parent  

# Add the project root path into sys
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pickle
from src.tokenizer.rule_tokenizer_iwt_boun import RuleBasedUnifiedTokenizer

DOMAIN = ["iwt"] 
TEST = "nb" # rule

if TEST == "nb":
    # NB tokenizer
    with open("naive_bayes_tokenizer_hibrit.pkl", "rb") as f:
        tokenizer = pickle.load(f)

if TEST == "rule":
    # rule-based tokenizer
    tokenizer = RuleBasedUnifiedTokenizer(mwe_path="tokenization_resources/wiktionary_turkish_mwe.txt", proper_mwe_path="normalization_resources/tr_gazetteer_large.txt")

print(tokenizer.tokenize("hey ben zeynep")) # domain="boun"

from src.normalizer.advanced_normalizer import UniversalAdvancedNormalizer
from zemberek.start_zemberek import ZemberekAnalyzer
import json

from src.eval.evaluate_normalization import load_gazetteer_txt

with open("normalization_resources/safe_vocab.txt", "r", encoding="utf-8") as f:
        safe_vocab = {line.strip() for line in f if line.strip()}

with open("normalization_resources/wiki_freq.json", "r", encoding="utf-8") as f:
    boun_freq = json.load(f)

gaz = load_gazetteer_txt("normalization_resources/tr_gazetteer_large.txt")

morph = ZemberekAnalyzer() #jar_path="zemberek/zemberek-full.jar")

advanced = UniversalAdvancedNormalizer(
        lexicon_path="normalization_resources/lexicon.tsv",
        safe_vocab=safe_vocab,
        boun_freq=boun_freq,
        freq_ok=25, # look up freq of word in safe vocab to decide, works really well
        allow_on_noisy=True, 
        proper_noun_gazetteers=gaz, # gazetteer
        morph_analyzer=morph,      # zemberek
        lv_mode="soft",        # mode for zemberek  
        use_slang=True,         
        use_accent_norm=True,   # gidiyom to gidiyorum
        use_vowel_restoration=False,  # capitaliza first word in sent
    )

print(advanced.normalize("hey nbr ben zeynep"))

