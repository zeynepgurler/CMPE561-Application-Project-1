from typing import List
from utils.io import read_tsv


class SimpleTurkishStemmer:
    """Heuristic stemmer using longest-suffix stripping with a suffix lexicon.
    Handles both inflectional and a subset of derivational suffixes.
    """
    def __init__(self, suffix_path: str):
        rows = read_tsv(suffix_path)
        self.suffixes = sorted([row[0] for row in rows], key=len, reverse=True)


    def final_devoicer(self, stem: str):
        stem_ends = {"b": "p", "c": "ç", "d": "t", "g": "k", "ğ": "k"}
        for end in stem_ends:
            if stem.endswith(end):
                devoiced = stem[:-1] + stem_ends[end] 
                return devoiced
        return stem
      

    def stem(self, token: str) -> str:
        if "'" in token:
            t = token.split("'")[0]
            return t
        else:
            t = token
            for suf in self.suffixes:
                if len(t) <= 3:   # most words with this length dont need stemming
                    continue
                if t.endswith(suf) and len(t) - len(suf) >= 2:
                    t = t[: -len(suf)]
                    break
            t = self.final_devoicer(t)
            return t.replace("I", "ı").replace("İ", "i").lower()


    def stem_sentence(self, tokens: List[str]) -> List[str]:
        return [self.stem(t) for t in tokens]
    

stemmer = SimpleTurkishStemmer("suffixes.tsv")
gold = read_tsv("gold_stemmer.tsv")



x = ['suya', "gidiyor", "diyor", "ediyor", "yiyor", "kitaba", "sebebe", "kazan", "kazana", "ye", "sev", "kadın", "pazar",
     "kapak", "köy", "sepet", "trafik", "trafiğe", "kitab", "sebeb", "ağ", "eleğ", "tac", "seped", "kepeng", "kapağa", "kitabımı",
     "kepengimin", "trafiğimizin"]


stm = stemmer.stem_sentence(x)

cap = ["ankara'nın", "istanbul'un", "trabzon'a", "konya'yı", "Fransızların"]

#print(x)
#print(stm)

# add final devoicing and irregular stems, this would ensure +80% acc
# b-p, c-ç, d-t, g,ğ-k
#neden, diyor --> stem di, yiyor --> stem yi ,önce, bana, sana, istiyorum, burnu, karnı, şehrin, şehri, doğ,