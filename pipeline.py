from joblib import load
from .normalizer.normalizer import Normalizer
from .tokenizer.rule_tokenizer import RuleTokenizer
from .tokenizer.nb_tokenizer import NaiveBayesTokenizer
from .splitter.rule_splitter import RuleSentenceSplitter
from .splitter.lr_splitter import LRSentenceSplitter
from .stemmer.stemmer import SimpleTurkishStemmer
from .stopwords.static import filter_static


class TurkishPreprocPipeline:
    def __init__(self,
                norm_lex: str = "lexicons/normalization_lexicon.tsv",
                suffixes: str = "lexicons/suffixes.tsv",
                use_ml_tokenizer: bool = True,
                use_ml_splitter: bool = True):
        
        self.normalizer = Normalizer(norm_lex)
        self.rule_tokenizer = RuleTokenizer()
        self.nb_tokenizer = None
        if use_ml_tokenizer:
            try:
                self.nb_tokenizer = load("models/tokenizer_nb.pkl")
            except Exception:
                self.nb_tokenizer = None
        self.rule_splitter = RuleSentenceSplitter()
        self.lr_splitter = None
        if use_ml_splitter:
            try:
                self.lr_splitter = load("models/splitter_lr.pkl")
            except Exception:
                self.lr_splitter = None
        self.stemmer = SimpleTurkishStemmer(suffixes)


    def process(self, text: str):
        t = self.normalizer.normalize(text)
        toks = (self.nb_tokenizer.tokenize(t) if self.nb_tokenizer else self.rule_tokenizer.tokenize(t))
        sents = (self.lr_splitter.split(toks) if self.lr_splitter else self.rule_splitter.split(toks))
        stemmed_sents = [[self.stemmer.stem(tok) for tok in sent] for sent in sents]
        # stopword elimination (static as default)
        sw_removed = [[tok for tok in sent if tok] for sent in stemmed_sents]
        sw_removed = [[tok for tok in sent if tok] for sent in sw_removed]
        return {
            "normalized": t,
            "tokens": toks,
            "sentences": sents,
            "stemmed": stemmed_sents,
            "stopword_filtered": sw_removed,
        }