from typing import List, Set
from sklearn.linear_model import LogisticRegression
from features.sentence_features import generate_boundary_feats
from nltk import word_tokenize
import conllu
import pandas as pd
from eval.evaluate import eval_sentence_splitting


train = "../UD_Turkish-BOUN_v2.11_unrestricted-main/train-unr.conllu"
test = "../UD_Turkish-BOUN_v2.11_unrestricted-main/test-unr.conllu"


def loader(path: str, length: int, test: bool = False):
    with open(path, "r", encoding="utf-8") as f:
        sentences = conllu.parse(f.read())

    if test:
        # flat list of all tokens in the first `length` sentences
        tokens = []
        for sentence in sentences[:length]:
            sent_text = sentence.metadata["text"]
            tokens.extend(word_tokenize(sent_text))
        return tokens
    else:
        # list of tokenized sentences
        tokenized_sentences = []
        for sentence in sentences[:length]:
            sent_text = sentence.metadata["text"]
            tokenized_sentences.append(word_tokenize(sent_text))
        return tokenized_sentences



ABBREV_SET = {"dr.", "sn.", "örn.", "alb.", "prof.", "bkz.", "mr.", "ms.", "vb.", "vs.", "ar-ge", "doç.", "dzl.",
              "ed.", "ekon.", "ens.", "fak.", "fel.", "fiz.", "fizy.", "gn.", "geom.", "gr.", "haz.", "hek.", "huk",
              "is.", "jeol.", "kim.", "koor.", "kr.", "krş.", "ltd.", "man.", "mat.", "mec.", "müz.", "no.", "ör.",
              "rus.", "rum.", "sf.", "sp.", "sos", "t.c.", "tar.", "tek.", "tel.", "telg.", "tic.", "tğm.", "tiy.",
              "tls.", "vd.", "vet.", "ünl.", "yy.", "zool.", "zm.", "sa.", "adr.", "alm.", "av.", "ecz.", "öğr.",
              "şti.", "a.ş.", "tıp.", "müh.", "u.s.a.", "hzn.", "ph.d.", "m.sc.", "st.", "ave."}




#sentences --> list of lists
sent_tk = loader(train, 10000)
sent_test = loader(test, 1000, True)


feats = ["snt_end", "punct", "abbrev", "next_cap", "prev_not_cap", "nxt_punct", "prev_punct"]          
feature_set = generate_boundary_feats(sent_tk, ABBREV_SET)
ftrs_df = pd.DataFrame(feature_set, columns=["token", "features"])
ftrs_df[feats] = ftrs_df['features'].apply(pd.Series)

tst_sent_tk = loader(test, 1000)
tst_feature_set = generate_boundary_feats(tst_sent_tk, ABBREV_SET)
tst_df = pd.DataFrame(tst_feature_set, columns=["token", "features"])
tst_df[feats] = tst_df['features'].apply(pd.Series)


#print(feature_set)
#print(ftrs_df)


class LRSentenceSplitter:
    def __init__(self):
        self.clf = LogisticRegression(max_iter=500, class_weight='balanced')
        self.fitted = False
        self.feature_columns = None

    def fit_from_df(self, df: pd.DataFrame, feature_columns: List[str]):
        """
        Train the logistic regression sentence splitter from a DataFrame.

        Args:
            df: DataFrame containing boolean feature columns and 'snt_end' label.
            feature_columns: list of column names to use as features.
        """
        self.feature_columns = feature_columns
        X = df[self.feature_columns]          
        y = df['snt_end']                     

        self.clf.fit(X, y)
        self.fitted = True
        return self

    def split_from_df(self, df: pd.DataFrame) -> List[List[str]]:
        """
        Predict sentence boundaries from a DataFrame with the same features.
        Assumes the DataFrame contains a 'token' column and the feature columns.

        Returns:
            List of sentences (each sentence is a list of tokens).
        """
        assert self.fitted, "Call fit_from_df() first"
        X = df[self.feature_columns]
        preds = self.clf.predict(X)

        sents, cur = [], []
        for i, tok in enumerate(df['token']):
            cur.append(tok)
            if preds[i]:  # True means sentence boundary
                sents.append(cur)
                cur = []
        if cur:
            sents.append(cur)
        return sents
    
    def split_from_tokens(self, token_list: List[str], abbrev: Set[str]) -> List[List[str]]:
        """
        Split a flat token list into sentence token lists using the trained model.
        """
        assert self.fitted, "Call fit_from_df() first"

        df = pd.DataFrame({"token": token_list})
        df["punct"] = df["token"].str.match(r'^[.!?…]$')
        df["prev_punct"] = df["punct"].shift(1, fill_value=False)
        df["nxt_punct"] = df["punct"].shift(-1, fill_value=False)
        df["prev_not_cap"] = ~df["token"].shift(1).fillna("A").str.match(r'^[A-ZÇĞİÖŞÜ]')
        df["next_cap"] = df["token"].shift(-1).fillna("a").str.match(r'^[A-ZÇĞİÖŞÜ]')   
        df["abbrev"] = df["token"].isin(abbrev)


        return self.split_from_df(df)

feature_cols = ['punct', "abbrev", "next_cap", 'nxt_punct', "prev_not_cap", 'prev_punct']
splitter = LRSentenceSplitter()
if splitter.fit_from_df(ftrs_df, feature_cols):
    print("Done!!")


sentence_tokens = splitter.split_from_tokens(sent_test, ABBREV_SET)

for sentence in sentence_tokens:    
    print(sentence)

pred_sents = splitter.split_from_df(tst_df)
p, r, f1 = eval_sentence_splitting(pred_sents, sent_test)
print("Precision:", p)  #0.08058608058608059
print("Recall:", r)     #1
print("F1:", f1)        #0.14915254237288136