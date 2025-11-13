import conllu
import pandas as pd
import string

# conllu.SentenceList  --> a list of TokenList's, sentences
# conllu.TokenList     --> a list of word dictionaries in a sentence
# conllu.Token         --> conllu features of the relevant word: [id, form, lemma, upos, xpos, feat]

#loader = dataset.main()
train = "UD_Turkish-BOUN_v2.11_unrestricted-main/train-unr.conllu"
test = "UD_Turkish-BOUN_v2.11_unrestricted-main/test-unr.conllu"


def extract_suffix_lexicon(path):
    with open(path, "r",encoding="utf-8") as f:
        sentences = conllu.parse(f.read())  # SentenceList

    suffix_lexicon = []
    gold = []

    for sentence in sentences:              # TokenList
        for token in sentence:              # Token
            id = token["id"]
            if isinstance(id, int): 
                form = token["form"].replace("I", "ı").replace("İ", "i").lower()     
                lemma = token["lemma"].replace("I", "ı").replace("İ", "i").lower()
                pos = token["upos"]

                if pos != 'AUX' and form not in string.punctuation:
                    gold.append([form, lemma])
                #print("Simple token")
                #print(f"ID: {id}\tForm: {form}\tLemma: {lemma}\tPOS: {pos}")
 
                if form.startswith(lemma) and form != lemma:
                    suffix = form[len(lemma):]
                    suffix_lexicon.append(suffix)
                    #print(f"Suffix: {suffix}")

            else:
                #print("Complex token", id)
                form = token["form"].lower()
                for token in sentence:
                    lemma = token["lemma"]
                    if form.startswith(lemma) and form != lemma:
                        suffix = form[len(lemma):]
                        suffix_lexicon.append(suffix)
                        gold.append([form, lemma])
                        #print(f"Suffix: {suffix}")
    return suffix_lexicon, gold




suffix_lexicon, gold = extract_suffix_lexicon(train)

lexicon = pd.Series(suffix_lexicon)
lexicon.drop_duplicates(inplace=True)
lexicon = lexicon[lexicon.str.len() != 1]                       # removed one-char suffixes
lexicon.to_csv("suffixes.tsv", sep="\t", index=False)

gold_df = pd.DataFrame(gold)
gold_df = gold_df[gold_df.iloc[:,1].str.len() != 1]
print(f"Before removing duplicates:\t{gold_df.shape}")
#gold_df.drop_duplicates(inplace=True)
#print(f"After removing duplicates:\t{gold_df.shape}")
gold_df.to_csv("gold_stemmer.tsv", sep="\t", index=False) 

