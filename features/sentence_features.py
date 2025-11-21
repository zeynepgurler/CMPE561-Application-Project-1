from typing import Set, List


# token text --> ok
# next token is capitalized --> ok
# next token is punctuation --> ok
# previous token is not capitalized --> ok
# previous token is punctuation --> ok
# is punctuation --> ok
# is abbreviation --> ok
# is proper


def generate_boundary_feats(sentences: List[List[str]], abbrev: Set[str]):
    feature_set = []
    punctuation = ['.', "!", "?", "..."]

    for i, sent in enumerate(sentences):
        for j, token in enumerate(sent):
            features = []
            ftr = []
            features.append(token)

            # sentence end
            if token == sent[-1]: ftr.append(True)
            else: ftr.append(False)

            # is punctuation
            if token in punctuation: ftr.append(True)
            else: ftr.append(False)

            #is abbrev
            if token.replace("I", "ı").replace("İ", "i").lower() in abbrev: ftr.append(True)
            else: ftr.append(False)

            # is proper:
    #        if token.lower() in proper: ftr.append(True)
    #        else: ftr.append(False)

            # next token is capitalized:
            if not token == sent[-1]:
                if sent[j+1][0].isupper(): ftr.append(True)
                else: ftr.append(False)
            else:
                if not sent == sentences[-1]:
                    if sentences[i+1][0][0].isupper(): ftr.append(True)
                    else: ftr.append(False)
                else: ftr.append(False)
           
            # previous token is not capitalized:
            if not token == sent[0]:
                if sent[j-1][0].islower(): ftr.append(True)
                else: ftr.append(False)
            else:
                if not sent == sentences[0]:
                    if sentences[i-1][-1][0].islower(): ftr.append(True)
                    else: ftr.append(False)
                else: ftr.append(False)

            # next token is punctuation:
            if not token == sent[-1]:
                if sent[j+1] in punctuation: ftr.append(True)
                else: ftr.append(False)
            else:
                if not sent == sentences[-1]:
                    if sentences[i+1][0] in punctuation: ftr.append(True)
                    else: ftr.append(False)
                else: ftr.append(False)

            # previous token is punctuation:
            if not token == sent[0]:
                if sent[j-1] in punctuation: ftr.append(True)
                else: ftr.append(False)
            else:
                if not sent == sentences[0]:
                    if sentences[i-1][-1] in punctuation: ftr.append(True)
                    else: ftr.append(False)
                else: ftr.append(False)

            features.append(ftr)
            feature_set.append(features)
    return feature_set