import pandas as pd
import numpy as np
from nltk.stem.snowball import SnowballStemmer

trainFile = "../../data/train.tsv"
testFile = "../../data/test.tsv"


def stem(phrase):
    if phrase.strip() == "":
        return phrase

    content = phrase.split()
    list_stemmed = []
    stemmer = SnowballStemmer("english")

    for word in content:
        if word != '\n':                
            list_stemmed.append(stemmer.stem(word))

    words = " ".join(list_stemmed)
    return words

df_train = pd.read_csv(trainFile, sep='\t')
df_test= pd.read_csv(testFile, sep='\t')

df_train['Phrase']= df_train['Phrase'].map(stem)
df_test['Phrase'] = df_test['Phrase'].map(stem)

df_train.to_csv('../../data/train_stemmed.tsv', sep='\t' , index = False) 
df_test.to_csv('../../data/test_stemmed.tsv', sep='\t', index = False)


