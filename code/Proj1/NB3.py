import numpy as np
import pandas as pd
import shutil
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

trainFile = "../../data/train.tsv"
testFile = "../../data/test.tsv"
clfFolder = "../../classifier/NB3/"


def make_train_test(df_train, df_test):
    vectorizer = CountVectorizer(ngram_range=(2, 2))
    
    X_train = vectorizer.fit_transform(df_train['Phrase'].values)
    Y_train = df_train['Sentiment'].values
    X_test = vectorizer.transform(df_test['Phrase'].values)
    return X_train, Y_train, X_test
    
    
    
if __name__ == '__main__':
    
    shutil.rmtree(clfFolder, ignore_errors=True)

    df_train = pd.read_csv(trainFile, sep='\t')
    df_test = pd.read_csv(testFile, sep='\t')
    
    df_output = pd.DataFrame(df_test[['PhraseId']])
    
    X_train, Y_train, X_test = make_train_test(df_train, df_test)
    clf = MultinomialNB()
    clf.fit(X_train, Y_train)
    
    
    df_output['Sentiment'] = clf.predict(X_test)
    
    if not os.path.exists(clfFolder):
        os.makedirs(clfFolder)
    
    score_file = open(clfFolder+"Score.txt", "w")
    score_file.write("Score = {0}".format(clf.score(X_train, Y_train)))
    score_file.close()
    df_output.to_csv(clfFolder + "output.csv", index = False)
    
# Kaggle score - 0.55518