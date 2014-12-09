import numpy as np
import pandas as pd
import shutil
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.grid_search import GridSearchCV

trainFile = "../../../data/train_morethantwoword.csv"
testFile = "../../../data/test_morethantwoword.csv"
clfFolder = "../../../classifier/NB_morethantwoword1/"

def cv_optimize(X_train, Y_train, clf):
    alpha_range = [1]
    param_grid = dict(alpha=alpha_range)

    gs = GridSearchCV(clf, param_grid = param_grid, cv = 100, n_jobs = 4, verbose = 3)
    gs.fit(X_train, Y_train)
    print "gs.best_params_ = {0}, gs.best_score_ = {1}".format(gs.best_params_, gs.best_score_)
    return gs.best_estimator_, gs.best_params_, gs.best_score_

def fit_clf(X_train, Y_train):
    clf = MultinomialNB() 

    clf, bp, bs = cv_optimize(X_train, Y_train, clf)    
    clf.fit(X_train, Y_train)
    return clf
    
def make_train_test(df_train, df_test):
    vectorizer = CountVectorizer(ngram_range=(1, 7))
    print df_train.dtypes
    X_train = vectorizer.fit_transform(df_train['Phrase'].values)
    Y_train = df_train['Sentiment'].values
    X_test = vectorizer.transform(df_test['Phrase'].values)
    return X_train, Y_train, X_test
    
    
    
if __name__ == '__main__':
    
    shutil.rmtree(clfFolder, ignore_errors=True)

    df_train = pd.read_csv(trainFile)
    df_test = pd.read_csv(testFile)
    
    df_train['Phrase'] = df_train['Phrase'].astype(str)
    df_test['Phrase'] = df_test['Phrase'].astype(str)
    
    df_output = pd.DataFrame(df_test[['PhraseId']])
    
    X_train, Y_train, X_test = make_train_test(df_train, df_test)
#    clf = MultinomialNB()
#    clf.fit(X_train, Y_train)
    
    clf = fit_clf(X_train, Y_train)
    
    df_output['Sentiment'] = clf.predict(X_test)
    
    if not os.path.exists(clfFolder):
        os.makedirs(clfFolder)
    
    score_file = open(clfFolder+"Score.txt", "w")
    score_file.write("Score = {0}".format(clf.score(X_train, Y_train)))
    score_file.close()
    df_output.to_csv(clfFolder + "output.csv", index = False)
    
