import numpy as np
import pandas as pd
import shutil
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.grid_search import GridSearchCV

trainFile = "../../data/train.tsv"
testFile = "../../data/test.tsv"
clfFolder = "../../classifier/SGD1/"



def cv_optimize(X_train, Y_train, clf):
    alpha_range = 10.0**np.arange(-7,-3)
    param_grid = dict(alpha=alpha_range)

    gs = GridSearchCV(clf, param_grid = param_grid, cv = 10, n_jobs = 4, verbose = 3)
    gs.fit(X_train, Y_train)
    print "gs.best_params_ = {0}, gs.best_score_ = {1}".format(gs.best_params_, gs.best_score_)
    return gs.best_estimator_, gs.best_params_, gs.best_score_

def fit_clf(X_train, Y_train):
    clf = SGDClassifier(loss = 'hinge', n_iter = 20, shuffle = True, random_state = 94) 

    clf, bp, bs = cv_optimize(X_train, Y_train, clf)    
    clf.fit(X_train, Y_train)
    return clf
    
def make_train_test(df_train, df_test):
    vectorizer = CountVectorizer()
    
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
    clf = fit_clf(X_train, Y_train)
    
    
    df_output['Sentiment'] = clf.predict(X_test)
    
    if not os.path.exists(clfFolder):
        os.makedirs(clfFolder)
    
    score_file = open(clfFolder+"Score.txt", "w")
    score_file.write("Score = {0}".format(clf.score(X_train, Y_train)))
    score_file.close()
    df_output.to_csv(clfFolder + "output.csv", index = False)
    

#gs.best_params_ = {'alpha': 1.0000000000000001e-05}, gs.best_score_ = 0.649256696143
# Kaggle score - 0.61392