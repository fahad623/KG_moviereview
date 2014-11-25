import sys
import os
import pandas as pd
import shutil
from sklearn.feature_extraction.text import CountVectorizer

sys.path.append('../../../xgboost/wrapper')
import xgboost as xgb

trainFile = "../../data/train.tsv"
testFile = "../../data/test.tsv"
clfFolder = "../../classifier/XGBT1/"


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
    
    xg_train = xgb.DMatrix(X_train, label=Y_train)
    xg_test = xgb.DMatrix(X_test)
    
    # setup parameters for xgboost
    param = {}
    # use softmax multi-class classification
    param['objective'] = 'multi:softmax'
    # scale weight of positive examples
    param['eta'] = 0.1
    param['max_depth'] = 10
    param['silent'] = 1
    param['nthread'] = 4
    param['num_class'] = 5
    
    watchlist = [ (xg_train,'train'), (xg_test, 'test') ]
    num_round = 1000
    
    bst = xgb.train( param, xg_train, num_round)    
    ypred = bst.predict( xg_test )
    
    df_output['Sentiment'] = ypred.astype(int)
    
    if not os.path.exists(clfFolder):
        os.makedirs(clfFolder)

    df_output.to_csv(clfFolder + "output.csv", index = False)