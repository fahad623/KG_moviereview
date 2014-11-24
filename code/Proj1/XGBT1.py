import sys
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
    
    dtrain = xgb.DMatrix(X_train, label=Y_train)
    
    param = {'bst:max_depth':2, 'bst:eta':1, 'silent':1, 'objective':'binary:logistic' }
    param['nthread'] = 4
    plst = param.items()
    
    num_round = 10
    bst = xgb.train( plst, dtrain, num_round)