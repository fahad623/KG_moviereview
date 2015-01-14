import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.snowball import SnowballStemmer

trainFile = "../../data/train.tsv"
testFile = "../../data/test.tsv"
clfFolderTop = "../../classifier/"
clfFolderBase = "../../classifier/Base/"
clfFolderMeta = "../../classifier/Meta/"
base_csv_name = "base_output.csv"
test_csv_name = "test_output.csv"

#base_clf_names = ['AdaBoostClassifier','GradientBoostingClassifier', 'KNeighborsClassifier', 'LinearSVC', 'XGBoostClassifier','RandomForestClassifier', 'SVC']
base_clf_names = ['KNeighborsClassifier', 'XGBoostClassifier', 'LinearSVC']
meta_clf_name = 'LogisticRegression'

class PreProcessBase(object):   

    def __init__(self, pre_process = True):
        self.pre_process = pre_process
        self.df_output_train = pd.DataFrame()
        self.df_output_test = pd.DataFrame()
        self.load()
     
    def clean(self):
        pass

    def preprocess_data(self):
        vectorizer = CountVectorizer(ngram_range=(1, 8)) 
        self.X_train = vectorizer.fit_transform(self.X_train)
        self.X_test = vectorizer.transform(self.X_test)

    def load(self):
        df_train = pd.read_csv(trainFile, sep='\t')
        df_test = pd.read_csv(testFile, sep='\t')
        self.df_output_train['PhraseId'] = pd.DataFrame(df_train.ix[:,0])
        self.df_output_test['PhraseId'] = pd.DataFrame(df_test.ix[:,0])

        self.X_train = df_train['Phrase'].values
        self.Y_train = df_train['Sentiment'].values
        self.X_test = df_test['Phrase'].values

        del df_train, df_test
        if self.pre_process:
            self.preprocess_data()

    def get_train_test(self):
        return self.X_train, self.X_test, self.Y_train


class PreProcessMeta(object):   

    def __init__(self, ):
        self.load()

    def load(self):

        df_list_train = []
        df_list_test = []
        for i, item in enumerate(base_clf_names):
            path_train = clfFolderBase + item + '/' + base_csv_name
            path_test = clfFolderBase + item + '/' + test_csv_name

            train1 = pd.read_csv(path_train)
            test1 = pd.read_csv(path_test)

            if i > 0:
                train1.drop(['id'], axis=1, inplace = True)
                test1.drop(['id'], axis=1, inplace = True)

            df_list_train.append(train1)
            df_list_test.append(test1)


        df_train = df_list_train[0].join(df_list_train[1:])
        df_test = df_list_test[0].join(df_list_test[1:])

        self.X_train = df_train.ix[:, 1:len(base_clf_names)+1].values
        self.X_test = df_test.ix[:, 1:len(base_clf_names)+1].values

        print self.X_test.shape

        df_train.to_csv(clfFolderTop + "meta_train.csv", index = False)
        df_test.to_csv(clfFolderTop + "meta_test.csv", index = False)

        del df_list_train, df_list_test, df_train, df_test

    def get_train_test(self):
        return self.X_train, self.X_test
