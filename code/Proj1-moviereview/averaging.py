import pandas as pd
import numpy as np
import pre_process

clf_names_avg = ['vw', 'BaggingClassifier', 'MultinomialNB']

def avg_to_output():
    newColNames =[]
    df_list_test = []
    for i, item in enumerate(clf_names_avg):
        path_test = pre_process.clfFolderBase + item + '/' + pre_process.test_csv_name

        test1 = pd.read_csv(path_test)
        newColName = "Sentiment" + str(i)
        newColNames.append(newColName)
        test1.rename(columns={'Sentiment': newColName}, inplace=True)
        if i > 0:
            test1.drop(['PhraseId'], axis=1, inplace = True)

        df_list_test.append(test1)


    df_test = df_list_test[0].join(df_list_test[1:])
    df_test['Sentiment'] = df_test.ix[:, 1:len(clf_names_avg)+1].mode(axis = 1)
    df_test.ix[df_test['Sentiment'].isnull(), 'Sentiment'] = df_test.ix[df_test['Sentiment'].isnull(), 'Sentiment0']
    df_test['Sentiment'] = df_test['Sentiment'].astype(np.int)
    df_test.drop(newColNames, axis = 1, inplace=True)

    df_test.to_csv(pre_process.clfFolderTop + "avg_test.csv", index = False)

if __name__ == '__main__':
    avg_to_output()