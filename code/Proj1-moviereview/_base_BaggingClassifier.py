from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import BaggingClassifier
import numpy as np
import cross_val
import pre_process
import output_csv

predict_method = ''

def make_best_classifier():
    return BaggingClassifier(MultinomialNB(), n_estimators = 190, max_samples=0.9, max_features=1.0), predict_method

def train_base_clf(pp):
    clf = make_best_classifier()[0]
    n_estimators_range = np.arange(50, 300, 10)
    max_samples_range = np.arange(0.7, 0.9, 0.02)
    param_grid = dict(n_estimators= n_estimators_range)
    clf, bp, bs = cross_val.fit_clf(clf, pp.X_train, pp.Y_train, param_grid)
    output_csv.write_test_csv(clf.__class__.__name__, pp.df_output_test, clf.predict(pp.X_test))
    output_csv.write_gs_params_base(clf.__class__.__name__, bp, bs, 
                                    clf.score(pp_base.X_train, pp_base.Y_train))
    return clf, predict_method


if __name__ == '__main__':
    pp_base = pre_process.PreProcessBase()
    clf = train_base_clf(pp_base)[0]
