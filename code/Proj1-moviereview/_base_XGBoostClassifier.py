import sys
sys.path.append('../../../../xgboost/wrapper/')
import xgboost as xgb
import numpy as np
from sklearn.metrics import log_loss
from sklearn.base import BaseEstimator, ClassifierMixin
import cross_val
import pre_process
import output_csv

predict_method = ''

class XGBoostClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, objective = 'multi:softmax', eta = 0.3, max_depth = 6, 
                 subsample = 0.7, num_round = 100):
        self.objective = objective
        self.eta = eta
        self.max_depth = max_depth
        self.subsample = subsample
        self.num_round = num_round    
    
    def _init_params(self):
        self.param = {}
        self.param['objective'] = self.objective
        self.param['eta'] = self.eta
        self.param['max_depth'] = self.max_depth
        self.param['subsample'] = self.subsample
        self.param['silent'] = 1
        self.param['nthread'] = 4
        self.param['seed'] = 31
        self.param['num_class'] = 5

    def fit(self, X_train, Y_train):
        self._init_params()
        xg_train = xgb.DMatrix(X_train, label=Y_train)    
        self.bst = xgb.train(self.param, xg_train, self.num_round) 

    def predict_proba(self, X_test):
        pass

    def predict(self, X_test):
        xg_test = xgb.DMatrix(X_test)
        return self.bst.predict( xg_test )

def make_best_classifier():
    return XGBoostClassifier(num_round = 1000, subsample = 0.7, eta = 0.2, max_depth = 100), predict_method

def train_base_clf(pp):    
    clf = make_best_classifier()[0]
    eta_range = np.arange(0.1, 0.4, 0.1)
    max_depth_range = np.arange(10, 150, 10) 
    num_round_range = np.arange(30, 100, 10)
    subsample_range = np.arange(0.50, 1.0, 0.1)

    param_grid = dict(eta = eta_range, max_depth = max_depth_range, num_round = num_round_range, subsample = subsample_range)
    #clf, bp, bs = cross_val.fit_clf(clf, pp.X_train, pp.Y_train, param_grid)
    output_csv.write_test_csv(clf.__class__.__name__, pp.df_output_test, clf.predict(pp.X_test))
    #output_csv.write_gs_params_base(clf.__class__.__name__, bp, bs, 
    #                                clf.score(pp_base.X_train, pp_base.Y_train))
    return clf, predict_method


if __name__ == '__main__':
    pp_base = pre_process.PreProcessBase()
    train_base_clf(pp_base)
    

