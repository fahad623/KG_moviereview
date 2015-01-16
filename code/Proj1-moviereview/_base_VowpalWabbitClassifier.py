import sys
sys.path.append('../../../../py_ml_utils/')
from VowpalWabbit import VowpalWabbitClassifier

import pre_process

predict_method = ''

def make_best_classifier():
    return VowpalWabbitClassifier(), predict_method

def train_base_clf():    
    clf = make_best_classifier()[0]

    clf.fit(pre_process.trainFile_vw)
    print clf.predict(pre_process.testFile_vw)


if __name__ == '__main__':
    train_base_clf()