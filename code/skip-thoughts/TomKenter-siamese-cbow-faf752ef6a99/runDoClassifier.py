#! /usr/bin/python

import sys,cPickle as pkl,numpy as np
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.utils.np_utils import to_categorical

fin = open("docClassifer.pkl","rb")
baseClassifiers = pkl.load(fin)
metaClassifier = pkl.load(fin)
classes = pkl.load(fin)

# load your test data here
TEST_Z = np.random.randn(2,300)

base_predictions = []
for bc in baseClassifiers:
        base_predictions.append(bc.predict_proba(TEST_Z)[:,:-1])
# the base classifiers classify first
test_for_MetaClassifier = np.c_[base_predictions[0], base_predictions[1], base_predictions[2], base_predictions[3], base_predictions[4]]

print classes
print metaClassifier.predict_proba(test_for_MetaClassifier,verbose=0)
