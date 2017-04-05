#!/usr/bin/python
# -*- coding: UTF-8 -*-

from reader import ScienceIEBratReader
from extras import VSM
from extras import utf8ify, read_and_map
from representation import VeryStupidCBOWMapper, ConcatMapper
import stack5 as st_learn

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.utils.np_utils import to_categorical
from keras.utils import np_utils
from sklearn.metrics import confusion_matrix
import sys,os,time,random,numpy as np

# example argline:
# python stackedLearner.py ../scienceie2017_train/train2 ../scienceie2017_dev/dev ../resources/vsm/glove.6B/glove.6B.100d.txt
if __name__=="__main__":

#    np.random.seed(4)  # for reproducibility

    train_src = sys.argv[1]
    dev_src = sys.argv[2]
    vsm_path = sys.argv[3]
    #test_src = sys.argv[4]

    print("Loading VSM")
    vsm = VSM(vsm_path)

    cs = 4
    try:
	cs = int(sys.argv[4])
    except IndexError:
	cs = 4

    try:
      domain_file = sys.argv[5]
      if domain_file=="None": domain_file = None
    except IndexError:
      domain_file = None

    if len(sys.argv)>5 and sys.argv[5]=="document":
      SB = False
    else:
      SB = True

    mapper = ConcatMapper(vsm,cs,sentence_boundaries=SB)

    print("Reading training data")
    X_train, y_train, y_values, _ = read_and_map(train_src, mapper, domain_file = domain_file)

    print(X_train.shape)
    print(y_train.shape)
    print(y_values)
    nclasses = len(y_values)

    print("Reading test data")
    X_dev, y_dev_gold, _, entities = read_and_map(dev_src, mapper, y_values, domain_file = domain_file)

    blend_train, blend_test, baseClassifiers = st_learn.stack_multiclass(X_train, y_train, X_dev, filename_x = "A"+str(random.random()), filename_y = "B"+str(time.time()))

    TEST_Z = X_dev
    base_predictions = []
    for bc in baseClassifiers:
        base_predictions.append(bc.predict_proba(TEST_Z)[:,:-1])
     # the base classifiers classify first
    test_for_MetaClassifier = np.c_[base_predictions[0], base_predictions[1], base_predictions[2], base_predictions[3], base_predictions[4]]



    Y_train_1hot = np.zeros((len(y_train),nclasses))
    Y_train_1hot[np.arange(len(y_train)),y_train] = 1
    Y_dev_1hot = np.zeros((len(y_dev_gold),nclasses))
    Y_dev_1hot[np.arange(len(y_dev_gold)),y_dev_gold] = 1

    hiddenDim = 20
    mymodel,y_dev_auto = st_learn.model_withValidation(blend_train, Y_train_1hot,
                                       blend_test, Y_dev_1hot,
                                       hiddenDim=hiddenDim,
                                       filename_x = "C",
                                       filename_y = "D")

    print "==PREDICTING=="
    for i in xrange(len(y_dev_auto)):
	print y_values[y_dev_auto[i]]
