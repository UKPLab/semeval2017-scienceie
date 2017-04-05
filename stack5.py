from __future__ import division

__author__      = "Ashish Airon"


from keras.models import Sequential
from keras.layers import Dense, Activation,Dropout
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import precision_recall_fscore_support,confusion_matrix

from keras.utils import np_utils

import xgboost as xgb

import pandas as pd
import numpy as np

from keras.models import Sequential
from sklearn.metrics import precision_recall_fscore_support,confusion_matrix
from keras.layers import Dense, Activation,Dropout
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier

import time


from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import train_test_split

import sys,os
if sys.version_info < (3,):
    import cPickle
else:
    import _pickle as cPickle


def check_blend(train, test):
    '''
    Unit tester to make sure dimension of blended train and test are same. 
    '''
    
    if len(train) != len(test):
        print("Length mismatch error of the blended dataset")
        
    else :
        print("All ok")
        
    

def model_withValidation(X_train_total, Y_train_total,X_test=None,Y_test=None,words_test=None,indices2labels=None,hiddenDim=250, filename_x = "none", filename_y = "none"):
    
    X_train, X_dev, Y_train, Y_dev = train_test_split(X_train_total, Y_train_total, test_size=0.10, random_state=0)

    model = Sequential()


    model.add(Dense(output_dim=hiddenDim, input_dim=X_train.shape[1]))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    
    model.add(Dense(3))
    model.add(Activation("softmax"))

    model.compile(loss='categorical_crossentropy', optimizer='adamax', metrics=['accuracy'])

    weightsPath = "./tmp/myfooo2%s.dat"%(time.time())
    checkpointer = ModelCheckpoint(filepath=weightsPath, verbose=1, save_best_only=True)

    model.fit(X_train, Y_train, verbose=2,  nb_epoch=100, batch_size=32, validation_data=(X_dev,Y_dev),callbacks=[checkpointer])

    model.load_weights(weightsPath)
    loss, acc = model.evaluate(X_test,Y_test, batch_size=32)
    
    print("loss : %0.5f Accuracy :%0.5f"%(loss,acc))

    cf = confusion_matrix(Y_test[:,1],model.predict_classes(X_test))
    print(cf)
    predictions = model.predict_classes(X_test)
    print("-->",predictions)
    
    return model,predictions
    

def stack_multiclass(X, y, XTest, shuffle =False, n_folds = 10, num_class = 3, filename_x = "none", filename_y = "none"):
    '''
    Stacking method for multi-class. 

    Parameters : 
    X               : Numpy training of size (number of sample X features)
    y               : Numpy training label pf size (number of samples,)
    XTest           : Numpy testing data of size (number of sample X features)
    yTest           : Numpy testing label pf size (number of samples,)
    shuffle         : To shuffle the training data or not. Default = False
    nfolds          : Number of folds to train the training data on. 
    num_class       : The number of classes in the dataset. Default = 3


    Returns :
    A numpy blended train and test set of size (number of samples X (number of classifiers X number of classes -1))

    '''
    

    if shuffle:
        idx = np.random.permutation(y.size)
        X = X[idx]
        y = y[idx]
    
    skf = list(StratifiedKFold(y, n_folds))
    

    clfs = [
            RandomForestClassifier(n_estimators=100, n_jobs=-1, criterion='gini'), 
            RandomForestClassifier(n_estimators=100, n_jobs=-1, criterion='entropy'),
            ExtraTreesClassifier(n_estimators=100, n_jobs=-1, criterion='gini'),
            ExtraTreesClassifier(n_estimators=100, n_jobs=-1, criterion='entropy'),
            xgb.XGBClassifier(objective='multi:softprob',silent =False, n_estimators=40)]
    
    dataset_blend_train_list = []
    for j, clf in enumerate(clfs):
        dataset_blend_train_list.append(np.zeros((X.shape[0], num_class-1 )))
    
    dataset_blend_test_list = []
    for j, clf in enumerate(clfs):
        dataset_blend_test_list.append(np.zeros((XTest.shape[0], num_class-1 )))
    

    for j, clf in enumerate(clfs):
        print(j, clf)
        dataset_blend_test_j_list = []
        
        for ii in range(n_folds):
            dataset_blend_test_j_list.append(np.zeros((XTest.shape[0], num_class-1)))
            
        for i, (train, test) in enumerate(skf):
            print("Fold Number : ", i)


            X_train_N = X[train]
            y_train_N= y[train]            
            X_test_N = X[test]
            y_test_N = y[test]
            
            clf.fit(X_train_N, y_train_N)
                    
            pred_prob_list = clf.predict_proba(X_test_N)
            
            cf = confusion_matrix(y_test_N, clf.predict(X_test_N))
            #print(cf)
                    
            dataset_blend_train_list[j][test] = pred_prob_list[:,:-1]
            
            print(dataset_blend_train_list[j].shape)
            
            dataset_blend_test_j_list[i][:, :] = clf.predict_proba(XTest)[:,:-1]
            
        temp =0
        for ff in range(n_folds):
            temp += dataset_blend_test_j_list[ff]
#	print "TEMP",temp/n_folds 
    
        dataset_blend_test_list[j] = temp/n_folds
    
	

    check_blend(dataset_blend_train_list, dataset_blend_test_list)

    # This needs to be changed depending on the number of classifiers.     
    blend_train = np.c_[dataset_blend_train_list[0], dataset_blend_train_list[1], dataset_blend_train_list[2], dataset_blend_train_list[3], dataset_blend_train_list[4]]
    
    blend_test = np.c_[dataset_blend_test_list[0], dataset_blend_test_list[1], dataset_blend_test_list[2], dataset_blend_test_list[3], dataset_blend_test_list[4]]

    return blend_train, blend_test, clfs

def read_input(filename):

    data = pd.read_table(filename, header=None, delim_whitespace=True)

    # Removing new line 
    data_new = data.dropna()

    X = data_new.iloc[:, 2:]
    Y = data_new.iloc[:, 1:2].replace({'O': 0, 'Arg-I': 1, 'Arg-B': 2})

    return X.values, np_utils.to_categorical(Y.values)
