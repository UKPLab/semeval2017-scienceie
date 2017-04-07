from __future__ import division

__author__      = "Ashish Airon"


import sys
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
import sys


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

import numpy as np

if sys.version_info < (3,):
    import cPickle
else:
    import _pickle as cPickle
import os


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
    
    
#    model.add(Dense(output_dim=1000))
#    model.add(BatchNormalization())
#    model.add(Activation("relu"))
    
#     model.add(Dense(output_dim=20))
#     model.add(BatchNormalization())
#     model.add(Activation("relu"))
    
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
#    print dataset_blend_test_list
#    print len(dataset_blend_test_list); 
#    print dataset_blend_test_list[0].shape
#    sys.exit(1)


    # This needs to be changed depending on the number of classifiers.     
    blend_train = np.c_[dataset_blend_train_list[0], dataset_blend_train_list[1], dataset_blend_train_list[2], dataset_blend_train_list[3], dataset_blend_train_list[4]]
    
    blend_test = np.c_[dataset_blend_test_list[0], dataset_blend_test_list[1], dataset_blend_test_list[2], dataset_blend_test_list[3], dataset_blend_test_list[4]]

#    path = "./blend_richFeatures"
    
#    print("Saving the blended data...")
    
#    with  open(os.path.join(path,"blend_train_"+filename_x+"_to_"+filename_y+"_"+str(n_folds)), 'wb') as output_file:
#            cPickle.dump(blend_train, output_file)


#    with  open(os.path.join(path, "blend_test_"+filename_x+"_to_"+filename_y+"_"+str(n_folds)), 'wb') as output_file:
#            cPickle.dump(blend_test, output_file)

    return blend_train, blend_test, clfs

def read_input(filename):

    data = pd.read_table(filename, header=None, delim_whitespace=True)

    # Removing new line 
    data_new = data.dropna()

    X = data_new.iloc[:, 2:]
    Y = data_new.iloc[:, 1:2].replace({'O': 0, 'Arg-I': 1, 'Arg-B': 2})

    return X.values, np_utils.to_categorical(Y.values)

if __name__ == "__main__":

    np.random.seed(3)
    start_time = time.time()


#    train_path = "./data/Stab201X/train_10_1.marmot.GN300d_5_embed"
#    test_path = "./data/Stab201X/test_10_1.marmot.GN300d_5_embed"

#    indices=range(-5,6,1)

#    X_train,Y_train,words_train,indices2labels_train = md.readData(sys.argv[1],indices=indices)
#    X_test, Y_test,words_test,indices2labels_test = md.readData(sys.argv[2],indices=indices)

    X_train, Y_train = read_input(sys.argv[1])
    X_test, Y_test = read_input(sys.argv[2])


#    X_train, Y_train ,words_train, indices2labels_train = md.readData(train_path,indices=indices)
#    X_test, Y_test, words_test, indices2labels_test = md.readData(test_path,indices=indices)
    
    Y_train_single_value = np_utils.categorical_probas_to_classes(Y_train)
    Y_test_single_value = np_utils.categorical_probas_to_classes(Y_test)
    
    
    blend_train , blend_test = stack_multiclass(X_train, Y_train_single_value, X_test, Y_test_single_value, filename_x = sys.argv[3], filename_y = sys.argv[4])

    
#     path = "./blend"
    
#     #To load the blended data.
#     with  open(os.path.join(path,'blend_train'), 'rb') as input_file: 
#         blend_train =cPickle.load(input_file)
    
    
#     #To load the blended data.
#     with  open(os.path.join(path,'blend_test'), 'rb') as input_file: 
#         blend_est =cPickle.load(input_file)
    
    print("Blended Data read ...")
    #for hiddenDim in [100,200,250,300,400,500,800,1000,2000]:
    for hiddenDim in [5, 10, 15, 20, 30, 50, 100]:
        print("####",hiddenDim)
#         mymodel = model_withValidation(X_train,Y_train,X_test,Y_test,words_test,indices2labels_test,hiddenDim=hiddenDim)
        mymodel = model_withValidation(blend_train, Y_train,
                                       blend_test, Y_test,
                                       hiddenDim=hiddenDim,
				       filename_x = sys.argv[3],
				       filename_y = sys.argv[4])



    print("\nTotal time :  %s seconds " % (time.time() - start_time))
