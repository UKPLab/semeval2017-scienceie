#!/usr/bin/python
# -*- coding: UTF-8 -*-

from extras import VSM, read_and_map
from representation import VeryStupidCBOWMapper, CharMapper

import sys, numpy as np,os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from keras.layers import Dense, Dropout, Activation, Embedding
from keras.models import Sequential
from keras.utils.np_utils import to_categorical
from keras.layers import Convolution1D, GlobalMaxPooling1D, Lambda, Merge
from keras.preprocessing import sequence
from keras import backend as K

maxlen=50
maxlen=100
maxlen=150
maxlen=50+2*30
try:
	L = int(sys.argv[5])
	M = int(sys.argv[6])
	R = int(sys.argv[7])
except IndexError:
	L = 30
	M = 50
	R = 30
maxlen=L+M+R

# this is a simple cnn
# if you would want to use it below, you would have to do
# X_train = X_train.reshape(len(X_train),input_shape[0],input_shape[1])
def build_cnn(input_shape, output_dim,nb_filter):
    clf = Sequential()
    clf.add(Convolution1D(nb_filter=nb_filter,
                          filter_length=4,border_mode="valid",activation="relu",subsample_length=1,input_shape=input_shape))
    clf.add(GlobalMaxPooling1D())
    clf.add(Dense(100))
    clf.add(Dropout(0.2))
    clf.add(Activation("tanh"))
    clf.add(Dense(output_dim=output_dim, activation='softmax'))

    clf.compile(optimizer='adagrad',
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])
    return clf

# just one filter
def build_cnn_char(input_dim, output_dim,nb_filter):
    clf = Sequential()
    clf.add(Embedding(input_dim,
                      32, # character embedding size
                      input_length=maxlen,
                      dropout=0.2))
    clf.add(Convolution1D(nb_filter=nb_filter,
                          filter_length=3,border_mode="valid",activation="relu",subsample_length=1))
    clf.add(GlobalMaxPooling1D())
    clf.add(Dense(100))
    clf.add(Dropout(0.2))
    clf.add(Activation("tanh"))
    clf.add(Dense(output_dim=output_dim, activation='softmax'))

    clf.compile(optimizer='adagrad',
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])
    return clf

# just one filter
def build_cnn_char_threeModels(input_dim, output_dim,nb_filter,filter_size=3):
    left = Sequential()
    left.add(Embedding(input_dim,
             32, # character embedding size
             input_length=L,
             dropout=0.2))
    left.add(Convolution1D(nb_filter=nb_filter,
                          filter_length=filter_size,border_mode="valid",activation="relu",subsample_length=1))
    left.add(GlobalMaxPooling1D())
    left.add(Dense(100))
    left.add(Dropout(0.2))
    left.add(Activation("tanh"))

    center = Sequential()
    center.add(Embedding(input_dim,
             32, # character embedding size
             input_length=M,
             dropout=0.2))
    center.add(Convolution1D(nb_filter=nb_filter,
                          filter_length=filter_size,border_mode="valid",activation="relu",subsample_length=1))
    center.add(GlobalMaxPooling1D())
    center.add(Dense(100))
    center.add(Dropout(0.2))
    center.add(Activation("tanh"))

    right = Sequential()
    right.add(Embedding(input_dim,
             32, # character embedding size
             input_length=R,
             dropout=0.2))
    right.add(Convolution1D(nb_filter=nb_filter,
                          filter_length=filter_size,border_mode="valid",activation="relu",subsample_length=1))
    right.add(GlobalMaxPooling1D())
    right.add(Dense(100))
    right.add(Dropout(0.2))
    right.add(Activation("tanh"))
    
    clf = Sequential()
    clf.add(Merge([left,center,right],mode="concat"))
    clf.add(Dense(output_dim=output_dim, activation='softmax'))
    
    clf.compile(optimizer='adagrad',
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])
    return clf



def max_1d(X):
    return K.max(X,axis=1)

# multiple filters
def build_cnn_char_complex(input_dim, output_dim,nb_filter):
    randomEmbeddingLayer = Embedding(input_dim,32, input_length=maxlen,dropout=0.1)
    poolingLayer = Lambda(max_1d, output_shape=(nb_filter,))
    conv_filters = []
    for n_gram in range(2,4):
        ngramModel = Sequential()
        ngramModel.add(randomEmbeddingLayer)
        ngramModel.add(Convolution1D(nb_filter=nb_filter,
                                     filter_length=n_gram,
                                     border_mode="valid",
                                     activation="relu",
                                     subsample_length=1))
        ngramModel.add(poolingLayer)
        conv_filters.append(ngramModel)
    
    clf = Sequential()
    clf.add(Merge(conv_filters,mode="concat"))
    clf.add(Activation("relu"))
    clf.add(Dense(100))
    clf.add(Dropout(0.1))
    clf.add(Activation("tanh"))
    clf.add(Dense(output_dim=output_dim, activation='softmax'))

    clf.compile(optimizer='adagrad',
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])
    return clf

def acc(correct, total):
    return 1.0*correct/total

# example argline:
# python convNet.py ../scienceie2017_train/train2 ../scienceie2017_dev/dev ../resources/vsm/glove.6B/glove.6B.100d.txt
if __name__=="__main__":

    train_src = sys.argv[1]
    dev_src = sys.argv[2]
#    vsm_path = sys.argv[3]
    vsm_path = None

    print("Loading VSM")
    vsm = VSM(vsm_path)
    
    try:
      csize = 2
    except IndexError:
      csize = int(sys.argv[4])
    try:
      n_filter = int(sys.argv[8])
    except IndexError:
      n_filter = 250 
    try:
      filter_size = int(sys.argv[9])
    except IndexError:
      filter_size = 3
    if len(sys.argv)>10 and sys.argv[10]=="document":
      SB = False
    else:
      SB = True

    mapper = CharMapper(vsm,csize,L=L,M=M,R=R,sentence_boundaries=SB)
    
    print("Reading training data")
    X_train, y_train, y_values, _ = read_and_map(train_src, mapper)
    X_dev, y_dev_gold, _, estrings = read_and_map(dev_src, mapper, y_values)
    vocabSize = mapper.curVal
    
    print(X_train.shape)
    print(y_train.shape)
    #sys.exit(1)
    
    print("Trainig a model")

    timesteps = 2*csize + 1 # left, right, center
    context_dim = 100
    input_shape = (timesteps,context_dim)
    clf = build_cnn_char(vocabSize+1, len(y_values)+1,n_filter)
    clf = build_cnn_char_threeModels(vocabSize+1, len(y_values)+1,n_filter)

    X_left = X_train[:,:L]
    X_center = X_train[:,L:L+M]
    X_right = X_train[:,L+M:L+M+R]
    print L,M,R,X_train.shape,X_left.shape,X_center.shape,X_right.shape,y_train,y_values
    clf.fit([X_left,X_center,X_right], to_categorical(y_train, len(y_values)+1), verbose=1, nb_epoch=15)


    print("Reading test data")
    print("Testing")

    X_dev_left = X_dev[:,:L]
    X_dev_center = X_dev[:,L:L+M]
    X_dev_right = X_dev[:,L+M:L+M+R]
    print(X_dev.shape,X_dev_left.shape,X_dev_center.shape,X_dev_right.shape)
    
    y_dev_auto = clf.predict_classes([X_dev_left,X_dev_center,X_dev_right])  # for LogisticRegression just do predict()

    print "==PREDICTING=="
    for i in xrange(len(y_dev_auto)):
        print y_values[y_dev_auto[i]]
