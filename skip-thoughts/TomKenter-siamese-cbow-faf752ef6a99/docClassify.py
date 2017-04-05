#! /usr/bin/python

import sys,cPickle as pkl,numpy as np
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.utils.np_utils import to_categorical
import stack5 as st_learn

# SAMPLE USAGE
# ./docClassify.py vectors_docs_siamese.pkl

def build_nn(input_dim, output_dim,hidden_dim=100):
    nn_classifier = Sequential()
    nn_classifier.add(Dense(hidden_dim, input_dim=input_dim, activation='tanh'))
    nn_classifier.add(Dropout(0.5))
    #nn_classifier.add(Dense(100, activation='tanh'))
    nn_classifier.add(Dense(output_dim=output_dim, activation='softmax'))

    nn_classifier.compile(optimizer='adagrad',
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])
    return nn_classifier



f=open(sys.argv[1],"rb")
classes = pkl.load(f)
vectors = pkl.load(f)

vecs_and_indices = [(i,v) for (i,v) in enumerate(vectors) if type(v)==type(vectors[0])]
vecs = np.vstack([x[-1] for x in vecs_and_indices])
print vecs.shape
relevant_classes = []
my_classes = {}
class_index = 0
for (i,v) in vecs_and_indices:
  if classes[i] not in my_classes:
	my_classes[classes[i]] = class_index
	class_index += 1
  c = my_classes[classes[i]]
  relevant_classes.append(c)
relevant_classes = np.array(relevant_classes)

nclasses = class_index
perm=np.random.permutation(len(relevant_classes))
#np.random.rand(perm,1)
training_ratio = 0.99
train_n = int( len(relevant_classes)*training_ratio )

train_x,test_x = vecs[perm][:train_n,:],vecs[perm][train_n:,:]
train_y,test_y = relevant_classes[perm][:train_n],relevant_classes[perm][train_n:]


blend_train, blend_test, baseClassifiers = st_learn.stack_multiclass(train_x, train_y, test_x, filename_x = "A", filename_y = "B", n_folds=10)

# TEST_Z is your test file
TEST_Z = test_x
base_predictions = []
for bc in baseClassifiers:
	base_predictions.append(bc.predict_proba(TEST_Z)[:,:-1])
# the base classifiers classify first
test_for_MetaClassifier = np.c_[base_predictions[0], base_predictions[1], base_predictions[2], base_predictions[3], base_predictions[4]]


Y_train_1hot = to_categorical(train_y,nclasses)
Y_test_1hot = to_categorical(test_y,nclasses)

hiddenDim = 20
mymodel,test_predict = st_learn.model_withValidation(blend_train, Y_train_1hot,
                                       blend_test, Y_test_1hot,
                                       hiddenDim=hiddenDim,
                                       filename_x = "C",
                                       filename_y = "D")

#print "\n=======CHECK"
# then the Meta Classifier combines the predictions of the base classifiers
print mymodel.predict_proba(test_for_MetaClassifier,verbose=0) #[:20,:]
#print test_predict[:20]; sys.exit(1)
f=open("docClassifer.pkl","wb")
pkl.dump(baseClassifiers,f)
pkl.dump(mymodel,f)
pkl.dump(my_classes,f)
f.close()

if False:

  print len(vectors),len(vecs),len(relevant_classes),train_x.shape,train_y.shape,test_x.shape
  print to_categorical([1,2,3,0],4)


  classifier = build_nn(train_x.shape[1], nclasses,hidden_dim=500)
  classifier.fit(train_x, to_categorical(train_y, nclasses), verbose=True, nb_epoch=5)
  test_predict = classifier.predict_classes(test_x)
  print test_predict
class0 = np.zeros((len(test_y),))
class1 = np.zeros((len(test_y),))+1
class2 = np.zeros((len(test_y),))+2


print "\nPREDICTION",sum(test_predict==test_y)*1.0/len(test_y)
for c in [class0,class1,class2]:
	print "Majority",sum(c==test_y)*1.0/len(test_y)
