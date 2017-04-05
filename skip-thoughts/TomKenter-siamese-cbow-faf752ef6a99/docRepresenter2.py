#! /usr/bin/python

import sys,codecs,numpy as np,cPickle as pkl
from nltk.tokenize import sent_tokenize,word_tokenize
import wordEmbeddings as we


# SAMPLE USAGE
# ./docRepresenter2.py cosine_sharedWeights_adadelta_lr_1_noGradClip_epochs_2_batch_100_neg_2_voc_65536x300_noReg_lc_noPreInit_vocab_65535.end_of_epoch_2.pickle ../../DocClassification/myarticles/*.xml

model = we.wordEmbeddings(sys.argv[1])

def readDoc(fn):
    return codecs.open(fn,"r","utf-8").readlines()

docs=[]
dim=300

vs=[]
cs=[]
vectors=[]
classes=[]
for doc in sys.argv[2:]:
    lines = readDoc(doc)
    docEmbedding = []
    for sentence in sent_tokenize(" ".join(lines)):
	for token in word_tokenize(sentence):
	  try:
	    vec = model[token]
	  except KeyError:
	    vec = np.zeros((dim,))
	  if vec is None: 
		vec = np.zeros((dim,))
		continue
	  docEmbedding.append(vec)
    classes.append(filter(lambda x: x!="",doc.split("/"))[-1][:2])
    vectors.append(np.mean(docEmbedding,axis=0))


fOut = open('vectors_docs_siamese_new.pkl', 'wb')
pkl.dump(classes, fOut, -1)
pkl.dump(vectors, fOut, -1)
fOut.close()

print "DONE"




