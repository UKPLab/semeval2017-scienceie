'''
Copyright 2016 Tom Kenter

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and limitations
under the License.
'''

import numpy as np
import codecs
import cPickle
import pickle
from gensim import matutils
import lasagne

import vocabUtils

class wordEmbeddings:
  '''
  This class deals with word embeddings.
  It's for storing, getting, displaying, finding most similar embeddings, etc.
  It somewhat resembles the gensim api.
  '''

  def __init__(self, sPickleFile=None, dDict=None, sVocabFile=None,
               bUnknownOmit=False, sAggregation='average', bDebug=False):
    self.bDebug = bDebug
    self.bUnknownOmit = bUnknownOmit

    if not self.bUnknownOmit:
      self.dRandomEmbeddings = {}

    if sAggregation not in ['average', 'sum']:
      print >>sys.stderr, \
          "[ERROR] Don't know aggregation method '%s'. Typo?" % sAggregation
      exit(1)
    self.sAggregation = sAggregation

    if sPickleFile is not None:
      fhFile = open(sPickleFile, mode='rb')
      dDict = cPickle.load(fhFile)
      fhFile.close()

    self.oArgs = dDict["oArgs"]
    self.npaWordEmbeddings = dDict["npaWordEmbeddings"]

    self.npaWordEmbeddings_units = dDict["npaWordEmbeddings"] / \
        np.sqrt(np.square(dDict["npaWordEmbeddings"]).sum(axis=1)).reshape(-1,
                                                                            1)
    if sVocabFile is None:
      self.oVocab = dDict["oVocab"]
    else:
      self.oVocab = vocabUtils.vocabulary(sVocabFile,
                                          iMaxNrOfWords=\
                                          self.oArgs.iMaxNrOfVocabWords)

  def __getitem__(self, sWord):
    iIndex = self.oVocab[sWord]
    return None if iIndex is None else self.npaWordEmbeddings[iIndex]

  def getUnitVector(self, sWord):
    try:
      iIndex = self.oVocab[sWord]
      return None if iIndex is None else self.npaWordEmbeddings_units[iIndex] 
    except KeyError:
      return None
    
  def getRandomEmbedding(self, sWord):
    try:
      return self.dRandomEmbeddings[sWord]
    except KeyError:
      self.dRandomEmbeddings[sWord] = \
          lasagne.init.Normal().sample((1,
                                        self.npaWordEmbeddings.shape[1])
                                       )[0]
      return self.dRandomEmbeddings[sWord]

  def getAggregate(self, aTokens):
    if self.bDebug:
      import pdb
      pdb.set_trace()

    iNrOfWords = 0
    npaV = np.zeros(self.npaWordEmbeddings.shape[1])

    for sToken in aTokens:
      npaWordEmbedding = self[sToken]
      if (npaWordEmbedding is None) and (not self.bUnknownOmit):
        npaWordEmbedding = self.getRandomEmbedding(sToken)

      if npaWordEmbedding is not None:
        npaV += npaWordEmbedding
        iNrOfWords += 1
        
    if iNrOfWords > 0:
      if self.sAggregation == 'average':
        npaV /= iNrOfWords

    # NOTE that we return all zeros if there are no known words...
    return npaV

  def sentence_similarity(self, aTokens1, aTokens2):
    npaAggrate1 = self.getAggregate(aTokens1)
    npaAggrate2 = self.getAggregate(aTokens2)

    # If they are identical (e.g. if they are both all zeros)
    if (npaAggrate1 == npaAggrate2).sum() == npaAggrate2.shape[0]:
      return 1.0

    # If one only has unknown words, we assume that they are dissimilar (but
    # not very. As in, not -1.0)
    if ((npaAggrate1 == 0.0).sum() == npaAggrate1.shape[0]) or \
          ((npaAggrate2 == 0.0).sum() == npaAggrate2.shape[0]):
      return 0.0

    if (npaAggrate1 is None) or (npaAggrate2 is None) :
      return None

    npaUnit1 = npaAggrate1 / np.sqrt(np.square(npaAggrate1).sum())
    npaUnit2 = npaAggrate2 / np.sqrt(np.square(npaAggrate2).sum())

    return np.dot(npaUnit1, npaUnit2)

  def most_similar(self, sWord, iTopN=10, fMinDist=-1.0):
    npaWord_unit = self.getUnitVector(sWord)

    if npaWord_unit is None:
      return None

    npaCosineSimilarities = np.dot(self.npaWordEmbeddings_units, npaWord_unit)

    npaBestIndices = \
        matutils.argsort(npaCosineSimilarities, topn=iTopN +1, reverse=True)

    # npaBestIndices[1:] - Ignore the first one (which is sWord itself)
    return [(self.oVocab.index2word(x), npaCosineSimilarities[x]) for x in npaBestIndices[1:] if npaCosineSimilarities[x] > fMinDist]

  def sortByNorm(self, iMin, iMax):
    if not hasattr(self, 'npaIndicesByNorm'):
      self.npaNorms = np.sqrt(np.square(self.npaWordEmbeddings).sum(axis=1))
      self.npaIndicesByNorm = matutils.argsort(self.npaNorms)

    return [(self.oVocab.index2word(x), self.npaNorms[x]) for x in self.npaIndicesByNorm[iMin:iMax]]

  def most_similar_simple(self, sWord, iTopN=10):
    npaWordEmbedding = self[sWord]

    if npaWordEmbedding is None:
      return None

    npaSimilarities = np.dot(self.npaWordEmbeddings, npaWordEmbedding)

    npaBestIndices = \
        matutils.argsort(npaSimilarities, topn=iTopN +1, reverse=True)

    # npaBestIndices[1:] - Ignore the first one (which is sWord itself)
    return [(self.oVocab.index2word(x), npaSimilarities[x]) for x in npaBestIndices[1:]]

  def most_similars(self, fMinDist=-1.0, aIndexRange=None):
    if aIndexRange is None:
      aIndexRange = range(1, self.oVocab.iNrOfWords)

    for iIndex in aIndexRange:
      print "[%d] %s: %s" % \
          (iIndex, self.oVocab.index2word(iIndex),
           ', '.join(["%s (%f)" % x for x in self.most_similar(self.oVocab.index2word(iIndex), fMinDist=fMinDist)] )
           )
