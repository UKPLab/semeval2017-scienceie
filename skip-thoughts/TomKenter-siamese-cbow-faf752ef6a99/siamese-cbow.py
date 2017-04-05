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

import theano
import theano.tensor as T
import lasagne
from lasagne.regularization import regularize_layer_params, l2, l1
import numpy as np
import time
import os
import sys
import gensim
import signal

import siamese_cbowUtils as scbowUtils
import vocabUtils
sys.path.append('./inexutils')
import inexUtils
sys.path.append('./ppdbutils')
import ppdbUtils
sys.path.append('./torontobookcorpusutils')
import torontoBookCorpusUtils

# You can get killed form the outside on some clusters.
# This is just to signal that.
def signal_term_handler(signal, frame):
  print >>sys.stderr, \
    "[ERROR siamese-cbow] This process got a SIGTERM. Quitting."
  exit(1)

def init(oArgs):
  sInputMode = inputMode(oArgs.DATA.lower())

  if (sInputMode == 'INEX') and (oArgs.sUpdate == 'sgd'):
    print >>sys.stderr, "[ERROR] I am sorry, 'sgd' is not supported for INEX data input (but 'adadelta' might come very close)"
    exit(1)

  oVocab = None
  if oArgs.sVocabFile is not None:
    oVocab = vocabUtils.vocabulary(oArgs.sVocabFile,
                                   sInputMode="freqword",
                                   iMaxNrOfWords=oArgs.iMaxNrOfVocabWords)
  oW2v = None
  if oArgs.sWord2VecFile is not None:
    if oArgs.bVerbose:
      print "Loading word2vec file"

    oW2v = gensim.models.Word2Vec.load_word2vec_format(oArgs.sWord2VecFile,
                                                       binary=True) 
    # In this case we make a vocabulary of the w2v model. This is initialized
    # later on in build_scbow.
    oVocab = vocabUtils.vocabulary(None)

  # NOTE, we take the number of words in the w2v model + 1, for the 0 embedding
  iNrOfEmbeddings = \
      oVocab.iNrOfWords if oW2v is None else oW2v.syn0.shape[0] + 1
  iEmbeddingSize = oArgs.iEmbeddingSize if oW2v is None else oW2v.syn0.shape[1]
  if iEmbeddingSize is None:
    print >>sys.stderr, "[ERROR] Don't know what the embedding size is"
    exit(1)

  if oArgs.bVerbose:                    # -1 because of 0-embedding
    print "Number of embeddings: %d" % (iNrOfEmbeddings-1)
    print "Embedding size: %d" % iEmbeddingSize

  sOutputFile = \
      scbowUtils.makeOutputFileName(oArgs, iNrOfEmbeddings, iEmbeddingSize)
  print "Will store result to %s" % sOutputFile

  iPos, npaTargets = getTargets(oArgs, sInputMode)

  return oVocab, oW2v, iNrOfEmbeddings, iEmbeddingSize, sInputMode, \
      sOutputFile, iPos, npaTargets

def getTargets(oArgs, sInputMode):
  # The number of positive examples is 1 in the PPDB case (where we have phrase
  # pairs) and 2 in the otherwise (in which case we are comparing to the
  # the previous and next sentence).
  iPos = 1 if sInputMode == 'PPDB' else 2

  # The targets are always the same. We always want the positive elements of
  # the softmax (so index 0 (and possibly 1)) to be maximum (so 1 (or .5 if we
  # have two)).
  npaTargets = \
      np.zeros((oArgs.iBatchSize, oArgs.iNeg + iPos), dtype=np.float32)
  npaTargets[:,[0,(iPos-1)]] = 1.0 / iPos

  return iPos, npaTargets

def inputMode(sData):
  if 'ppdb' in sData:
    return 'PPDB'
  elif 'inex' in sData:
    return 'INEX'
  elif 'toronto' in sData:
    return 'TORONTO'
  else:
    print >>sys.stderr, "[ERROR]: Don't know what input mode to run in (INEX, PPDB)or TORONTO from '%s'" % sData
    exit(1)

def getSentenceIterators(oArgs, sInputMode):
  oSentenceIterator, funcRandomIterator, fTotalNrOfBatches = None, None, None

  if sInputMode == 'INEX':
    oSentenceIterator = \
        InexUtils.InexIterator(oArgs.DATA, 
                               bDontLowercase=oArgs.bDontLowercase,
                               sName='inexTripleIterator')
    # In the INEX case it just takes way too long to calculate the total amount
    # of sentences there are. So, for now, we simply don't do it
    # Because of this, you can use sgd in this case.

    oRandomIterator = \
        InexUtils.InexIterator(oArgs.DATA,
                               bDontLowercase=oArgs.bDontLowercase,
                               bRandom=True, sName='inexRandomIterator')
    funcRandomIterator = oRandomIterator.__iter__()
  elif sInputMode == "TORONTO":
    oSentenceIterator = \
        torontoBookCorpusUtils.torontoBookCorpusIterator(
            sCorpusDir=oArgs.DATA, sSentencePositionsDir=oArgs.DATA,
            sName='torontoSentenceIterator', bVerbose=oArgs.bVerbose)

    fTotalNrOfBatches = \
        ( float(oSentenceIterator.iTotalNrOfSentences) / oArgs.iBatchSize) * \
        oArgs.iEpochs

    # In this case, as we have a large file of sentence positions, we want to 
    # use the same object again...
    funcRandomIterator = oSentenceIterator.yieldRandomSentence()
  else: ## PPDB
    oSentenceIterator = ppdbUtils.ppdb(oArgs.DATA,
                                       sName='ppdbSentenceIterator')

    oRandomIterator = ppdbUtils.ppdb(oArgs.DATA, bRandom=True,
                                     sName='ppdbRandomIterator')
    funcRandomIterator = oRandomIterator.__iter__()

    fTotalNrOfBatches = \
        (float(oRandomIterator.iTotalNrOfSentences) / oArgs.iBatchSize) * \
        oArgs.iEpochs

  return oSentenceIterator, funcRandomIterator, fTotalNrOfBatches

def storeWordEmbeddings(oNetwork, oVocab, oArgs, sOutputFile):
  '''
  Store the word embeddings to disc as a pickle file.
  '''

  if not oArgs.bShareWeights:
    sOutputFile = sOutputFile.replace(".pickle", ".in.pickle") 

  if oArgs.sLastLayer == 'cosine':
    # We have one more layer here (the softmax) at the end
    scbowUtils.storeWordEmbeddings(sOutputFile,
                                    oNetwork.input_layer.input_layers[0].input_layer.input_layers[1].W.get_value(),                                    
                                    oVocab,
                                    oArgs)
    if not oArgs.bShareWeights:
      scbowUtils.storeWordEmbeddings(sOutputFile.replace(".in.pickle",
                                                          ".out.pickle"),
                                      oNetwork.input_layer.input_layers[1].input_layer.input_layer.input_layers[1].W.get_value(),
                                      oVocab,
                                      oArgs)
  else: # Negative sampling
    scbowUtils.storeWordEmbeddings(sOutputFile,
                                    oNetwork.input_layers[0].input_layer.input_layers[1].W.get_value(),
                                    oVocab,
                                    oArgs)
    if not oArgs.bShareWeights:
      scbowUtils.storeWordEmbeddings(sOutputFile.replace(".in.pickle",
                                                          ".out.pickle"),
                                      oNetwork.input_layers[1].input_layer.input_layer.input_layer.input_layers[1].W.get_value(),
                                      oVocab,
                                      oArgs)

if __name__ == "__main__":
  # First things first
  signal.signal(signal.SIGTERM, signal_term_handler)

  oArgs = scbowUtils.parseArguments()

  fStartTime = time.time() if oArgs.bVerbose else None

  oVocab, oW2v, iNrOfEmbeddings, iEmbeddingSize, sInputMode, sOutputFile, \
      iPos, npaTargets = init(oArgs)

  if oArgs.bVerbose:
    print "Building network"
  oNetwork, forward_pass_fn, thsLearningRate, train_fn = \
      scbowUtils.build_scbow(oArgs, iPos=iPos, oW2v=oW2v, oVocab=oVocab,
                             tWeightShape=(iNrOfEmbeddings, iEmbeddingSize))

  oSentenceIterator, funcRandomIterator, fTotalNrOfBatches = \
      getSentenceIterators(oArgs, sInputMode)

  # Everything is set up. Now we can start doing something.
  if oArgs.bDryRun: #  Or can we...
    exit(1)

  if oArgs.bVerbose:
    print "Start training"

  # Print a number every so-many batches (just so the progress is noticable)
  iBatchBunch = 100

  # Pre-allocate memory
  npaBatch_1 = np.zeros((oArgs.iBatchSize, oArgs.iMaxNrOfTokens),
                        dtype=np.uint16)
  npaBatch_2 = \
      np.zeros((oArgs.iBatchSize, oArgs.iNeg + iPos, oArgs.iMaxNrOfTokens),
               dtype=np.uint16)

  bStoredEmbeddingsAtLastEpoch = False
  iNrOfBatchesSoFar = 0 # This is a counter for the grand total (not per epoch)
  # This is the main loop. We iterate over epochs:
  for iEpoch in range(oArgs.iEpochs):
    # In each epoch, we do a full pass over the training data:
    fEpochTrainErr, fBatchBunchError = 0.0, 0.0
    iTrainBatches = 0 # This is a counter per epoch

    fEpochStartTime = time.time()
    fBatchBunchStartTime = fEpochStartTime

    for i in scbowUtils.nextBatch(oSentenceIterator,
                                  funcRandomIterator,
                                  oVocab=oVocab,
                                  iMaxNrOfTokens=oArgs.iMaxNrOfTokens,
                                  npaBatch_1=npaBatch_1, npaBatch_2=npaBatch_2,
                                  iBatchSize=oArgs.iBatchSize,
                                  iPos=iPos, iNeg=oArgs.iNeg):
      iTrainBatches += 1
      iNrOfBatchesSoFar += 1
      if oArgs.bVeryVerbose:
        print iTrainBatches,
  
      if oArgs.bVeryVerbose:
        print "[%d] Prediction: %s" % (iTrainBatches,
                                       forward_pass_fn(npaBatch_1, npaBatch_2))

      fBatchTrainError = train_fn(npaBatch_1, npaBatch_2, npaTargets) \
          if oArgs.sLastLayer == 'cosine' \
          else train_fn(npaBatch_1, npaBatch_2)

      if oArgs.bVeryVerbose:
        print "[%d] loss: %f" % (iTrainBatches, fBatchTrainError)

      fEpochTrainErr += fBatchTrainError
      fBatchBunchError += fBatchTrainError

      if oArgs.sUpdate == "sgd":
        scbowUtils.updateLearningRate(thsLearningRate, iNrOfBatchesSoFar,
                                      fTotalNrOfBatches, oArgs)

      # Print the results for every bunch of batches
      if oArgs.bVerbose and (iTrainBatches % iBatchBunch == 0):
        print \
            "\nEpoch %d, batch %d (took %.3f seconds). Training loss: %.6f" % \
            (iEpoch+1, iTrainBatches, time.time() - fBatchBunchStartTime, \
               fBatchBunchError / iBatchBunch)
        fBatchBunchError = 0.0
        fBatchBunchStartTime = time.time()

      if (oArgs.iStoreAtBatch is not None) and \
            (iTrainBatches % oArgs.iStoreAtBatch == 0):
        sTmpOutputFile = \
                         sOutputFile.replace(".pickle", \
                                             ".epoch_%d_batch_%d.pickle" % \
                                             (iEpoch+1,iTrainBatches) )
        storeWordEmbeddings(oNetwork, oVocab, oArgs, sTmpOutputFile)

    # Then we print the results for this epoch
    if oArgs.bVerbose:
      print "\nEpoch %d of %d took %.3f seconds" % \
          (iEpoch + 1, oArgs.iEpochs, time.time() - fEpochStartTime)
      print "Training loss: %.6f" % (fEpochTrainErr / iTrainBatches)

    # Also store the embeddings, if necessary
    if (oArgs.iStoreAtEpoch is not None) and \
       ( (iEpoch+1) % oArgs.iStoreAtEpoch == 0) and \
       ( (iEpoch+1) > oArgs.iStartStoringAt):
      sTmpOutputFile = \
                       sOutputFile.replace(".pickle", \
                                           ".end_of_epoch_%d.pickle" % \
                                           (iEpoch+1) )
      storeWordEmbeddings(oNetwork, oVocab, oArgs, sTmpOutputFile)
      bStoredEmbeddingsAtLastEpoch = True
    else:
      bStoredEmbeddingsAtLastEpoch = False

  if oArgs.bVerbose:
    fEndTime = time.time() if oArgs.bVerbose else None
    fTotalSeconds = fEndTime - fStartTime
    iHours = int(fTotalSeconds/3600)
    iMinutes = int((fTotalSeconds % 3600) / 60)
    fSeconds = fTotalSeconds % 60
    
    sHours = '' if iHours == 0 else "%d hours, " % iHours \
        if iHours > 1 else "%d hour, " % iHours 
    sMinutes = "%d minutes" % iMinutes if iMinutes != 1 \
        else "%d minute" % iMinutes
    print "\nIt took %s%s and %.2f seconds (%f seconds in total)\n" % \
        (sHours, sMinutes, fSeconds, fTotalSeconds)

  if not bStoredEmbeddingsAtLastEpoch:
    # Make it clear that this is the final one
    sOutputFile = sOutputFile.replace(".pickle", ".final.pickle")
    storeWordEmbeddings(oNetwork, oVocab, oArgs, sOutputFile)
