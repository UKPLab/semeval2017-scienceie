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
import numpy as np
import sys
import os
import re
import cPickle

def makeOutputFileName(oArgs, iNrOfEmbeddings, iEmbeddingSize):
  sShareWeights = "sharedWeights" if oArgs.bShareWeights else "noSharedWeights"
  sReg = "reg" if oArgs.bRegularize else "noReg"
  sLower = "noLc" if oArgs.bDontLowercase else "lc"
  sPreInit = "preInit" if oArgs.sWord2VecFile else "noPreInit"
  sGradientClippingBound = "noGradClip" \
      if oArgs.fGradientClippingBound is None \
      else ("gradClip_%f" % oArgs.fGradientClippingBound).replace(".", "_")

  sOutputFile = "%s_%s_%s_lr_%s_%s_epochs_%d_batch_%d_neg_%d_voc_%dx%d_%s_%s_%s.pickle" % \
      (oArgs.sLastLayer,
       sShareWeights,
       oArgs.sUpdate,
       re.sub("_?0*$", '', ("%f" % oArgs.fLearningRate).replace(".", "_")),
       sGradientClippingBound,
       oArgs.iEpochs,
       oArgs.iBatchSize,
       oArgs.iNeg,
       iNrOfEmbeddings - 1, # -1, because of the 0-embedding
       iEmbeddingSize,
       sReg,
       sLower,
       sPreInit)

  return os.path.join(oArgs.OUTPUT_DIR, sOutputFile)

def storeWordEmbeddings(sOutputFile, npaWordEmbeddings, oVocab, oArgs):
  if oArgs.bVerbose:
    print "Storing word embeddings to %s" % sOutputFile

  fhOut = open(sOutputFile, mode='wb')
  dSCBOW = {"oArgs": oArgs,
            "npaWordEmbeddings": npaWordEmbeddings,
            "oVocab": oVocab
            }
  cPickle.dump(dSCBOW, fhOut)
  fhOut.close()


class softMaxLayer_matrix(lasagne.layers.MergeLayer):
  '''
  First layer gives a vector (or a batch of vectors, really)
  Second layer gives a matrix (well, a batch of matrices)
  We return a vector of numbers, just as many as there are cols in the second 
  layer matrix (NOTE that the second input layer is a transposed version of
  the layer before it)
  '''

  def __init__(self, incomings, iEmbeddingSize, **kwargs):
    super(softMaxLayer, self).__init__(incomings, **kwargs)

    self.iEmbeddingSize = iEmbeddingSize

  def get_output_shape_for(self, input_shapes):
    # input_shapes come like this:
    #  [(batch_size, vectors_size), (batch_size, rows, cols)]
    return (input_shapes[0][0], input_shapes[1][1])

  def get_output_for(self, inputs, **kwargs):
    exps = T.exp((inputs[0].reshape((-1, self.iEmbeddingSize, 1)) * \
                    inputs[1]).sum(axis=1))

    return exps / exps.sum(axis=1).dimshuffle((0, 'x'))

class softMaxLayer(lasagne.layers.Layer):
  def __init__(self, incoming, **kwargs):
    super(softMaxLayer, self).__init__(incoming, **kwargs)

  def get_output_shape_for(self, input_shape):
    '''
    The input is just a vector of numbers.
    The output is also a vector, same size as the input.
    '''
    return input_shape

  def get_output_for(self, input, **kwargs):
    '''
    Take the exp() of all inputs, and divide by the total.
    '''
    exps = T.exp(input)

    return exps / exps.sum(axis=1).dimshuffle((0, 'x'))

class sigmoidLayer(lasagne.layers.MergeLayer):
  '''
  First layer gives a vector (or a batch of vectors, really)
  Second layer gives a matrix (well, a batch of matrices)
  We return a vector of numbers, just as many as there are cols in the second 
  layer matrix (NOTE that the second input layer is a transposed version of
  the layer before it)
  '''

  def __init__(self, incomings, iEmbeddingSize, **kwargs):
    super(sigmoidLayer, self).__init__(incomings, **kwargs)

    self.iEmbeddingSize = iEmbeddingSize

  def get_output_shape_for(self, input_shapes):
    # input_shapes come like this:
    #  [(batch_size, vectors_size), (batch_size, rows, cols)]
    return (input_shapes[0][0], input_shapes[1][1])

  def get_output_for(self, inputs, **kwargs):
    '''
    We want a dot product of every row in inputs[0] (a vector) with every
    row in inputs[1] (a matrix).
    We do this 'by hand': we do a element-wise multiplication of every vector
    in inputs[0] with every matrix in inputs[1], and sum the result.
    '''
    dots = (inputs[0].reshape((-1, self.iEmbeddingSize, 1)) * \
        inputs[1]).sum(axis=1)

    # Take the sigmoid
    return 1.0 / (1.0 + T.exp(dots))

class cosineLayer(lasagne.layers.MergeLayer):
  '''
  First layer gives a vector (or a batch of vectors, really)
  Second layer gives a matrix (well, a batch of matrices)
  We return a vector of numbers, just as many as there are cols in the second 
  layer matrix (NOTE that the second input layer is a transposed version of
  the layer before it)
  '''

  def __init__(self, incomings, iEmbeddingSize, **kwargs):
    super(cosineLayer, self).__init__(incomings, **kwargs)

    self.iEmbeddingSize = iEmbeddingSize

  def get_output_shape_for(self, input_shapes):
    # input_shapes come like this:
    #  [(batch_size, vectors_size), (batch_size, rows, cols)]
    return (input_shapes[0][0], input_shapes[1][1])

  def get_output_for(self, inputs, **kwargs):
    '''
    We want a dot product of every row in inputs[0] (a vector) with every
    row in inputs[1] (a matrix).
    We do this 'by hand': we do a element-wise multiplication of every vector
    in inputs[0] with every matrix in inputs[1], and sum the result.
    '''
    dots = (inputs[0].reshape((-1, self.iEmbeddingSize, 1)) * \
              inputs[1]).sum(axis=1)

    # Make sure the braodcasting is right
    norms_1 = T.sqrt(T.square(inputs[0]).sum(axis=1)).dimshuffle(0, 'x')
    # NOTE that the embeddings are transposed in the previous layer 
    norms_2 = T.sqrt(T.square(inputs[1]).sum(axis=1))

    norms = norms_1 * norms_2

    return dots / norms

class averageLayer(lasagne.layers.Layer):
  def __init__(self, incoming, fGradientClippingBound=None, **kwargs):
    super(averageLayer, self).__init__(incoming, **kwargs)

    self.fGradientClippingBound = fGradientClippingBound

  def get_output_shape_for(self, input_shape):
    '''
    The input is a batch of word vectors.
    The output is a single vector, same size as the input word embeddings
    In other words, since we are averaging, we loose the penultimate dimension
    '''
    return (input_shape[0], input_shape[2])

  def get_output_for(self, input, **kwargs):
    '''
    The input is a batch of word vectors.
    The output the sum of the word embeddings divided by the number of
    non-null word embeddings in the input.

    What we do with the normalizers is, we go from
    [[[.01, .02, .03],  # Word embedding sentence 1, word 1
      [.02, .3, .01],   # Word embedding sentence 1, word 2
      [.0,  .0,  .0]],
     [[.05, .06, .063], # Word embedding sentence 2, word 1
      [.034,.45, .05],
      [.01, .001, .03]],
      ...
    ]
    first to (so that is the inner non-zero sum(axis=2) part):
    [[3, 3, 0], # Number of non-zero components per vector in sentence 1
     [3, 3, 3], # Number of non-zero components per vector in sentence 1
    ...
    ]
    and finally to (so that is the outer non-zero sum(axis=1) part):
    [2, 3, ...]
    and we reshape that to:
    [[2], # Number of words in sentence 1
     [3], # Number of words in sentence 2
     ...]
    '''

    # Sums of word embeddings (so the zero embeddings don't matter here)
    sums = input.sum(axis=1) 

    # Can we do this cheaper (as in, more efficient)?
    # NOTE that we explicitly cast the output of the last sum() to floatX
    # as otherwise Theano will cast the result of 'sums / normalizers' to
    # float64
    normalisers = T.neq((T.neq(input, 0.0)).sum(axis=2, dtype='int32'), 0.0).sum(axis=1, dtype='floatX').reshape((-1, 1))
    
    averages = sums / normalisers

    if self.fGradientClippingBound is not None:
      averages = theano.gradient.grad_clip(averages,
                                           - self.fGradientClippingBound,
                                           self.fGradientClippingBound)


    return averages

class averageLayer_matrix(lasagne.layers.Layer):
  def __init__(self, incoming, iNrOfSentences=None,
               fGradientClippingBound=None, **kwargs):
    super(averageLayer_matrix, self).__init__(incoming, **kwargs)

    self.iNrOfSentences = iNrOfSentences

    self.fGradientClippingBound = fGradientClippingBound

  def get_output_shape_for(self, input_shape):
    '''
    The input is a batch of matrices of word vectors.
    The output is a batch of vectors, one for each matrix, the same size as
    the input word embeddings
    In other words, since we are averaging, we loose the penultimate dimension
    '''
    return (input_shape[0], input_shape[1], input_shape[3])

  def get_output_for(self, input, **kwargs):
    '''
    The input is a batch of matrices of word vectors.
    The output the sum of the word embeddings divided by the number of
    non-zero word embeddings in the input.

    The idea with the normalisers is similar as in the normal averageLayer
    '''

    # Sums of word embeddings (so the zero embeddings don't matter here)
    sums = input.sum(axis=2) 

    # Can we do this cheaper (as in, more efficient)?
    # NOTE that we explicitly cast the output of the last sum() to floatX
    # as otherwise Theano will cast the result of 'sums / normalizers' to
    # float64
    normalisers = T.neq((T.neq(input, 0.0)).sum(axis=3, dtype='int32'), 0.0).sum(axis=2, dtype='floatX').reshape((-1, self.iNrOfSentences, 1))
    
    averages = sums / normalisers

    if self.fGradientClippingBound is not None:
      averages = theano.gradient.grad_clip(averages,
                                           - self.fGradientClippingBound,
                                           self.fGradientClippingBound)

    return averages

class gateLayer(lasagne.layers.MergeLayer):
  def __init__(self, incomings, **kwargs):
    super(gateLayer, self).__init__(incomings, **kwargs)

  def get_output_shape_for(self, input_shapes):
    return input_shapes[1]

  def get_output_for(self, inputs, **kwargs):
    '''
    First layer is a batch of embedding indices:
    [[11,21,43,0,0],
     [234,543,0,0,0,],
     ...
    ]
    Second layer are the embeddings:
    [ [[.02, .01...],
       [.004, .005, ...],
       ...,
       .0 .0 .0 ... ,
       .0 .0 .0 ...],
      [[...],
       ....
      ]
    ]
    ''' 

    return \
        T.where(T.eq(inputs[0],0), np.float32(0.0), np.float32(1.0)).dimshuffle((0,1,'x')) * inputs[1]

class gateLayer_matrix(lasagne.layers.MergeLayer):
  def __init__(self, incomings, **kwargs):
    super(gateLayer_matrix, self).__init__(incomings, **kwargs)

  def get_output_shape_for(self, input_shapes):
    return input_shapes[1]

  def get_output_for(self, inputs, **kwargs):
    '''
    First layer is a batch of matrices of embedding indices:
    Second layer are the corresponding embeddings:
    ''' 

    return \
        T.where(T.eq(inputs[0],0), np.float32(0.0), np.float32(1.0)).dimshuffle((0,1,2,'x')) * inputs[1]

class flipLayer(lasagne.layers.Layer):
  '''
  Flip the word embeddings of the negative examples.
  So the word embeddings <we> of the negative examples will be become <-we> 
  '''
  def __init__(self, incoming, iPos=None, iNrOfSentences=None, **kwargs):
    super(flipLayer, self).__init__(incoming, **kwargs)

    # Set all the values to -1
    npaFlipper = np.ones(iNrOfSentences, dtype=np.int8) * -1
    # Except for the first one/two (the positive examples) 
    npaFlipper[0:iPos] = 1
    # Set the broadcasting right
    self.flipper = theano.shared(npaFlipper).dimshuffle('x', 0, 'x', 'x')

  def get_output_shape_for(self, input_shape):
    return input_shape

  def get_output_for(self, input, **kwargs):
    return input * self.flipper 

def preInit(tWeightShape, oW2v, oVocab):
  assert tWeightShape == (oW2v.syn0.shape[0] + 1, oW2v.syn0.shape[1])

  # Copy the embeddings
  W = np.empty(tWeightShape, dtype=np.float32)
  # NOTE that we start at 1 here (rather than 0)
  W[1:tWeightShape[0],:] = oW2v.syn0

  # Make a corresponding vocabulary
  # We start at index 1 (0 is a dummy 0.0 embedding)
  for i in range(oW2v.syn0.shape[0]):
    sWord = oW2v.index2word[i]
    iVocabIndex = i+1

    oVocab.dVocab[sWord] = iVocabIndex
    oVocab.dIndex2word[iVocabIndex] = sWord

  return W

def nextBatch(oSentenceIterator, funcRandomIterator, oVocab=None,
              npaBatch_1=None, npaBatch_2=None, iMaxNrOfTokens=None,
              iBatchSize=None, iPos=None, iNeg=None):
  '''
  This function gives back a batch to train/test on.
  It needs:
  - a sentence iterator object that yields a triple of sentences:
     (sentence n, sentence n-1, sentence n+1)
    which are next to one another in the corpus.
    These are considered positive examples.
  - a sentence iterator that yields random sentences (so single sentences) from
    the corpus. These are used as negative examples.
  - the vocabulary object is usually empty

  npaBatch_1 and npaBatch_2 should be pre-allocated arrays of size:
  npaBatch_1: (iBatchSize, iMaxNrOfTokens)
  npaBatch_2: (iBatchSize, iNeg + iPos, iMaxNrOfTokens)
  '''
  npaBatch_1[:] = 0.0 # Set the pre-allocated arrays to 0 again
  npaBatch_2[:] = 0.0
 
  iSentencePairsSampled = 0

  # NOTE that because of how we do things, the last batch isn't included if
  # it's smaller than the batch size
  for tSentenceTuple in oSentenceIterator:
    # NOTE in the toronto case, the sentence iterator already yields tokens
    aWeIndices1 = \
        [oVocab[sToken] for sToken in tSentenceTuple[0] \
           if oVocab[sToken] is not None] \
           if oSentenceIterator.sName == "torontoSentenceIterator" else \
           [oVocab[sToken] for sToken in tSentenceTuple[0].split(' ') \
              if oVocab[sToken] is not None]
  
    aWeIndices2 = \
        [oVocab[sToken] for sToken in tSentenceTuple[1] \
           if oVocab[sToken] is not None] \
           if oSentenceIterator.sName == "torontoSentenceIterator" else \
           [oVocab[sToken] for sToken in tSentenceTuple[1].split(' ') \
              if oVocab[sToken] is not None]

    aWeIndices3 = None
    if iPos == 2:
      aWeIndices3 = \
          [oVocab[sToken] for sToken in tSentenceTuple[2] \
             if oVocab[sToken] is not None] \
             if oSentenceIterator.sName == "torontoSentenceIterator" else \
             [oVocab[sToken] for sToken in tSentenceTuple[2].split(' ') \
                if oVocab[sToken] is not None]
           
    # We only deal with triples all of which members contain at least one known
    # word
    if (len(aWeIndices1) == 0) or (len(aWeIndices2) == 0) or \
          ((iPos == 2) and (len(aWeIndices3) == 0)):
      continue

    npaBatch_1[iSentencePairsSampled][0:min(len(aWeIndices1),iMaxNrOfTokens)]=\
        aWeIndices1[:iMaxNrOfTokens]
    npaBatch_2[iSentencePairsSampled][0][0:min(len(aWeIndices2),iMaxNrOfTokens)] = aWeIndices2[:iMaxNrOfTokens]
    if iPos == 2:
      npaBatch_2[iSentencePairsSampled][1][0:min(len(aWeIndices3),iMaxNrOfTokens)] = aWeIndices3[:iMaxNrOfTokens]
      
    iRandomSamples = 0
    while 1: # We break from inside the loop
      if iRandomSamples == iNeg: # So if iNeg == 0, we break right away
        break

      aWeIndicesRandom = []
      while len(aWeIndicesRandom) == 0: # Get a non-empty random sentence
        # NOTE that randomSentence is a list of tokens in the Toronto case
        randomSentence = next(funcRandomIterator)

        aWeIndicesRandom = \
            [oVocab[sToken] for sToken in randomSentence \
               if oVocab[sToken] is not None] \
               if oSentenceIterator.sName == "torontoSentenceIterator" \
               else [oVocab[sToken] for sToken in randomSentence.split(' ') \
                       if oVocab[sToken] is not None]

      iRandomSamples += 1
  
      npaBatch_2[iSentencePairsSampled][(iPos-1)+iRandomSamples][0:min(len(aWeIndicesRandom),iMaxNrOfTokens)] = aWeIndicesRandom[:iMaxNrOfTokens]
     
    iSentencePairsSampled += 1

    if iSentencePairsSampled == iBatchSize:
      # Just yield something (npaBatch_1, npaBatch_2 are filled already)
      yield 1

      # Reset
      iSentencePairsSampled = 0

def build_scbow(oArgs, iPos=None, oW2v=None, oVocab=None, tWeightShape=None):
  # Input variable for a batch of sentences (so: sentence n)
  input_var_1 = T.matrix('input_var_1', dtype='uint32')

  # Input variable for a batch of positive and negative examples
  # (so sentence n-1, sentence n+1, neg1, neg2, ...)
  input_var_2 = T.tensor3('input_var_2', dtype='uint32')

  W_init_1, W_init_2 = None, None

  # First embedding input layer
  llIn_1 = lasagne.layers.InputLayer(shape=(None, oArgs.iMaxNrOfTokens),
                                     input_var=input_var_1,
                                     name='llIn_1')

  # Second embedding input layer
  llIn_2 = lasagne.layers.InputLayer(shape=(None, iPos + oArgs.iNeg,
                                            oArgs.iMaxNrOfTokens),
                                     input_var=input_var_2,
                                     name='llIn_2')

  W_init_1 = None
  if oW2v is None:
    W_init_1 = lasagne.init.Normal().sample(tWeightShape)
  else:  ## Here is the pre-initialization
    W_init_1 = preInit(tWeightShape, oW2v, oVocab) 

  W_init_1[0,:] = 0.0

  # First embedding layer
  llEmbeddings_1 = lasagne.layers.EmbeddingLayer(
    llIn_1,
    input_size=tWeightShape[0],
    output_size=tWeightShape[1],
    W=W_init_1,
    name='llEmbeddings_1')

  llGate_1 = gateLayer([llIn_1, llEmbeddings_1], name='llGate_1')

  llAverage_1 = averageLayer(llGate_1,
                             fGradientClippingBound=oArgs.fGradientClippingBound,
                             name='llAverage_1')

  W_init_2 = None
  if not oArgs.bShareWeights:
    if oW2v is None:
      W_init_2 = lasagne.init.Normal().sample(tWeightShape)
    else: # We are not sharing, but we are pre-initializing
      preInit(W_init_2, oW2v, oVocab) 

    W_init_2[0,:] = 0.0

  # Second embedding layer, the weights tied with the first embedding layer
  llEmbeddings_2 = lasagne.layers.EmbeddingLayer(
    llIn_2,
    input_size=tWeightShape[0],
    output_size=tWeightShape[1],
    W=llEmbeddings_1.W if oArgs.bShareWeights else W_init_2,
    name='llEmbeddings_2')

  llGate_2 = gateLayer_matrix([llIn_2, llEmbeddings_2], name='llGate_2')

  llAverage_2 = None
  if oArgs.sLastLayer == 'cosine':
    llAverage_2 = \
        averageLayer_matrix(llGate_2, iNrOfSentences=iPos + oArgs.iNeg,
                            fGradientClippingBound=\
                              oArgs.fGradientClippingBound,
                            name="llAverage_2")
  else:
    llFlip_2 = flipLayer(llGate_2, iPos=iPos, iNrOfSentences=iPos + oArgs.iNeg,
                         name='llFlip_2')
    llAverage_2 = \
        averageLayer_matrix(llFlip_2, iNrOfSentences=iPos + oArgs.iNeg,
                            fGradientClippingBound=\
                              oArgs.fGradientClippingBound,
                            name="llAverage_2")

  llTranspose_2 = lasagne.layers.DimshuffleLayer(llAverage_2, (0,2,1),
                                                 name='llTranspose_2')

  llFinalLayer = None
  if oArgs.sLastLayer == 'cosine':
    llCosine = cosineLayer([llAverage_1, llTranspose_2], tWeightShape[1],
                           name='llCosine')

    llFinalLayer = softMaxLayer(llCosine, name='llSoftMax')
  else:
    llFinalLayer = sigmoidLayer([llAverage_1, llTranspose_2], tWeightShape[1],
                             name='llSigmoid')

  ### That was all the network stuff
  ### Now let's build the functions

  # Target var if needed 
  target_var = T.fmatrix('targets') if oArgs.sLastLayer == "cosine" else None

  if oArgs.bVerbose:
      print "Building prediction functions"
  # Create a loss expression for training, i.e., a scalar objective we want
  # to minimize (for our multi-class problem, it is the cross-entropy loss):
  prediction = lasagne.layers.get_output(llFinalLayer)

  if oArgs.bVerbose:
      print "Building loss functions"
  # For checking/debugging
  forward_pass_fn = theano.function([input_var_1, input_var_2], prediction)

  loss = None
  if oArgs.sLastLayer == 'cosine':
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
  else: # sigmoid
    loss = - T.log(prediction).sum(axis=1)

  if oArgs.bRegularize:
    l2_penalty = regularize_layer_params(llFinalLayer, l2)
    loss = loss + l2_penalty
  loss = loss.mean()
  
  if oArgs.bVerbose:
    print "Building update functions"

  params = lasagne.layers.get_all_params(llFinalLayer, trainable=True)

  fStartLearningRate = np.float32(oArgs.fLearningRate)
  
  thsLearningRate = None
  updates = None
  if oArgs.sUpdate == 'nesterov':
    updates = lasagne.updates.nesterov_momentum(
      loss, params, learning_rate=oArgs.fLearningRate,
      momentum=oArgs.fMomentum)
  elif oArgs.sUpdate == 'adamax':
    updates = lasagne.updates.adamax(loss, params,
                                     learning_rate=oArgs.fLearningRate)
  elif oArgs.sUpdate == 'adadelta':
    updates = lasagne.updates.adadelta(loss, params,
                                       learning_rate=oArgs.fLearningRate)
  elif oArgs.sUpdate == "sgd":
    # This doesn't work with INEX for now...

    thsLearningRate = theano.shared(fStartLearningRate)
    updates = lasagne.updates.sgd(loss, params, learning_rate=thsLearningRate)

  if oArgs.bVerbose:
    print "Building training function"
  # Compile a function performing a training step on a mini-batch (by giving
  # the updates dictionary) and returning the corresponding training loss:
  train_fn = None
  if oArgs.sLastLayer == "cosine":
    train_fn = theano.function([input_var_1, input_var_2, target_var],
                               loss, 
                               updates=updates)
  else:
    train_fn = theano.function([input_var_1, input_var_2],
                               loss, 
                               updates=updates)

  return llFinalLayer, forward_pass_fn, thsLearningRate, train_fn

def updateLearningRate(thsLearningRate, iNrOfBatchesSoFar, fTotalNrOfBatches,
                       oArgs):
  fNewLearningRate = \
      max(oArgs.fLearningRate * 0.0001,
          oArgs.fLearningRate * (1.0 - (iNrOfBatchesSoFar / fTotalNrOfBatches))
          )
  thsLearningRate.set_value(fNewLearningRate)
  if oArgs.bVeryVerbose:
    print "Batch %d of %.0f" % (iNrOfBatchesSoFar, fTotalNrOfBatches)
    print "Learning rate: %f"  % thsLearningRate.get_value()

def parseArguments():
  import argparse

  oArgsParser = argparse.ArgumentParser(description='Siamese CBOW')
  oArgsParser.add_argument('DATA',
                           help="File (in PPDB case) or directory (in Toronto Book Corpus and INEX case) to read the data from. NOTE that the program runs in aparticular input mode (INEX/PPDB/TORONTO) which is deduced from the directory/file name)")
  oArgsParser.add_argument('OUTPUT_DIR',
                           help="A file to store the final and possibly intermediate word embeddings to (in cPickle format)")
  oArgsParser.add_argument('-batch_size', metavar="INT", dest="iBatchSize",
                           help="Batch size. Default: 1",
                           type=int, action="store", default=1)
  oArgsParser.add_argument('-dont_lowercase', dest='bDontLowercase',
                           help="By default, all input text is lowercased. Use this option to prevent this.",
                           action='store_true')
  oArgsParser.add_argument('-dry_run', dest="bDryRun",
                           help="Build the network, print some statistics (if -v is on) and quit before training starts.",
                           action="store_true")
  oArgsParser.add_argument('-embedding_size', metavar="INT",
                           dest="iEmbeddingSize",
                           help="Dimensionality of the word embeddings. Default: 300",
                           type=int, action="store", default=300)
  oArgsParser.add_argument('-epochs', metavar="INT", dest="iEpochs",
                           help="Maximum number of epochs for training. Default: 10",
                           type=int, action="store", default=10)
  oArgsParser.add_argument('-gradient_clipping_bound', metavar="FLOAT",
                           dest="fGradientClippingBound",
                           help="Gradient clipping bound (so gradients will be clipped to [-FLOAT, +FLOAT]).",
                           type=float, action="store")
  oArgsParser.add_argument('-last_layer', metavar="LAYER",
                           dest="sLastLayer",
                           help="Last layer is 'cosine' or 'sigmoid'. NOTE that this choice also determines the loss function (binary cross entropy or negative sampling loss, respectively). Default: cosine",
                           action="store", default='cosine',
                           choices=['cosine', 'sigmoid'])
  oArgsParser.add_argument('-learning_rate', metavar="FLOAT",
                           dest="fLearningRate",
                           help="Learning rate. Default: 1.0",
                           type=float, action="store", default=1.0)
  oArgsParser.add_argument('-max_nr_of_tokens', metavar="INT",
                           dest="iMaxNrOfTokens",
                           help="Maximum number of tokens considered per sentence. Default: 50",
                           type=int, action="store", default=50)
  oArgsParser.add_argument('-max_nr_of_vocab_words', metavar="INT",
                           dest="iMaxNrOfVocabWords",
                           help="Maximum number of words considered. If this is not specified, all words are considered",
                           type=int, action="store")
  oArgsParser.add_argument('-momentum', metavar="FLOAT",
                           dest="fMomentum",
                           help="Momentum, only applies when 'nesterov' is used as update method (see -update). Default: 0.0",
                           type=float, action="store", default=0.0)
  oArgsParser.add_argument('-neg', metavar="INT", dest="iNeg",
                           help="Number of negative examples. Default: 1",
                           type=int, action="store", default=1)
  oArgsParser.add_argument('-regularize', dest="bRegularize",
                           help="Use l2 normalization on the parameters of the network",
                           action="store_true")
  oArgsParser.add_argument('-share_weights', dest="bShareWeights",
                           help="Turn this option on (a good idea in general) for the embedding weights of the input sentences and the other sentences to be shared.",
                           action="store_true")
  oArgsParser.add_argument('-start_storing_at', metavar="INT",
                           dest="iStartStoringAt",
                           help="Start storing embeddings at epoch number INT. Default: 0. I.e. start storing right away (if -store_at_epoch is on, that is)",
                           action="store", type=int, default=0)  
  oArgsParser.add_argument('-store_at_batch', metavar="INT",
                           dest="iStoreAtBatch",
                           help="Store embeddings every INT batches.",
                           action="store", type=int, default=None)
  oArgsParser.add_argument('-store_at_epoch', dest="iStoreAtEpoch",
                           metavar="INT",
                           help="Store embeddings every INT epochs (so 1 for storing at the end of every epoch, 10 for for storing every 10 epochs, etc.).",
                           action="store", type=int)
  oArgsParser.add_argument('-update', metavar="UPDATE_ALGORITHM",
                           dest="sUpdate",
                           help="Update algorithm. Options are 'adadelta', 'adamax', 'nesterov' (which uses momentum) and 'sgd'. Default: 'adadelta'",
                           action="store", default='adadelta',
                           choices=['adadelta', 'adamax', 'sgd',
                                    'nesterov'])
  oArgsParser.add_argument("-v", dest="bVerbose", action="store_true",
                           help="Be verbose")
  oArgsParser.add_argument('-vocab', dest="sVocabFile", metavar="FILE",
                           help="A vocabulary file is simply a file, SORTED BY FREQUENCY of frequence<SPACE>word lines. You can take the top n of these (which is why it should be sorted by frequency). See -max_nr_of_vocab_words.",
                           action="store")
  oArgsParser.add_argument("-vv", dest="bVeryVerbose", action="store_true",
                           help="Be very verbose")
  oArgsParser.add_argument('-w2v', dest="sWord2VecFile", metavar="FILE",
                           help="A word2vec model can be used to initialize the weights for words in the vocabulary file from (missing words just get a random embedding). If the weights are not initialized this way, they will be trained from scratch.",
                           action="store")
  oArgs = oArgsParser.parse_args()

  if (oArgs.sVocabFile is None) and (oArgs.sWord2VecFile is None):
    print >>sys.stderr, "[ERROR] Please specify either a word2vec file or a vocab file"
    exit(1)

  if oArgs.bVeryVerbose: # If we are very verbose, we are also just verbose
    oArgs.bVerbose=True

  return oArgs

if __name__ == "__main__":
  oArgs = parseArguments()

  iPos=2

  # Prepare Theano variables for inputs and targets
  # Input variable for a batch of left sentences
  input_var_1 = T.matrix('input_var_1', dtype='int32')
  # Input variable for a batch of right sentences, plus negative examples
  input_var_2 = T.tensor3('input_var_2', dtype='int32')

  target_var = T.fmatrix('targets') if oArgs.sLastLayer == "cosine" else None

  npaWordEmbeddings = np.array([[.1, .2, .3, .4],
                                [.2, .3, .4, 5],
                                [-.7, -.4, -.5, -.6],
                                [-.8, -.9, -.45, -.56],
                                [.2131, .213, .434, .652]]
                               ).astype(np.float32)
  dModel = None
  if oArgs.sStoredModel is not None:
    import cPickle
    fhFile = open(oArgs.sStoredModel, mode='rb')
    dModel = cPickle.load(fhFile)
    fhFile.close()
    npaWordEmbeddings = dModel['npaWordEmbeddings']

  npaTargets = np.zeros((oArgs.iBatchSize, oArgs.iNeg + iPos),
                        dtype=np.float32)
  if iPos == 2:
    npaTargets[:,[0,1]] = .5
  else:
    npaTargets[:,0] = 1.0
  
  iNrOfEmbeddings, iEmbeddingSize = npaWordEmbeddings.shape
  
  npaInput_1 = np.array([[0,1], [0,1], [1,0]]).astype('int32')
  npaInput_2 = np.array([[2,1], [3,2], [1,0]]).astype('int32')
  npaInput_3 = np.array([[2,3], [1,2], [1,4]]).astype('int32')

  iMaxNrOfTokens = 2

  network = build_scbow(input_var_1, input_var_2,
                        iBatchSize=oArgs.iBatchSize,
                        iPos=iPos,
                        iNeg=oArgs.iNeg, iMaxNrOfTokens=iMaxNrOfTokens,
                        tWeightShape=npaWordEmbeddings.shape,
                        npaWordEmbeddings=npaWordEmbeddings,
                        sLastLayer=oArgs.sLastLayer,
                        bVerbose=oArgs.bVerbose)

  prediction = lasagne.layers.get_output(network)

  forward_pass_fn = theano.function([input_var_1, input_var_2],
                                    prediction)

  # We want to maximize the sum of the log probabilities, so we want to
  # minimize this loss objective
  # NOTE that we expect the word embeddings of the negative examples to be 
  # reversed (as in: -1 * word embedding) 
  npaLossBoost = np.ones(oArgs.iNeg + iPos, dtype=np.float32)
  #npaLossBoost[0:iPos] = oArgs.fLossBoost

  loss = None
  if oArgs.sLastLayer == 'cosine':
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
  else: # sigmoid
    loss = - (T.log(prediction) * npaLossBoost).sum(axis=1)

  loss_fn = theano.function([prediction, target_var], loss)

  # Pre-allocate memory
  npaBatch_1 = np.zeros((oArgs.iBatchSize, iMaxNrOfTokens),
                        dtype=np.int8)
  npaBatch_2 = np.zeros((oArgs.iBatchSize, oArgs.iNeg + iPos, iMaxNrOfTokens),
                        dtype=np.int8)

  # Check that the network is producing anything
  for i in moetNog(npaInput_1, npaInput_2, npaInput_3,
                               npaBatch_1, npaBatch_2,
                               iNeg=oArgs.iNeg, iBatchSize=oArgs.iBatchSize,
                               bShuffle=False):
    # Check the batch itself
    print "Batch (1):\n%s\n      (2)\n%s" % (npaBatch_1, npaBatch_2)
    
    npaPredictions = forward_pass_fn(npaBatch_1, npaBatch_2)
    
    print "Predictions (%s):\n%s" % (npaPredictions[0].dtype, npaPredictions)

    L = loss_fn(npaPredictions, npaTargets)
    print "Loss: %s" % L

    
