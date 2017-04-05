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

import codecs
import numpy as np
import sys

class vocabulary:
  '''
  The input file should:
  - be sorted by frequency
  - consist of space-separated lines, like:
    10 the
    9 a
    9 an
    8 ...
  '''
  def __init__(self, sFile, iMaxNrOfWords=None, sInputMode="wordfreq"):
    self.dVocab = {"__DUMMY_WORD__": 0}
    self.dIndex2word = {0: "__DUMMY_WORD__"}
    self.sInputMode = sInputMode

    if sFile is not None:
      fhFile = codecs.open(sFile, mode='r', encoding='utf8')
  
      # We intentionally start at 1. No word has index 0
      # This is because of the embedding layer later on...
      iIndex = 1     
      iLineNr = 0
      for sLine in fhFile:
        iLineNr += 1
      
        # Because we have a dummy word, the maximum number of words is actually 
        # 1 higher than iMaxNrOfWords
        if (iMaxNrOfWords is None) or (iIndex <= iMaxNrOfWords):
          try:
            if self.sInputMode == 'freqword':
              sFreq, sWord = sLine.strip().split(' ')
            else: # wordfreq
              sWord, sFreq = sLine.strip().split(' ')
  
            if sWord in self.dVocab:
              print >>sys.stderr, \
                "[WARNING]: word '%s' is already in" % sWord
            else:
              self.dVocab[sWord] = iIndex
              self.dIndex2word[iIndex] = sWord
              iIndex += 1
          except ValueError:
            sLine = sLine[:-1] if sLine.endswith("\n") else sLine
            print >>sys.stderr, \
                "[WARNING]: error in line %d: '%s'" % (iLineNr, sLine)
        else:
          break
      
      fhFile.close()
  
    self.iNrOfWords = len(self.dVocab)

  def init_fromList(self, aWordList):
    # We intentionally start at 1. No word has index 0
    # This is because of the embedding layer later on...
    iIndex = 1     
    for sWord in aWordList:
      if sWord in self.dVocab:
        print >>sys.stderr, \
            "[WARNING]: word '%s' is already in" % sWord
      else:
        self.dVocab[sWord] = iIndex
        self.dIndex2word[iIndex] = sWord
        iIndex += 1
    
    self.iNrOfWords = len(self.dVocab)

  def __iter__(self):
    '''
    Iterate of the words in sorted order. Ignore the first word.
    '''
    for i in range(1, len(self.dIndex2word)):
      return oVocab.dIndex2word[i]

  def __getitem__(self, sKey):
    '''
    Returns an index (NOT the frequency)
    '''
    try:
      return self.dVocab[sKey]
    except KeyError:
      return None

  def index2word(self, iIndex):
    try:
      return self.dIndex2word[iIndex]
    except KeyError:
      return None

  def write(self):
    for iIndex in range(1, self.iNrOfWords):
      print "[%d] %s" % (iIndex, self.index2word(iIndex))

if __name__ == "__main__":
  import argparse
  oArgsParser = argparse.ArgumentParser(description='Vocabulary utils')
  oArgsParser.add_argument('VOCAB_FILE')
  oArgsParser.add_argument('-input_mode', metavar="MODE",
                           dest="sInputMode",
                           help="Format of the input file 'wordfreq' or 'freqword'. Default: 'wordfreq'",
                           action="store", choices=("wordfreq", "freqword"))
  oArgsParser.add_argument('-max_nr_of_vocab_words', metavar="INT",
                           dest="iMaxNrOfVocabWords",
                           help="Maximum number of words considered (default: infinity (i.e. all words in the vocab file are considered))",
                           type=int, action="store", default=np.inf)
  oArgsParser.add_argument("-d", dest="bDebug", action="store_true")
  oArgsParser.add_argument("-v", dest="bVerbose", action="store_true")
  oArgs = oArgsParser.parse_args()

  oVocab = vocabulary(oArgs.VOCAB_FILE,
                      iMaxNrOfVocabWords=oArgs.iMaxNrOfVocabWords,
                      sInputMode=oArgs.sInputMode)

  for i in range(10):
    print "%s: %d" % (oVocab.dIndex2word[i], i)
