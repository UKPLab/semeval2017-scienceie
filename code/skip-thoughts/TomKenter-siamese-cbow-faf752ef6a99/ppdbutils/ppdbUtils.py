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
import sys
import gzip
import numpy as np

sys.path.append("./tokenizationutils")
import tokenizer

def fastForward(fhFile):
  c = ''
  try:
    c = fhFile.read(1)
  except UnicodeDecodeError:
    pass

  while c != "\n":
    try:
      c = fhFile.read(1)
    except UnicodeDecodeError:
      pass

class ppdb:
  def __init__(self, sFile, bTokenMode=False, bSingleSentence=False,
               bRandom=False, sName=''):
    self.sFile = sFile
    self.bRandom = bRandom
    self.bTokenMode = bTokenMode
    self.bSingleSentence = bSingleSentence
    self.sName = sName

    self.iNrOfCharacters = None
    self.iTotalNrOfSentences = None

    if self.bRandom:
      if 'ppdb-1.0-xl-phrasal' in sFile:
        # Very simple shortcut to avoid calculating this number over and over
        # again
        self.iNrOfCharacters = 4225022351
        self.iLastLineLength = 606
        self.iTotalNrOfSentences = 6925144
      else: # Yet, to be safe...
        import subprocess
        
        # Get the number of characters in the file
        sCommand = "zcat %s | wc -c" % sFile if sFile.endswith(".gz") \
            else "wc -c < %s" % sFile
        # When wc reads from stdin it just gives a number as output
        self.iNrOfCharacters = \
            int(subprocess.check_output(sCommand, shell=True) )

        # Get the number of lines in the file
        sCommand = "zcat %s | wc -l" % sFile if sFile.endswith(".gz") \
            else "wc -l < %s" % sFile
        # When wc reads from stdin it just gives a number as output
        self.iTotalNrOfSentences = \
            int(subprocess.check_output(sCommand, shell=True) )

        # Get the length of the last line
        sCommand = "tail -1 %s | wc -c" % sFile
        self.iLastLineLength = \
            int(subprocess.check_output(sCommand, shell=True))

  def __iter__(self):
    '''
    NOTE that in random mode, this iterator will NEVER stop of itself.
    NOTE that in random mode, reading from a gzipped is VERY slow
    '''
    fhFile = None
    if self.sFile.endswith(".gz"):
      fhFile = gzip.open(self.sFile, 'rb')
    else:
      fhFile = codecs.open(self.sFile, mode='r', encoding='utf8')

    if self.bRandom:
      fhFile.seek(max(0, (np.random.randint(0, self.iNrOfCharacters) - \
                            self.iLastLineLength) ) )
      fastForward(fhFile)

    aLine = None
    for sLine in fhFile:
      if self.bRandom:
        while len(sLine) == 0: # We might have read past the very last line
          # Sample again
          fhFile.seek(max(0, (np.random.randint(0, self.iNrOfCharacters) - \
                                self.iLastLineLength) ) )
          fastForward(fhFile)
          sLine = fhFile.readline() # Read a new line

      if sLine.startswith('#'): # Allow for comment
        continue

      if self.sFile.endswith(".gz"):
        # For some or another reason we have to do something about the encoding
        # in the gzip case
        aLine = sLine.encode("utf8").split(" ||| ")
      else:
        aLine = sLine.split(" ||| ")

      if len(aLine) < 3:
        print >>sys.stderr, "[ERROR %s] in line '%s'" % (self.sName, sLine)
        exit(1)

      aTokens1 = [x for x in tokenizer.tokenizeSentence(aLine[1], bLowerCase=True) if (x != 'rrb') and (x != 'lrb')]
      aTokens2 = [x for x in tokenizer.tokenizeSentence(aLine[2], bLowerCase=True) if (x != 'rrb') and (x != 'lrb')]

      if (len(aTokens1) > 0) and (len(aTokens2) > 0):
        if self.bTokenMode:
          if aTokens1 != aTokens2:
            yield {"iLabel": 1, # We only have positive instances
                   "aTokens1": aTokens1,
                   "aTokens2": aTokens2}
        else: # Yield the sentences, the same way the InexIterator does
          if self.bRandom:
            # Just output one sentence
            yield ' '.join(aTokens1) if np.random.randint(0,2) == 1 \
                             else ' '.join(aTokens2)
          elif self.bSingleSentence:
            yield ' '.join(aTokens1)
            yield ' '.join(aTokens2)
          elif(aTokens1 != aTokens2):
            # NOTE that we ignore identical sentences here as there isn't much
            # to learn from them.
            yield (' '.join(aTokens1), ' '.join(aTokens2))

      if self.bRandom:
        fhFile.seek(max(0, (np.random.randint(0, self.iNrOfCharacters) - \
                            self.iLastLineLength) ) )
        fastForward(fhFile)

    fhFile.close()

# The next bit is just to test something
if __name__ == "__main__":
  import argparse
  oArgsParser = argparse.ArgumentParser(description='PPDB utilities')
  oArgsParser.add_argument('PPDB_FILE',
                           help="A PPDB file of the 'phrasal' type.")
  oArgsParser.add_argument('-random', dest="bRandom", help="Random mode",
                           action="store_true")
  oArgsParser.add_argument('-single_sentence', dest="bSingleSentence",
                           help="Single sentence mode",
                           action="store_true")
  oArgsParser.add_argument('-d', dest="bDebug", help="Dubugging mode",
                           action="store_true")
  oArgsParser.add_argument('-v', dest="bVerbose", help="Be verbose",
                           action="store_true")
  oArgs = oArgsParser.parse_args()

  if oArgs.bDebug:
    import pdb
    pdb.set_trace()

  oPPDB = ppdb(oArgs.PPDB_FILE, bSingleSentence=oArgs.bSingleSentence,
               bRandom=oArgs.bRandom)

  for result in oPPDB:
    if oArgs.bRandom or oArgs.bSingleSentence:
      print "%d\t%s" % (len(result.split(' ')), result)
    else:
      print "%s - %s" % (result[0], result[1])
