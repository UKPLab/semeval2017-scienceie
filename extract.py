#! /usr/bin/python

import sys

cue="==PREDICTING=="
read=False

for line in sys.stdin:
  line = line.strip()
  if line.startswith(cue): 
	read=True
	continue
  if read==True:
	print line
