#! /usr/bin/python

import sys
from extras import read_and_write

# SAMPLE USAGE:
# ./writeout.py test/scienceie2017_test_unlabelled TEST_RESULTS/maj.pred OUR_PREDS/ > msg 2>err_msg

def read_preds(fn):
  h=[]
  for line in open(fn):
	h.append(line.strip())
  return h

src=sys.argv[1]
preds=sys.argv[2]
outdir=sys.argv[3]

pred_list = read_preds(preds)

read_and_write(src,pred_list,outdir)
