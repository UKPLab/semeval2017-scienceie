#! /usr/bin/python

import sys,numpy as np

mu=float(sys.argv[1])
sigma=float(sys.argv[2])

positive=True

d=np.random.normal(mu,sigma)

if positive and d<1:
	print 1
else:
	print int(d)
