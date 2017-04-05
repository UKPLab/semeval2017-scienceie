#! /bin/bash

baseCMD="convNet.py train3+dev/ test/scienceie2017_test_unlabelled/ empty"
out=TEST_RESULTS/char_corr_3_fixed/


for i in 0 1 2 3 4 5; do 

  cs=`./draw.py`
  L=`./drawNormal.py 30 5`
  M=`./drawNormal.py 50 10`
  R=`./drawNormal.py 30 5`
  nfilter=`./drawNormal.py 250 50`
  filter_length=`./drawVariable.py 2 2 3 3 3 4`
  document=`./drawVariable.py document document sentence`

  python ${baseCMD} ${cs} ${L} ${M} ${R} ${nfilter} ${filter_length} ${document} > ${out}/${cs}_${L}_${M}_${R}_${nfilter}_${filter_length}_${document}_${i}.out 

done &

for i in 6 7 8 9 10 11; do

  cs=`./draw.py`
  L=`./drawNormal.py 30 5`
  M=`./drawNormal.py 50 10`
  R=`./drawNormal.py 30 5`
  nfilter=`./drawNormal.py 250 50`

  filter_length=`./drawVariable.py 2 2 3 3 3 4`
  document=`./drawVariable.py document document sentence`

  python ${baseCMD} ${cs} ${L} ${M} ${R} ${nfilter} ${filter_length} ${document} > ${out}/${cs}_${L}_${M}_${R}_${nfilter}_${filter_length}_${document}_${i}.out

done &

for i in 12 13 14 15 16 17; do

  cs=`./draw.py`
  L=`./drawNormal.py 30 5`
  M=`./drawNormal.py 50 10`
  R=`./drawNormal.py 30 5`
  nfilter=`./drawNormal.py 250 50`

  filter_length=`./drawVariable.py 2 2 3 3 3 4`
  document=`./drawVariable.py document document sentence`

  python ${baseCMD} ${cs} ${L} ${M} ${R} ${nfilter} ${filter_length} ${document} > ${out}/${cs}_${L}_${M}_${R}_${nfilter}_${filter_length}_${document}_${i}.out

done &


