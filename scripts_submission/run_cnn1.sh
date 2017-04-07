#! /bin/bash

train_dir=../data/combined
test_dir=../data/test
out_dir=TEST_RESULTS/char_fixed

baseCMD="../code/convNet.py ${train_dir} ${test_dir} empty"

mkdir -p ${out_dir}

for i in 0 1 2 3 4 5; do 

  cs=`../code/draw.py`
  L=`../code/drawNormal.py 30 5`
  M=`../code/drawNormal.py 50 10`
  R=`../code/drawNormal.py 30 5`
  nfilter=`../code/drawNormal.py 250 50`

  python ${baseCMD} ${cs} ${L} ${M} ${R} ${nfilter} > ${out_dir}/${cs}_${L}_${M}_${R}_${nfilter}_${i}.out 

done

for i in 6 7 8 9 10 11; do

  cs=`../code/draw.py`
  L=`../code/drawNormal.py 30 5`
  M=`../code/drawNormal.py 50 10`
  R=`../code/drawNormal.py 30 5`
  nfilter=`../code/drawNormal.py 250 50`

  python ${baseCMD} ${cs} ${L} ${M} ${R} ${nfilter} > ${out_dir}/${cs}_${L}_${M}_${R}_${nfilter}_${i}.out

done

for i in 12 13 14 15 16 17; do

  cs=`../code/draw.py`
  L=`../code/drawNormal.py 30 5`
  M=`../code/drawNormal.py 50 10`
  R=`../code/drawNormal.py 30 5`
  nfilter=`../code/drawNormal.py 250 50`

  python ${baseCMD} ${cs} ${L} ${M} ${R} ${nfilter} > ${out_dir}/${cs}_${L}_${M}_${R}_${nfilter}_${i}.out

done
