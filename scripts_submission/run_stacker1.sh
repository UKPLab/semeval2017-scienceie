#! /bin/bash

train_dir=../data/combined
test_dir=../data/test
out_dir=TEST_RESULTS/FIXED1

baseCMD="../code/stackedLearner.py ${train_dir} ${test_dir}"

mkdir -p ${out_dir}


for i in 0 1 2 3 4 5; do 

  cs=`../code/draw.py`

  python ${baseCMD} ../data/embeddings/glove.6B.50d.txt ${cs} > ${out_dir}/stacker_glove50_${i}_${cs}.out & 

  cs=`../code/draw.py`

  python ${baseCMD} ../data/embeddings/glove.6B.100d.txt ${cs} > ${out_dir}/stacker_glove100_${i}_${cs}.out &

  cs=`../code/draw.py`

  python ${baseCMD} ../data/embeddings/glove.42B.300d.txt ${cs} > ${out_dir}/stacker_glove300_${i}_${cs}.out

  cs=`../code/draw.py`

  python ${baseCMD} ../data/embeddings/wiki_extvec_words ${cs} > ${out_dir}/stacker_kominos100_${i}_${cs}.out &

  cs=`../code/draw.py`

  python ${baseCMD} ../data/embeddings/bow2.words ${cs} > ${out_dir}/stacker_levy_${i}_${cs}.out & 

done
