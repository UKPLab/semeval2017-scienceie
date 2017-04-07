#! /bin/bash

train_dir=../data/combined
test_dir=../data/test
out_dir=TEST_RESULTS/FIXED_doc

baseCMD="../code/stackedLearner.py ${train_dir} ${test_dir}"

mkdir -p ${out_dir}

# file where documents are classified
dc=../skip-thoughts/TomKenter-siamese-cbow-faf752ef6a99/PREDS/train+dev+test.out


for i in 0 1 2 3 4 5; do 

  cs=`../code/draw.py`

  python ${baseCMD} ../data/embeddings/glove.6B.50d.txt ${cs} ${dc} document > ${out_dir}/dc_stacker_glove50_${i}_${cs}.out & 

  cs=`../code/draw.py`

  python ${baseCMD} ../data/embeddings/glove.6B.100d.txt ${cs} ${dc} document > ${out_dir}/dc_stacker_glove100_${i}_${cs}.out &

  cs=`../code/draw.py`

  python ${baseCMD} ../data/embeddings/glove.42B.300d.txt ${cs} ${dc} document > ${out_dir}/dc_stacker_glove300_${i}_${cs}.out

  cs=`../code/draw.py`

  python ${baseCMD} ../data/embeddings/wiki_extvec_words ${cs} None document > ${out_dir}/stacker_kominos100_${i}_${cs}.out &

  cs=`../code/draw.py`

  python ${baseCMD} ../data/embeddings/bow2.words ${cs} None document > ${out_dir}/stacker_levy_${i}_${cs}.out & 

done


