#! /bin/bash

train_dir=../data/combined
test_dir=../data/test
out_dir=TEST_RESULTS/blstm
embeddings_file=../data/embeddings/glove.6B.100d.txt

mkdir -p ${out_dir}

# the 20 randomly drawn configurations used in the submitted results
for config in `cat blstm_configs.txt`; do 
    python ../code/blstm.py ${train_dir} ${test_dir} ${embeddings_file} ${out_dir} ${config} > blstm_config
    output_dir=${out_dir}/`cat blstm_config`
    python ../code/writeout.py ${test_dir} ${output_dir}/pred.txt ${output_dir} > msg 2>err_msg
done

# 20 random runs
#for i in `seq 1 20`; do 
#    python blstm.py ${train_dir} ${test_dir} ${embeddings_file} > config
#    output_dir=`cat config`
#    python writeout.py ${test_dir} ${output_dir}/pred.txt ${output_dir} > msg 2>err_msg
#done
