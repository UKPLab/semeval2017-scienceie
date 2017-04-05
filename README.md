# SemEval 2017 Task 10: ScienceIE - Extracting Keyphrases and Relations from Scientific Publications

This repository contains the code needed to reproduce our results for the shared task [ScienceIE] [science-ie] reported in Eger et al., *EELECTION at SemEval-2017 Task 10: Ensemble of nEural Learners for kEyphrase ClassificaTION*.

Please cite the paper as:

```
@InProceedings{semeval2017-eger-eelection,
  author    = {Eger, Steffen and Do Dinh, Erik-Lân and Kutsnezov, Ilia and Kiaeeha, Masoud and Gurevych, Iryna},
  title     = {{EELECTION at SemEval-2017 Task 10: Ensemble of nEural Learners for kEyphrase ClassificaTION}},
  booktitle = {Proceedings of the 11th International Workshop on Semantic Evaluation (SemEval 2017)},
  month     = {August},
  year      = {2017},
  address   = {Vancouver, Canada},
  publisher = {Association for Computational Linguistics},
  pages     = {(to appear)},
  url       = {https://github.com/UKPLab/semeval2017-scienceie}
}
```

> **Abstract:** This paper describes our approach to the SemEval 2017 Task 10: “Extracting Keyphrases and Relations from Scientific Publications”, specifically to Subtask (B): “Classification of identified keyphrases”.
> We explored three different deep learning approaches: a character-level convolutional neural network (CNN), a stacked learner with an MLP meta-classifier, and an attention based Bi-LSTM. From these approaches, we created an ensemble of differently hyper-parameterized systems, achieving a micro-F 1 -score of 0.63 on the test data. Our approach ranks 2nd (score of 1st placed system: 0.64) out of five according to this official score. 
> However, we erroneously trained 2 out of 3 neural nets (the stacker and the CNN) on only roughly 15% of the full data (the original development set). When trained on the full data (training+development), our ensemble has a micro-F 1 -score of 0.69.

https://www.ukp.tu-darmstadt.de/

https://www.tu-darmstadt.de/

Don't hesitate to contact us if something is broken (and it shouldn't be) or if you have further questions.

> This repository contains experimental software and 
is published for the sole purpose of giving additional 
background details on the respective publication. 

## Project structure

* `crawl/` -- this folder contains scripts to crawl additional Elsevier articles
* `scripts_submission/` -- shell scripts for running the individual systems
* `skip-thoughts/` -- document classifier, incorporating code from [[https://bitbucket.org/TomKenter/siamese-cbow/src]]
* `requirements.txt` -- a text file with the names of the required Python modules
* The data can be obtained from the shared task website: [[https://scienceie.github.io/resources.html]]

## Requirements

* 64-bit Linux versions (not tested on other platforms)
* Python 2.7
* Python modules in the `requirements.txt` file
* [keras] [keras]
* Suitable word embeddings in **text format** (e.g. [glove.6B.zip] [glove_embed])

## Running the experiments

* Stacker:

    ```
    train=train3+dev/ # training data directory, each file therein in *.ann,*.xml,*.txt form 
    embeddings=/data/wordvecs/ExtendedDependencyBasedSkip-gram/wiki_extvec_words # Komninos embeddings 
    cs=4 
    
    python stackedLearner.py ${train} test/scienceie2017_test_unlabelled/ ${embeddings} ${cs} None document > stacker.out 
    
    The predictions will be in the file "stacker.out". Further do: 
    
    ./extract.py < stacker.out > stacker.extracted 
    ./writeout.py test/scienceie2017_test_unlabelled stacker.extracted MY_PRED_DIR > msg 2>err_msg 
    python scripts/eval.py GoldTest/semeval_articles_test/ MY_PRED_DIR rel
    ```

* Char-CNN: 

    ```
    baseCMD="convNet.py train3+dev/ test/scienceie2017_test_unlabelled/ empty"
    cs=4 
    L=50 
    M=80 
    R=50 
    nfilter=300 
    filter_length=3
    document=document
    python ${baseCMD} ${cs} ${L} ${M} ${R} ${nfilter} ${filter_length} ${document} > conv.out
    
    The predictions will be in the file "conv.out". Further do: 
    
    ./extract.py < conv.out > conv.extracted 
    ./writeout.py test/scienceie2017_test_unlabelled conv.extracted MY_PRED_DIR > msg 2>err_msg 
    python scripts/eval.py GoldTest/semeval_articles_test/ MY_PRED_DIR rel
    ```
* AB-LSTM:
    * Modify `run_blstm.sh` to reflect your data paths, then run it.
    * This will create and run 20 different configurations of the AB-LSTM, and write output ANN files into resp. output folders.

   [keras]: <https://keras.io/>
   [tensorflow]: <https://www.tensorflow.org/>
   [glove_embed]: <http://nlp.stanford.edu/data/glove.6B.zip>
   [science-ie]: <https://scienceie.github.io/>
