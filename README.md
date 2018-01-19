# EELECTION at SemEval-2017 Task 10: Ensemble of nEural Learners for kEyphrase ClassificaTION
## SemEval 2017 Task 10: ScienceIE - Extracting Keyphrases and Relations from Scientific Publications.


This repository contains the code needed to reproduce our results for the shared task [ScienceIE] [science-ie] reported in Eger et al., *[EELECTION at SemEval-2017 Task 10: Ensemble of nEural Learners for kEyphrase ClassificaTION](https://www.aclweb.org/anthology/S/S17/S17-2163.pdf)*. 

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

> **Abstract:** This paper describes our approach to the SemEval 2017 Task 10: "Extracting Keyphrases and Relations from Scientific Publications", specifically to Subtask (B): "Classification of identified keyphrases".
> We explored three different deep learning approaches: a character-level convolutional neural network (CNN), a stacked learner with an MLP meta-classifier, and an attention based Bi-LSTM. From these approaches, we created an ensemble of differently hyper-parameterized systems, achieving a micro-F1-score of 0.63 on the test data. Our approach ranks 2nd (score of 1st placed system: 0.64) out of four according to this official score. 
> However, we erroneously trained 2 out of 3 neural nets (the stacker and the CNN) on only roughly 15% of the full data, namely, the original development set. When trained on the full data (training+development), our ensemble has a micro-F1-score of 0.69.

Contact persons: 
  * Steffen Eger, eger@ukp.informatik.tu-darmstadt.de
  * Erik-Lân Do Dinh, dodinh@ukp.informatik.tu-darmstadt.de
  * Ilia Kutsnezov, kutsnezov@ukp.informatik.tu-darmstadt.de
  * Masoud Kiaeeha, kiaeeha@ukp.informatik.tu-darmstadt.de

https://www.ukp.tu-darmstadt.de/

https://www.tu-darmstadt.de/

Don't hesitate to contact us if something is broken (and it shouldn't be) or if you have further questions.

> This repository contains experimental software and 
is published for the sole purpose of giving additional 
background details on the respective publication. 

## Project structure

* `code/`
   * `crawl/` -- this folder contains scripts to crawl additional Elsevier articles
   * `skip-thoughts/` -- document classifier, incorporating code from https://bitbucket.org/TomKenter/siamese-cbow/src
* `data/` -- the data can be obtained from the shared task website: https://scienceie.github.io/resources.html
* `scripts_submission/` -- shell scripts for running the individual systems
* `scripts/` -- evaluation scripts provided by the task organizers
* `requirements.txt` -- a text file with the names of the required Python modules

## Requirements

* 64-bit Linux versions (not tested on other platforms)
* Python 2.7
* Python modules in the `requirements.txt` file
* [keras] with [tensorflow] or [theano]
* Suitable word embeddings in **text format** (see below)

## Running the experiments

To run the experiments described in our paper you have to aquire following resources.

Put the following embeddings into `data/embeddings`:
* Glove word embeddings: [glove] (glove.6B.zip and glove.42B.300d.zip)
* Komninos word embeddings: [komninos] (wiki_extvec.gz)
* Levy word embeddings: [levy] (Bag of Words (k = 2) [words])

Put the training, dev and test data into `data/train`, `data/dev` and `data/test`, respectively. For running the experiment scripts below, also create `data/combined`, and copy the `train` and `dev` data into it.
* Training and test data: [science-ie-data]

Further, the keras version has a bug regarding unicode, which has to be fixed as e.g. described in [keras-fix].

The scripts to start the experiments can be found in `scripts_submission`.

   [keras]: <https://keras.io/>
   [tensorflow]: <https://www.tensorflow.org/>
   [theano]: <https://github.com/Theano/Theano>
   [glove]: <http://nlp.stanford.edu/projects/glove>
   [komninos]: <https://www.cs.york.ac.uk/nlp/extvec/>
   [levy]: <https://levyomer.wordpress.com/2014/04/25/dependency-based-word-embeddings/>
   [science-ie]: <https://scienceie.github.io/>
   [science-ie-data]: <https://scienceie.github.io/resources.html>
   [keras-fix]: <https://github.com/fchollet/keras/issues/1072#issuecomment-241682313>
