Documentation: SE
14.12.2016


1. Represent your document as a 300d embedding as in skip-thoughts/TomKenter-siamese-cbow-faf752ef6a99/docRepresenter2.py
   NOTE: You must use these embeddings: skip-thoughts/TomKenter-siamese-cbow-faf752ef6a99/cosine_sharedWeights_adadelta_lr_1_noGradClip_epochs_2_batch_100_neg_2_voc_65536x300_noReg_lc_noPreInit_vocab_65535.end_of_epoch_2.pickle
   If you don't like these, re-train the doc classifier as in: skip-thoughts/TomKenter-siamese-cbow-faf752ef6a99/docClassify.py

2. Once your documents are represented, have a look at skip-thoughts/TomKenter-siamese-cbow-faf752ef6a99/runDoClassifier.py
   To classify your documents 
