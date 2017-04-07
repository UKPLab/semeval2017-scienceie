#!/usr/bin/python
# -*- coding: UTF-8 -*-

import collections
import logging
import keras.backend as keras_backend
import numpy as np
import os
import random
import sklearn.datasets
import sklearn.metrics
import sys
from attention_lstm import AttentionLSTM
from reader import ScienceIEBratReader
from extras import VSM
from representation import IndexListMapper
from keras.engine import Input
from keras.layers import merge, Activation, Convolution1D, Dense, Dropout, Lambda
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.models import Model, Sequential
from keras.preprocessing import sequence as sequence_module
from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import probas_to_classes, to_categorical
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight

def get_folder(parent_path):
    fid = os.path.basename(__file__)[:-3] + "_" + "-".join(map(str, config))
#    output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "results-" + fid)
    output_dir = os.path.join(parent_path, "results-" + fid)
    print output_dir
    return output_dir


def build_lstm(output_dim, embeddings):

    loss_function = "categorical_crossentropy"

    # this is the placeholder tensor for the input sequences
    sequence = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype="int32")

    # this embedding layer will transform the sequences of integers
    embedded = Embedding(embeddings.shape[0], embeddings.shape[1], input_length=MAX_SEQUENCE_LENGTH, weights=[embeddings], trainable=True)(sequence)

    # 4 convolution layers (each 1000 filters)
    cnn = [Convolution1D(filter_length=filters, nb_filter=1000, border_mode="same") for filters in [2, 3, 5, 7]]
    # concatenate
    merged_cnn = merge([cnn(embedded) for cnn in cnn], mode="concat")
    # create attention vector from max-pooled convoluted
    maxpool = Lambda(lambda x: keras_backend.max(x, axis=1, keepdims=False), output_shape=lambda x: (x[0], x[2]))
    attention_vector = maxpool(merged_cnn)

    forwards = AttentionLSTM(64, attention_vector)(embedded)
    backwards = AttentionLSTM(64, attention_vector, go_backwards=True)(embedded)

    # concatenate the outputs of the 2 LSTM layers
    bi_lstm = merge([forwards, backwards], mode="concat", concat_axis=-1)

    after_dropout = Dropout(0.5)(bi_lstm)

    # softmax output layer
    output = Dense(output_dim=output_dim, activation="softmax")(after_dropout)

    # the complete omdel
    model = Model(input=sequence, output=output)

    # try using different optimizers and different optimizer configs
    model.compile("adagrad", loss_function, metrics=["accuracy"])

    return model


def read_and_map(src, mapper, y_values = None):
    r = ScienceIEBratReader(src)
    X = []
    y = []
    entities = []
    for document in r.read():
        for entity in document.entities:
            if entity.uid in document.cover_index:  # only proceed if entity has been successfully mapped to tokens
                X += [mapper.to_repr(entity, document)]
                y += [entity.etype]
                entities += [entity]

    y_values = y_values if y_values is not None else list(set(y))
    y = np.array([y_values.index(y_val) for y_val in y])
    return X, y, y_values, entities


def acc(correct, total):
    return 1.0 * correct / total


# Usage: 
# python blstm.py path_to_training path_to_testing path_to_emb_file [configuration_string]
if __name__ == "__main__":
    verbose_logging = 0

    np.random.seed(4)  # for reproducibility
    FORMAT = "[%(levelname)s] [%(asctime)-15s] %(message)s"
    logging.basicConfig(format=FORMAT)
    logger = logging.getLogger("main")
    logger.setLevel(logging.INFO)

    if len(sys.argv) < 5:
        print("Usage: python blstm.py path_to_training path_to_testing path_to_emb_file output_path [configuration_string]")
        sys.exit()

    train_src = sys.argv[1]
    dev_src = sys.argv[2]
    vsm_path = sys.argv[3]
    output_path = sys.argv[4]

    # if a configuration string is given, use that (for re-running experiments)
    if len(sys.argv) > 5:
        imported_config = sys.argv[5]
        MAX_SEQUENCE_LENGTH, MAX_NB_WORDS, EPOCHS, BATCH_SIZE, CLASS_WEIGHTS = imported_config.split("_")[-1].split("-")

        MAX_SEQUENCE_LENGTH = int(MAX_SEQUENCE_LENGTH) if MAX_SEQUENCE_LENGTH != "None" else None
        MAX_NB_WORDS = int(MAX_NB_WORDS) if MAX_NB_WORDS != "None" else None
        EPOCHS = int(EPOCHS)
        BATCH_SIZE = int(BATCH_SIZE)
        CLASS_WEIGHTS = True if CLASS_WEIGHTS == "True" else False
        config = [MAX_SEQUENCE_LENGTH, MAX_NB_WORDS, EPOCHS, BATCH_SIZE, CLASS_WEIGHTS]
        output_dir = get_folder(output_path)

        # don't rerun if experiment already exists
        if os.path.exists(output_dir):
            logger.info("SKIPPING " + output_dir + " since it already exists.")
            sys.exit()
    # otherwise use a random config
    else:
        # yeah, this is a bit sloppy, don't look too closely...
        while True:
            MAX_SEQUENCE_LENGTH = random.choice([5,10,20])
            MAX_NB_WORDS = random.choice([None, 1000, 2500])
            EPOCHS = 10
            BATCH_SIZE = random.choice([12,24,32])
            CLASS_WEIGHTS = random.choice([True, False])
            config = [MAX_SEQUENCE_LENGTH, MAX_NB_WORDS, EPOCHS, BATCH_SIZE, CLASS_WEIGHTS]
            output_dir = get_folder(output_path)

            # only run non-existing configurations (i.e. exit loop)
            if not os.path.exists(output_dir):
                break
    print(output_dir)
    logger.info("Writing to path: " + output_dir)
    os.makedirs(output_dir)

    logger.info("Loading VSM")
    vsm = VSM(vsm_path)
    EMBEDDING_DIM = vsm.dim

    # word -> embedding
    embeddings_index = vsm.map

    # word -> idx
    r = ScienceIEBratReader(train_src)
    texts = []
    for document in r.read():
        texts.append(document.text)
    tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    word_index = tokenizer.word_index
    word_index["#BEGIN_OF_TEXT#".lower()] = 0
    word_index["#END_OF_TEXT#".lower()] = 0

    # idx -> embeddings
    embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    mapper = IndexListMapper(vsm, word_index, MAX_SEQUENCE_LENGTH, "both")

    logger.info("Reading training data")
    X_train, y_train, y_values, _ = read_and_map(train_src, mapper)

    logger.info("Training a model")
    if CLASS_WEIGHTS:
        class_weight = dict(enumerate(compute_class_weight("balanced", range(0, len(set(y_values))), y_train)))
    else:
        class_weight = None
    logger.info("Config: " + str(config))
    classifier = build_lstm(len(y_values)+1, embedding_matrix)
    logger.info("Summary: ")
    classifier.summary()
    classifier.fit(X_train, to_categorical(y_train, len(y_values)+1), batch_size=BATCH_SIZE, class_weight=class_weight, verbose=1, nb_epoch=EPOCHS)

    logger.debug(classifier)

    logger.info("Reading test data")
    X_dev, y_dev_gold, _, entities = read_and_map(dev_src, mapper, y_values)

    logger.info("Testing")
    y_dev_pred = probas_to_classes(classifier.predict(X_dev))

    # output entities
    with open(os.path.join(output_dir, "entities.txt"), "w") as f:
        for entity in entities:
            f.write(str(entity) + '\n')

    # output gold labels
    with open(os.path.join(output_dir, "gold.txt"), "w") as f:
        for i in xrange(len(y_dev_gold)):
            f.write(y_values[y_dev_gold[i]] + '\n')

    # output predicted labels
    with open(os.path.join(output_dir, "pred.txt"), "w") as f:
        for i in xrange(len(y_dev_pred)):
            f.write(y_values[y_dev_pred[i]] + '\n')
