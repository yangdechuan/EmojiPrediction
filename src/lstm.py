import os
import sys
import logging
import time

import numpy as np
import gensim
import tensorflow as tf
from tensorflow import keras

from src.util.emoji_dataset import EmojiDataset
from src.util.util import ClassificationMacroF1

class LSTMModel(object):
    def __init__(self, mode="basic"):
        """LSTM model.

        Arguments:
            mode: one of `basic`, `two-layers`, `bi-dir`
        """
        self.mode = mode

        self.vocab_size = 10000
        self.embedding_dim = 300
        self.maxlen = 20
        self.lstm_output_size = 300
        self.num_classes = 20

        self.model_file = os.path.join("output", "models", "mymodel")

        self.embedding_model_path = os.path.join("data", "word2vec", "model_swm_300-6-10-low.w2v")
        self.emb_matrix = None

        self.model = None
        self.x_train, self.y_train = None, None
        self.x_test, self.y_test = None, None
        self.x_valid, self.y_valid = None, None

    def load_data(self):
        """Load dataset and generate embedding matrix."""
        emoji_dataset = EmojiDataset()

        logging.info("Loading data...")
        (self.x_train, self.y_train), (self.x_test, self.y_test), (self.x_valid, self.y_valid) \
            = emoji_dataset.load_data(num_words=self.vocab_size, maxlen=self.maxlen, x_padding=True, y_categorial=True)
        logging.info("Finish loading data.")

        word_index = emoji_dataset.get_wordindex()

        logging.info("Loading word2vec...")
        w2v_model = gensim.models.KeyedVectors.load_word2vec_format(self.embedding_model_path, binary=False)

        emb_matrix = np.zeros([self.vocab_size + 1, self.embedding_dim])
        for (word, index) in word_index.items():
            if index > self.vocab_size:
                break
            try:
                emb_matrix[index] = w2v_model.wv[word]
            except KeyError:  # word not found in the pretrained vectors will be set zeros
                pass
        self.emb_matrix = emb_matrix
        logging.info("Finish loading word2vec and generate embedding matrix.")

    def build_model(self):
        """Build the model"""
        logging.info("Building model...")
        model = keras.Sequential()

        # init with pretrained word embedding
        embedding_layer = keras.layers.Embedding(self.vocab_size + 1,  # due to mask_zero
                                                 self.embedding_dim,
                                                 input_length=20,
                                                 embeddings_initializer=keras.initializers.Constant(self.emb_matrix),
                                                 # mask_zero=False,
                                                 trainable=True)

        model.add(embedding_layer)
        if self.mode == "basic":
            model.add(keras.layers.LSTM(self.lstm_output_size, return_sequences=False, bias_initializer=keras.initializers.Ones()))
        elif self.mode == "two-layers":
            model.add(keras.layers.LSTM(self.lstm_output_size, return_sequences=True, bias_initializer=keras.initializers.Ones()))
            model.add(keras.layers.LSTM(self.lstm_output_size, return_sequences=False, bias_initializer=keras.initializers.Ones()))
        elif self.mode == "bi-dir":
            model.add(keras.layers.Bidirectional(
                keras.layers.LSTM(self.lstm_output_size, bias_initializer=keras.initializers.Ones()),merge_mode="sum")
            )
        else:
            logging.error("Error lstm mode!!")
            sys.exit()

        model.add(keras.layers.Dense(self.num_classes))
        model.add(keras.layers.Activation("softmax"))

        optimizer = tf.train.AdamOptimizer(0.001)
        model.compile(optimizer,
                      loss="categorical_crossentropy",
                      metrics=["accuracy"])

        self.model = model
        logging.info("Model has been build.")

    def train_model(self):
        """Train the model."""
        logging.info("Starting training...")

        early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=2)
        macro_f1 = ClassificationMacroF1()

        start = time.time()
        self.model.fit(x=self.x_train, y=self.y_train,
                       batch_size=64,
                       epochs=10,
                       verbose=2,
                       callbacks=[macro_f1],
                       validation_data=(self.x_test, self.y_test),
                       )
        stop = time.time()
        logging.info("Finish training.\n")
        logging.info("Training used {} s.".format(stop - start))

    def save_model(self, filename=None):
        if filename is None:
            filename = self.model_file
        self.model.save(filename)
        logging.info("trained model has been saved into {}".format(filename))
