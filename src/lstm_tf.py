import os
import logging

import numpy as np
import gensim
import tensorflow as tf
from tensorflow.contrib.rnn import BasicLSTMCell, MultiRNNCell

from src.util.emoji_dataset import EmojiDataset


class LSTMModel(object):
    def __init__(self, mode="basic-lstm"):
        """LSTM Model.

        Arguments:
            mode: one of `basic-lstm`, `two-lstm`, `bi-lstm`
        """
        # type of model
        self.mode = mode

        # variables
        self.vocab_size = 58205
        self.embedding_dim = 300
        self.maxlen = 20
        self.lstm_output_size = 300
        self.num_classes = 20

        # filepath
        self.model_file = os.path.join("output", "models", "mymodel")
        self.embedding_model_path = os.path.join("data", "word2vec", "model_swm_300-6-10-low.w2v")

        # data
        self.emb_matrix = None
        self.x_train, self.y_train = None, None
        self.x_test, self.y_test = None, None
        self.x_valid, self.y_valid = None, None

        # model
        self.model = None

    def load_data(self):
        """Load dataset and generate embedding matrix.

        Dataset are load in (self.x_train, self.y_train), (self.x_test, self.y_test), (self.x_valid, self.y_valid).
        EMbedding are load in self.emb_matrix.
        """
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
        """Build the lstm model."""
        logging.info("Building model...")

        with tf.device("/cpu:0"):
            sequences = tf.placeholder("int32", shape=[None, self.maxlen])
            embedding = tf.get_variable("embedding", dtype="float32", initializer=self.emb_matrix)
            inputs = tf.nn.embedding_lookup(embedding, sequences)
        def make_cell():
            cell = BasicLSTMCell(self.lstm_output_size)
            return cell
        cells = MultiRNNCell([make_cell() for _ in range(self.maxlen)], state_is_tuple=True)
        output, state = cells(inputs)
