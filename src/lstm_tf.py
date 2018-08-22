import os
import time
import logging

import numpy as np
import gensim
import tensorflow as tf
# from tensorflow.contrib.rnn import BasicLSTMCell
from tensorflow.nn.rnn_cell import BasicLSTMCell

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
        logging.info("train data size: {}".format(self.x_train.shape[0]))
        logging.info("test data size: {}".format(self.x_test.shape[0]))

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

        # with tf.device("/cpu:0"):
        X = tf.placeholder("int32", shape=[None, self.maxlen])
        y_ = tf.placeholder(tf.float64, shape=[None, self.num_classes])
        # [batch_size, maxlen, embedding_dim]
        embedding = tf.get_variable("embedding", dtype=tf.float64, initializer=self.emb_matrix)

        inputs = tf.nn.embedding_lookup(embedding, X)
        cell = BasicLSTMCell(self.lstm_output_size)
        # initial_state = rnn_cell.zero_state(batch_size, dtype=tf.float32)
        outputs, state = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float64)
        # w = tf.Variable(initial_value=tf.truncated_normal([self.lstm_output_size, self.num_classes], stddev=0.1),
        #                 dtype=tf.float32,
        #                 name="W")
        # b = tf.Variable(tf.constant(0.1, shape=[self.num_classes]),
        #                 dtype=tf.float32,
        #                 name="B")
        w = tf.get_variable("w", shape=[self.lstm_output_size, self.num_classes], dtype=tf.float64)
        b = tf.get_variable("b", shape=[self.num_classes], dtype=tf.float64)

        act = tf.matmul(outputs[:, -1, :], w) + b
        y = tf.nn.softmax(act)
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), axis=[1]))
        train_step = tf.train.AdagradOptimizer(0.1).minimize(cross_entropy)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        train_size = self.x_train.shape[0]
        logging.info("Training model...")
        s = time.time()
        for i in range(10):
            print("epoch {}".format(i))
            j = 0
            while j + 64 < train_size:
                batch_xs = self.x_train[j: j + 64]
                batch_ys = self.y_train[j: j + 64]
                sess.run(train_step, feed_dict={X: batch_xs, y_: batch_ys})
                j += 64
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))
            print(accuracy.eval({X: self.x_test, y_: self.y_test}))
        t = time.time()
        logging.info("Train model use {}s".format(t - s))





