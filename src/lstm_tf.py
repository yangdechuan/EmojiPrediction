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

        X = tf.placeholder("int32", shape=[None, self.maxlen], name="x")
        y_ = tf.placeholder(tf.float64, shape=[None, self.num_classes], name="y_true")
        with tf.name_scope("embedding"):
            # [batch_size, maxlen, embedding_dim]
            embedding = tf.get_variable("embedding", dtype=tf.float64, initializer=self.emb_matrix)
        
        with tf.name_scope("lstm"):
            inputs = tf.nn.embedding_lookup(embedding, X, name="inputs")
            cell = BasicLSTMCell(self.lstm_output_size, name="cell")
            # initial_state = rnn_cell.zero_state(batch_size, dtype=tf.float32)
            outputs, state = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float64)
        
        with tf.name_scope("fc"):
            w = tf.get_variable("w", shape=[self.lstm_output_size, self.num_classes], dtype=tf.float64)
            b = tf.get_variable("b", shape=[self.num_classes], dtype=tf.float64)
            act = tf.matmul(outputs[:, -1, :], w) + b
            y = tf.nn.softmax(act)
            tf.summary.histogram("w", w)
            tf.summary.histogram("b", b)

        with tf.name_scope("train"):
            cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), axis=[1]))

            # Define train step.
            train_step = tf.train.AdagradOptimizer(0.1).minimize(cross_entropy)

            # Define accuracy.
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))

            tf.summary.scalar("cross_entropy", cross_entropy)
            tf.summary.scalar("accuracy", accuracy)
            

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        summ_acc = tf.summary.merge_all(scope="train")
        summ_fc = tf.summary.merge_all(scope="fc")

        writer = tf.summary.FileWriter("tmp")
        writer.add_graph(sess.graph)

        logging.info("Training model...")
        data_size = self.x_train.shape[0]
        s = time.time()
        for i in range(10):
            print("epoch {}".format(i))
            j = 0
            while j + 100 < data_size:
                batch_xs = self.x_train[j: j + 100]
                batch_ys = self.y_train[j: j + 100]
                sess.run(train_step, feed_dict={X: batch_xs, y_: batch_ys})
                j += 100

                if j % 10000 == 0:
                    summ_fc_tmp = sess.run(summ_fc, feed_dict={X: batch_xs, y_: batch_ys})
                    summ_acc_tmp = sess.run(summ_acc, feed_dict={X: self.x_test, y_: self.y_test})
                    writer.add_summary(summ_fc_tmp, global_step=j + i * data_size)
                    writer.add_summary(summ_acc_tmp, global_step=j + i * data_size)
            
            acc = sess.run(accuracy, feed_dict={X: self.x_test, y_: self.y_test})
            logging.info("Accuracy: {}".format(acc))
        t = time.time()
        logging.info("Train model use {}s".format(t - s))





