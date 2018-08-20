import os
import json

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer


def load_json_data(filename):
    """Load json twitter data.

    Arguments:
        filename: json file name
    Returns:
        texts: []: twitter text list
        labels: []: twitter emoji labels list
    """
    with open(filename, "r", encoding="utf-8") as fr:
        json_data = json.load(fr)
        texts, labels = [], []
        for dic in json_data:
            texts.append(dic["text"])
            labels.append(dic["label"])
    return texts, labels


class EmojiDataset(object):
    def __init__(self, data_dir="data"):
        """Emoji dataset class.

        Arguments:
            data_dir: path to data dir
        """
        self.num_classes = 20

        self.train_filename = os.path.join(data_dir, "us_train.json")
        self.test_filename = os.path.join(data_dir, "us_test.json")
        self.valid_filename = os.path.join(data_dir, "us_valid.json")

    def load_data(self, num_words=None, maxlen=None, x_padding=True, y_categorial=True):
        """Load train, test, valid data.

        Arguments:
            num_words: number of words considered
            x_padding: whether padding the sentence with 0
            y_categorial: whether transform y into categorical form
            maxlen: maximum length of all sequences
        Returns:
            Tuple of Numpy arrays: (x_train, y_train), (x_test, y_test), (x_valid, y_valid)
            x_train: shape=(data_size, maxlen)  dtype=int32
            y_train: shape=(data_size, num_class)  dtype=int32
        """
        train_texts, train_labels = load_json_data(self.train_filename)
        test_texts, test_labels = load_json_data(self.test_filename)
        valid_texts, valid_labels = load_json_data(self.valid_filename)

        tokenizer = Tokenizer(num_words=num_words)
        tokenizer.fit_on_texts(train_texts)

        self.word_index = tokenizer.word_index

        train_data = tokenizer.texts_to_sequences(train_texts)
        test_data = tokenizer.texts_to_sequences(test_texts)
        valid_data = tokenizer.texts_to_sequences(valid_texts)

        if x_padding:
            train_data = tf.keras.preprocessing.sequence.pad_sequences(train_data,
                                                                       maxlen=maxlen,
                                                                       padding="pre",
                                                                       truncating="pre",
                                                                       value=0)
            test_data = tf.keras.preprocessing.sequence.pad_sequences(test_data,
                                                                      maxlen=maxlen,
                                                                      padding="pre",
                                                                      truncating="pre",
                                                                      value=0)
            valid_data = tf.keras.preprocessing.sequence.pad_sequences(valid_data,
                                                                       maxlen=maxlen,
                                                                       padding="pre",
                                                                       truncating="pre",
                                                                       value=0)

        if y_categorial:
            train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=self.num_classes)
            test_labels = tf.keras.utils.to_categorical(test_labels, num_classes=self.num_classes)
            valid_labels = tf.keras.utils.to_categorical(valid_labels, num_classes=self.num_classes)

        self.train_data, self.train_labels = train_data, train_labels
        self.test_data, self.test_labels = test_data, test_labels
        self.valid_data, self.valid_labels = valid_data, valid_labels

        return (np.array(self.train_data), self.train_labels.astype("int32")), \
               (np.array(self.test_data), self.test_labels.astype("int32")), \
               (np.array(self.valid_data), self.valid_labels.astype("int32"))

    def get_wordindex(self):
        """Get word index dict.

        Returns:
            A dict with word as key, index as value.
            e.g. {'the': 1, 'user': 2, ...}
        """
        return self.word_index




