import os

from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics

from src.util.emoji_dataset import EmojiDataset, load_json_data


class SVMModel(object):
    def __init__(self):
        """SVM model."""
        self.x_train, self.y_train = None, None
        self.x_test, self.y_test = None, None
        self.x_valid, self.y_valid = None, None

        self.model = None

    def load_data(self):
        self.x_train, self.y_train = load_json_data(os.path.join("data", "us_train.json"))
        self.x_test, self.y_test = load_json_data(os.path.join("data", "us_test.json"))
        self.x_valid, self.y_valid = load_json_data(os.path.join("data", "us_valid.json"))

        # Use TF-IDF features.
        tfidf = TfidfVectorizer(ngram_range=(1,2))
        self.x_train = tfidf.fit_transform(self.x_train)
        self.x_test = tfidf.transform(self.x_test)
        self.x_valid = tfidf.transform(self.x_valid)

    def build_model(self):
        self.model = LinearSVC()

    def train_model(self):
        self.model.fit(self.x_train, self.y_train)

    def test_model(self):
        y_pred = self.model.predict(self.x_test)
        f1_score = metrics.f1_score(self.y_test, y_pred, average="macro")
        accuracy = metrics.accuracy_score(self.y_test, y_pred)
        print("f1-score: {}".format(f1_score))
        print("accuracy: {}".format(accuracy))
