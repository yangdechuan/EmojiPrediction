from src.svm import SVMModel
# from src.lstm import LSTMModel
from src.lstm_tf2 import LSTMModel

def svm():
    model = SVMModel()
    model.load_data()
    model.build_model()
    model.train_model()
    model.test_model()

def lstm(mode=None):
    model = LSTMModel(mode=mode)
    model.load_data()
    model.build_model()
    # model.train_model()
    # model.save_model()