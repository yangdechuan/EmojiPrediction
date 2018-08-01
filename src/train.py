from src.svm import SVMModel
from src.lstm import LSTMModel

def svm():
    model = SVMModel()
    model.load_data()
    model.build_model()
    model.train_model()
    model.test_model()

def lstm():
    model = LSTMModel(mode="two-layers")
    model.load_data()
    model.build_model()
    model.train_model()
    model.save_model()