from src.svm import SVMModel


def svm():
    model = SVMModel()
    model.load_data()
    model.build_model()
    model.train_model()
    model.test_model()