import numpy as np
from tensorflow import keras

def macro_f1(y_true, y_pred):
    """Compute macro_f1 score give pred ys and true labels.
    We will change prob arys into one hot arys to ensure its correctness.
    Args:
        y_true: true labels, Tensor: (samples, max_class).
                Each sample has only 1., 0.s else.
                e.g. [[0., 1., 0., 0., 0., ..., 0.], [0.,, 0., 1., 0., ..., 0.], ... []]
        y_pred: model prediction outputs, Tensor: (samples, max_class).
                Each sample are probabilities across 20 classes, whose sum is near to 1.0.
                e.g. [[0.00232432, 0.100023, 0.678438, 0.0001, 0.0324132, ..., 0.0030], ... []]
    Returns:
        Macro-f1 value, Tensor.
    """
    # count appeared classes
    num_classes = np.sum(np.any(y_true, axis=0))

    def f1(precision_, recall_):
        return (2.0 * precision_ * recall_) / (precision_ + recall_ + np.finfo(float).eps)

    def precision(y_true, y_pred):
        true_positives = np.sum(y_true * y_pred, axis=0)  # axis=0: add by column
        pred_positives = np.sum(y_pred, axis=0)
        p = true_positives / (pred_positives + np.finfo(float).eps)
        return p

    def recall(y_true, y_pred):
        true_positives = np.sum(y_true * y_pred, axis=0)
        real_positives = np.sum(y_true, axis=0)
        r = true_positives / (real_positives + np.finfo(float).eps)
        return r

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    f1_total = np.sum(f1(p, r))

    macrof1 = f1_total / num_classes
    return macrof1

class ClassificationMacroF1(keras.callbacks.Callback):
    def __init__(self):
        self.macrof1s = 0.

    def on_epoch_end(self, epoch, logs={}):
        y_pred = np.asarray(self.model.predict(self.validation_data[0]))
        y_true_onehot = np.asarray(self.validation_data[1])
        y_pred_onehot = np.zeros_like(y_pred)
        y_pred_onehot[np.arange(y_pred.shape[0]), np.argmax(y_pred, axis=-1)] = 1.0
        self.macrof1s = macro_f1(y_true_onehot, y_pred_onehot)

        print(">> test dataset size: prediction: {}, true label: {}".format(y_pred_onehot.shape, y_true_onehot.shape))
        print(">> macro f1 after this epoch: {:.4f}\n".format(self.macrof1s * 100))
        return