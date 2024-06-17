import numpy as np


def get_accuracy(tests, predictions):
    return np.sum(tests == predictions) / len(tests)
