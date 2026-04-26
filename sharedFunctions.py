import numpy as np
import matplotlib.pyplot as plt


def loadData(filepath):
    print(f"Loading data from {filepath}...")
    data = np.loadtxt(filepath)
    X = data[:, :1024]
    X = X / 16.0 
    y = data[:, 1024].astype(int)
    return X, y


def oneHotEncode(y, numClasses=10):
    numSamples = y.shape[0]
    yEncoded = np.zeros((numSamples, numClasses))
    yEncoded[np.arange(numSamples), y] = 1
    return yEncoded


def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))

