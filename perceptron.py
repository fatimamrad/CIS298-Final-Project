import numpy as np
import copy
from sharedFunctions import oneHotEncode, sigmoid, sigmoidDerivative, proxyError, initializeWeights


class Perceptron:
    def __init__(self, inputSize = 1024, outputSize = 10): 
        self.W, self.b = initializeWeights(inputSize, outputSize)
    

    
    def forward(self, X):
        Z = np.dot(X, self.W) + self.b 
        A = sigmoid(Z) 
        return Z, A


    def backward(self, X, yTrue, A, Z):
        N = X.shape[0]
        numOutputs = yTrue.shape[1] 
        
        dA = (A - yTrue) / (N * numOutputs) 
        dZ = dA * sigmoidDerivative(Z) 
        
        dW = np.dot(X.T, dZ) 
        db = np.sum(dZ, axis = 0, keepdims = True) 
        
        return dW, db
