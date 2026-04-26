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

    def updateWeights(self, dW, db, learningRate):
        self.W -= learningRate * dW 
        self.b -= learningRate * db 


    
    def predict(self, X):
        _, A = self.forward(X) 
        return np.argmax(A, axis = 1)


    
    def evaluateError(self, X, yLabels):
        predictions = self.predict(X) 
        incorrect = np.sum(predictions != yLabels) 
        return incorrect / len(yLabels)
    
    def fit(self, xTrain, yTrain, xTest = None, yTest = None, learningRate = 0.1, iterations = 100):
        yTrainEncoded = oneHotEncode(yTrain, numClasses = 10)
        
        hasTestData = False
        if xTest is not None and yTest is not None: 
            yTestEncoded = oneHotEncode(yTest, numClasses = 10)
            hasTestData = True
            
        history = {
            'trainProxy': [], 'testProxy': [],
            'trainMisc': [], 'testMisc': []
        }
        
        bestMiscError = float('inf')
        bestW = None
        bestB = None

        for i in range(iterations): 
            Z, A = self.forward(xTrain)
            
            trainProxy = proxyError(yTrainEncoded, A) 
            history['trainProxy'].append(trainProxy) 
            
            trainMisc = self.evaluateError(xTrain, yTrain) 
            history['trainMisc'].append(trainMisc)
            
            if hasTestData: 
                _, aTest = self.forward(xTest)
                testProxy = proxyError(yTestEncoded, aTest)
                testMisc = self.evaluateError(xTest, yTest)
                history['testProxy'].append(testProxy)
                history['testMisc'].append(testMisc)
                
            if trainMisc < bestMiscError:
                bestMiscError = trainMisc
                bestW = copy.deepcopy(self.W)
                bestB = copy.deepcopy(self.b)
                
            dW, db = self.backward(xTrain, yTrainEncoded, A, Z) # back propagation to get the gradients
            self.updateWeights(dW, db, learningRate)

        self.W = bestW
        self.b = bestB
        
        return history