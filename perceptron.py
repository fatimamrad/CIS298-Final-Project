import numpy as np
import copy
from sharedFunctions import oneHotEncode, sigmoid, sigmoidDerivative, proxyError, initializeWeights


class Perceptron:
    def __init__(self, inputSize = 1024, outputSize = 10): # initializes the network with 1024 input pixels and 10 output nodes
        self.W, self.b = initializeWeights(inputSize, outputSize)
    

    # computed the forward pass
    def forward(self, X):
        Z = np.dot(X, self.W) + self.b # calculate the linear combination of inputs, weights, and biases
        A = sigmoid(Z) # apply the sigmoid activation function to get the output probabilities
        return Z, A


    # computes the gradients for the weights and biases using backpropagation
    def backward(self, X, yTrue, A, Z):
        N = X.shape[0]
        numOutputs = yTrue.shape[1] # get number of output nodes
        
        dA = (A - yTrue) / (N * numOutputs) # calculate the derivative of the proxy error function
        dZ = dA * sigmoidDerivative(Z) # apply chain rule with the derivative of the sigmoid function
        
        dW = np.dot(X.T, dZ) # compute the gradient for the weights matrix
        db = np.sum(dZ, axis = 0, keepdims = True) # compute the gradient for the bias array
        
        return dW, db

    def updateWeights(self, dW, db, learningRate):
        self.W -= learningRate * dW # update weight matrix
        self.b -= learningRate * db # update bias array


    # generates predictions for given set of inputs
    def predict(self, X):
        _, A = self.forward(X) # get the activated outputs from the forward pass
        return np.argmax(A, axis = 1)


    # calculates the actual misclassification error rate
    def evaluateError(self, X, yLabels):
        predictions = self.predict(X) # get the models predictions
        incorrect = np.sum(predictions != yLabels) # count how many predictions dont match the true
        return incorrect / len(yLabels)  # return the ratio of incorrect predictions to total samples
    

    # trains the perceptron using gradient descent over a specified number of iterations
    def fit(self, xTrain, yTrain, xTest = None, yTest = None, learningRate = 0.1, iterations = 100):
        yTrainEncoded = oneHotEncode(yTrain, numClasses = 10)
        
        hasTestData = False
        if xTest is not None and yTest is not None: # checks if test data was provided
            yTestEncoded = oneHotEncode(yTest, numClasses = 10)
            hasTestData = True
            
        history = {
            'trainProxy': [], 'testProxy': [],
            'trainMisc': [], 'testMisc': []
        }
        
        bestMiscError = float('inf')
        bestW = None
        bestB = None

        for i in range(iterations): # gradient desecent training loop
            Z, A = self.forward(xTrain)
            
            trainProxy = proxyError(yTrainEncoded, A) # calculates proxy error
            history['trainProxy'].append(trainProxy) # adds the value to the dictionary
            
            trainMisc = self.evaluateError(xTrain, yTrain) # this line and the next do the same but for miclassification error
            history['trainMisc'].append(trainMisc)
            
            if hasTestData: # checks if test data exists, if it does then it foes the same for that data
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