import numpy as np
import matplotlib.pyplot as plt


# reads the dataset and seperates the deatures from the labels, the first 1024 columns are pixel data and the 1025th column is the integer class label
def loadData(filepath):
    print(f"Loading data from {filepath}...")
    data = np.loadtxt(filepath)
    X = data[:, :1024]
    X = X / 16.0 
    y = data[:, 1024].astype(int)
    return X, y


# converts a 1 dimensional array of integer labels into a 2D matrix of one hot encoded vectors
def oneHotEncode(y, numClasses=10):
    numSamples = y.shape[0]
    yEncoded = np.zeros((numSamples, numClasses))
    yEncoded[np.arange(numSamples), y] = 1
    return yEncoded


# applies the sigmoid activation function, im clipping inputs to [-500, 500] to prevent mathematical overflow errors in numpy
def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


# calculates the derivative of the sigmoid
def sigmoidDerivative(z):
    sig = sigmoid(z)
    return sig * (1.0 - sig)


# calculates the proxy error 
def proxyError(yTrue, yPred):
    return np.mean(np.mean(0.5 * (yTrue - yPred)**2, axis = 1)) # takes the mean across the outputs, then takes the mean across all samples in the batch


# initializes the weights and biases
def initializeWeights(inputSize, outputSize):
    W = np.random.uniform(low = -0.1, high = 0.1, size = (inputSize, outputSize)) # weight matrix
    b = np.random.uniform(low = -0.1, high = 0.1, size = (1, outputSize)) # bias array
    return W, b


# fitting curve graphs
def plotFittingCurves(history, modelName):
    # proxy error fitting curve
    plt.figure(figsize = (8, 5))
    plt.plot(history['trainProxy'], label = 'Train Proxy Error')
    plt.plot(history['testProxy'], label = 'Test Proxy Error')
    plt.title(f'{modelName}: Proxy Error over Iterations')
    plt.xlabel('Iterations')
    plt.ylabel('Half MSE Proxy Error')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"graphs/{modelName}_Fitting_Curve_Proxy.png")
    plt.close()

    # misclassification error over iterations fitting curve
    plt.figure(figsize = (8, 5))
    plt.plot(history['trainMisc'], label = 'Train Misclassification')
    plt.plot(history['testMisc'], label = 'Test Misclassification')
    plt.title(f'{modelName}: Misclassification Error over Iterations')
    plt.xlabel('Iterations')
    plt.ylabel('Error Rate')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"graphs/{modelName}_Fitting_Curve_Misclassification.png")
    plt.close()