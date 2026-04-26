import sys
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QPushButton, QTextEdit, QVBoxLayout, QHBoxLayout, QWidget, QLabel, QLineEdit, QMessageBox, QDialog, QScrollArea, QTabWidget, QTableWidget, QTableWidgetItem, QHeaderView) 
from PyQt6.QtGui import QPixmap, QColor 
from PyQt6.QtCore import QThread, pyqtSignal, Qt

from perceptron import Perceptron
from sharedFunctions import loadData, plotFittingCurves

 
class ModelTrainingWorker(QThread):
    finished = pyqtSignal(dict) 
    logMsg = pyqtSignal(str)
    errorMsg = pyqtSignal(str)

    def __init__(self, xTrain, yTrain, xTest, yTest, learningRate, iterations):
        super().__init__()
        self.xTrain = xTrain
        self.yTrain = yTrain
        self.xTest = xTest
        self.yTest = yTest
        self.learningRate = learningRate
        self.iterations = iterations

    def run(self):
        try:
            self.logMsg.emit("Initializing Perceptron...")
            np.random.seed(42)
            model = Perceptron()

            self.logMsg.emit(f"Training started with LR = {self.learningRate}, Iterations = {self.iterations}. Please wait...")
            
            history = model.fit(
                self.xTrain, self.yTrain,
                xTest = self.xTest, yTest = self.yTest,
                learningRate = self.learningRate, 
                iterations = self.iterations
            )
            
            if self.xTest is not None:
                history['predictions'] = model.predict(self.xTest)
                history['actuals'] = self.yTest
            
            self.logMsg.emit("Training complete!")
            self.finished.emit(history)
            
        except Exception as e:
            self.errorMsg.emit(str(e))