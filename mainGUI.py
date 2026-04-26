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

class ResultsDialog(QDialog):
    def __init__(self, history):
        super().__init__()
        self.setWindowTitle("Perceptron Training Results")
        self.resize(850, 600)
        
        layout = QVBoxLayout(self)
        
        tabs = QTabWidget()
        
        graphTab = QWidget()
        graphLayout = QVBoxLayout(graphTab)
        
        scrollArea = QScrollArea()
        scrollArea.setWidgetResizable(True)
        scrollContent = QWidget()
        scrollLayout = QVBoxLayout(scrollContent)
        
        miscLabel = QLabel()
        miscPixmap = QPixmap("graphs/Perceptron_Fitting_Curve_Misclassification.png")
        if not miscPixmap.isNull():
            miscLabel.setPixmap(miscPixmap.scaledToWidth(800, Qt.TransformationMode.SmoothTransformation))
            miscLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
            scrollLayout.addWidget(miscLabel)
        
        proxyLabel = QLabel()
        proxyPixmap = QPixmap("graphs/Perceptron_Fitting_Curve_Proxy.png")
        if not proxyPixmap.isNull():
            proxyLabel.setPixmap(proxyPixmap.scaledToWidth(800, Qt.TransformationMode.SmoothTransformation))
            proxyLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
            scrollLayout.addWidget(proxyLabel)
            
        scrollContent.setLayout(scrollLayout)
        scrollArea.setWidget(scrollContent)
        graphLayout.addWidget(scrollArea)
        tabs.addTab(graphTab, "Fitting Curves")
        
        tableTab = QWidget()
        tableLayout = QVBoxLayout(tableTab)
        
        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["Sample #", "Predicted Label", "Actual Label", "Correct?"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        
        if 'predictions' in history and 'actuals' in history:
            preds = history['predictions']
            actuals = history['actuals']
            self.table.setRowCount(len(preds))
            
            for i in range(len(preds)):
                pred = preds[i]
                actual = actuals[i]
                isCorrect = "Yes" if pred == actual else "No"
                
                itemIdx = QTableWidgetItem(str(i))
                itemPred = QTableWidgetItem(str(pred))
                itemActual = QTableWidgetItem(str(actual))
                itemCorrect = QTableWidgetItem(isCorrect)
            
                if isCorrect == "Yes":
                    itemCorrect.setForeground(QColor("green"))
                else:
                    itemCorrect.setForeground(QColor("red"))
                    itemPred.setForeground(QColor("red")) 
                    
                self.table.setItem(i, 0, itemIdx)
                self.table.setItem(i, 1, itemPred)
                self.table.setItem(i, 2, itemActual)
                self.table.setItem(i, 3, itemCorrect)
                
        tableLayout.addWidget(self.table)
        tabs.addTab(tableTab, "Test Data Predictions")
        
        layout.addWidget(tabs)
        
        closeBtn = QPushButton("Close")
        closeBtn.clicked.connect(self.accept)
        layout.addWidget(closeBtn)