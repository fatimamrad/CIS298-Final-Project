import sys
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QPushButton, QTextEdit, QVBoxLayout, QHBoxLayout, QWidget, QLabel, QLineEdit, QMessageBox, QDialog, QScrollArea, QTabWidget, QTableWidget, QTableWidgetItem, QHeaderView) 
from PyQt6.QtGui import QPixmap, QColor 
from PyQt6.QtCore import QThread, pyqtSignal, Qt

from perceptron import Perceptron
from sharedFunctions import loadData, plotFittingCurves

# worker thread class to train the model without causing the GUI to freeze
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


# display the results
class ResultsDialog(QDialog):
    def __init__(self, history):
        super().__init__()
        self.setWindowTitle("Perceptron Training Results")
        self.resize(850, 600)
        
        layout = QVBoxLayout(self)
        
        # Create the Tab Manager
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
        

        # populate the results table
        if 'predictions' in history and 'actuals' in history:
            preds = history['predictions']
            actuals = history['actuals']
            self.table.setRowCount(len(preds))
            
            for i in range(len(preds)):
                pred = preds[i]
                actual = actuals[i]
                isCorrect = "Yes" if pred == actual else "No"
                

                # Create the table cells
                itemIdx = QTableWidgetItem(str(i))
                itemPred = QTableWidgetItem(str(pred))
                itemActual = QTableWidgetItem(str(actual))
                itemCorrect = QTableWidgetItem(isCorrect)
            
                if isCorrect == "Yes":
                    itemCorrect.setForeground(QColor("green"))
                else:
                    itemCorrect.setForeground(QColor("red"))
                    itemPred.setForeground(QColor("red")) # Highlight the wrong guess
                    
                self.table.setItem(i, 0, itemIdx)
                self.table.setItem(i, 1, itemPred)
                self.table.setItem(i, 2, itemActual)
                self.table.setItem(i, 3, itemCorrect)
                
        tableLayout.addWidget(self.table)
        tabs.addTab(tableTab, "Test Data Predictions")
        
        # Add the tabs to the main window
        layout.addWidget(tabs)
        
        closeBtn = QPushButton("Close")
        closeBtn.clicked.connect(self.accept)
        layout.addWidget(closeBtn)



# help system
class HelpDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Help")
        self.resize(450, 350)
        
        layout = QVBoxLayout(self)
        
        helpText = QTextEdit()
        helpText.setReadOnly(True)
        
         # we used HTML to format and keep the help window clean and organized
        helpText.setHtml("""
            <h2 style='font-family: Arial;'>How to Use the neural network trainer</h2>
            <p style='font-family: Arial;'>This tool trains a single layer neural network to classify handwritten digits.</p>
            
            <h3 style='font-family: Arial;'>Hyperparameters</h3>
            <ul style='font-family: Arial;'>
                <li><b>Learning Rate:</b> Controls how aggressively the model updates its weights. A smaller value, like 0.01, is safer and smoother, while a larger value, like 0.1, is faster but might cause the model to overshoot.</li>
                <li><b>Iterations:</b> The maximum number of passes the model will make over the dataset during gradient descent.</li>
            </ul>
            
            <h3 style='font-family: Arial;'>Data Requirements</h3>
            <p style='font-family: Arial;'>Ensure the following files are in the same directory as this application:</p>
            <ul style='font-family: Arial;'>
                <li><code>optdigits_train.dat</code></li>
                <li><code>optdigits_test.dat</code></li>
            </ul>
            
            <p style='font-family: Arial;'><b>To begin:</b> Adjust your parameters and click <i>Train Perceptron</i>. Once training is complete, your fitting curves will open automatically.</p>
        """)
        layout.addWidget(helpText)
        
        closeBtn = QPushButton("Close Help")
        closeBtn.clicked.connect(self.accept)
        layout.addWidget(closeBtn)


# main GUI window
class MLExplorerGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Neural Network Trainer")
        self.resize(650, 500)
        
        centralWidget = QWidget()
        self.setCentralWidget(centralWidget)
        mainLayout = QVBoxLayout(centralWidget)
        
        # top bar for adjusting parameters
        controlsLayout = QHBoxLayout()
        
        # learning rate input box
        self.lrLabel = QLabel("Learning Rate:")
        self.lrInput = QLineEdit("0.1")
        self.lrInput.setFixedWidth(60)
        
        # iterations input box
        self.iterLabel = QLabel("Iterations:")
        self.iterInput = QLineEdit("100")
        self.iterInput.setFixedWidth(60)
        
        controlsLayout.addWidget(self.lrLabel)
        controlsLayout.addWidget(self.lrInput)
        controlsLayout.addWidget(self.iterLabel)
        controlsLayout.addWidget(self.iterInput)
        controlsLayout.addStretch()
        
        mainLayout.addLayout(controlsLayout)
        
        # train button 
        self.trainBtn = QPushButton("Train Perceptron")
        self.trainBtn.clicked.connect(self.startTraining)
        mainLayout.addWidget(self.trainBtn)
        
        # putput console messages on main GUI
        self.console = QTextEdit()
        self.console.setReadOnly(True)
        self.console.setStyleSheet("background-color: #1e1e1e; color: #d4d4d4; font-family: Consolas, monospace;")
        mainLayout.addWidget(self.console)
        
        # bottom bar with help button
        bottomLayout = QHBoxLayout()
        self.helpBtn = QPushButton("Help")
        self.helpBtn.clicked.connect(self.openHelp)
        bottomLayout.addWidget(self.helpBtn)
        bottomLayout.addStretch() 
        
        mainLayout.addLayout(bottomLayout)
        
        # setup and data loading
        self.updateConsole("Loading datasets... make sure 'optdigits_train.dat' and 'optdigits_test.dat' are in the directory.")
        try:
            self.xTrain, self.yTrain = loadData("optdigits_train.dat")
            self.xTest, self.yTest = loadData("optdigits_test.dat")
            self.updateConsole("Data loaded successfully! Ready to train.")
        except Exception as e:
            self.updateConsole(f"Error loading data: make sure the .dat files are present.")
            self.trainBtn.setEnabled(False)

    def startTraining(self):
        try:
            lr = float(self.lrInput.text())
            iters = int(self.iterInput.text())
        except ValueError:
            QMessageBox.warning(self, "Input Error", "Please enter valid numbers for learning rate and iteration number.")
            return

        self.trainBtn.setEnabled(False) 
        self.console.clear()
        
        self.worker = ModelTrainingWorker(self.xTrain, self.yTrain, self.xTest, self.yTest, lr, iters)
        self.worker.logMsg.connect(self.updateConsole)
        self.worker.errorMsg.connect(self.handleError)
        self.worker.finished.connect(self.onTrainingComplete)
        self.worker.start()
    def updateConsole(self, msg):
        self.console.append(msg)
        scrollbar = self.console.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def handleError(self, error):
        self.updateConsole(f"ERROR: {error}")
        self.trainBtn.setEnabled(True)

    def onTrainingComplete(self, history):
        self.updateConsole(f"Final Train Misclassification Error: {history['trainMisc'][-1]:.4f}")
        if 'testMisc' in history and len(history['testMisc']) > 0:
            self.updateConsole(f"Final Test Misclassification Error: {history['testMisc'][-1]:.4f}")
        
        self.updateConsole("\nGenerating fitting curves...")
        try:
            plotFittingCurves(history, "Perceptron")
            self.updateConsole("Success: Fitting curves saved to the 'graphs' folder.")
            
            self.updateConsole("Opening graph results...")
            self.graphWindow = ResultsDialog(history) 
            self.graphWindow.exec()
            
        except Exception as e:
            self.updateConsole(f"Could not display graphs: {e}")

        self.trainBtn.setEnabled(True)

    def openHelp(self):
        self.helpWindow = HelpDialog()
        self.helpWindow.exec()


# main execution block
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion") 
    window = MLExplorerGUI()
    window.show()
    sys.exit(app.exec())