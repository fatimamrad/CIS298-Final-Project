Hello, this project is meant to simplify the process of training neural networks. It allows users to change hyper parameters of their model without having to dig through hundreds of lines of code to find and change each value. 

DEPENDENCIES:
In order to run the program, you must install numpy, matplotlib, and PyQt6. To do so simply type the following commands into the terminal:
- pip install numpy
- pip install matplotlib
- pip install PyQt6

After installing, ensure your data files are in the same directory as your project files then simply run the program using the run button in your IDE or by typing "python mainGUI.py"

Once the program runs and the GUI is visible, simply change the hyper parameters by typing in appropriate values into the learning rate and iterations text boxes, then press train perceptron and the results will be displayed for you shorty after.
----------------------------------------------------------------------------------------------------------------------------------
Team evaluation:
Loren Danaoui- 5
Marcus Gharbieh- 5
Fatima Mrad- 5

Time spent each commit:
Loren Danaoui - 25-30 hours.. The functions I did: plotFittingCurves, sigmoidDerivavtive, proxyError, initialWeights, updateConsole, handleError, onTrainingComplete, openHelp, .. Class: ResultsDialog 

Marcus Gharbieh - 20-25 hours.. The functions I did: sigmoidDerivative, proxyError, initializeWeights, fit, .. Class: ModelTrainingWorker. 

Fatima Mrad - 