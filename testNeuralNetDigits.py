import numpy as np
from sklearn import datasets
from sklearn.metrics import accuracy_score
from nn import NeuralNet

filename_X = 'data/digitsX.dat'
filename_y = 'data/digitsY.dat'
X = np.loadtxt(filename_X,delimiter=',')
y = np.loadtxt(filename_y,delimiter=',')

# takes roughly 1s for each epoch
clf_NN = NeuralNet(layers = np.array([25]),learningRate = 2.0,numEpochs=450)
clf_NN.fit(X,y)
y_predict = clf_NN.predict(X)
accuracy_NN = accuracy_score(y_predict,y)

print "Accuracy: \t"+str(accuracy_NN)
