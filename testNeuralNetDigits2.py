from numpy import loadtxt, ones, zeros, where
import numpy as np
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import sys, traceback

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from nn import NeuralNet
# load the data set
filename = 'data/digitsX.dat'
data = loadtxt(filename, delimiter=',')
X = data[:,:]
filename = 'data/digitsY.dat'
data1 = loadtxt(filename, delimiter=',')
y = data1
layers = np.array([25])

clf = NeuralNet(layers = layers, learningRate = 2.0, numEpochs = 10)
clf.fit(X,y)
predicted = clf.predict(X)
# print predicted
print np.mean(predicted == y)