'''
    TEMPLATE FOR MACHINE LEARNING HOMEWORK
    AUTHOR Eric Eaton
'''

import numpy as np
from sklearn import preprocessing
class NeuralNet:

    def __init__(self, layers, epsilon=0.12, learningRate=2.0, numEpochs=100):
        '''
        Constructor
        Arguments:
            layers - a numpy array of L-2 integers (L is # layers in the network) 
            epsilon - one half the interval around zero for setting the initial weights
            learningRate - the learning rate for backpropagation
            numEpochs - the number of epochs to run during training
        '''
        self.layers = layers
        self.epsilon = epsilon
        self.learningRate = learningRate
        self.numEpochs = numEpochs
        self._lambda = 0.0001
    def fit(self, X, y):
        '''
        Trains the model
        Arguments:
            X is a n-by-d numpy array
            y is an n-dimensional numpy array
        '''

        '''
            Theta initialization
        '''
        n,d = X.shape
        self.numOfClasses = len(set(y))

        
        binary_transformer = preprocessing.LabelBinarizer()
        binary_transformer.fit(y)
        binary_y = binary_transformer.transform(y)
        # print binary_y[0]
        # print unique_y
        layersWithIO = np.concatenate((self.layers, [self.numOfClasses]))
        layersWithIO = np.concatenate(([d], layersWithIO))

        self.thetas = {}

        for i in range(len(layersWithIO) -1):
            self.thetas[i+1] = np.random.random_sample((layersWithIO[i+1], layersWithIO[i] +1)) * (self.epsilon*2) - self.epsilon
            # print self.thetas[i+1]
        #thetas start with 1

        # print self.numOfClasses
        # print self.thetas[2].shape
                    
        '''
            backprob
        '''
        #initialize gradient
        self.gradient = {}
        



        for e in range(self.numEpochs):
            self.forwardPropogation(X,self.thetas)
            
            totalNumOfLayers = len(self.nodes)
            # print self.thetas[1]
            
            # print len(self.nodes)
            error = {}
            error[totalNumOfLayers-1] = self.nodes[totalNumOfLayers-1][:,1:] - binary_y
            # print binary_y
            # if e<5:
            #     print error[totalNumOfLayers-1]
            
            for i in reversed(range(1,totalNumOfLayers-1)): #i from totalNumOfLayers-2 to 1
                # print i
                g_prime = np.multiply(self.nodes[i][:,1:], (1- self.nodes[i][:,1:]))
                error[i] = np.multiply(np.dot(error[i+1], self.thetas[i+1][:,1:]), g_prime)
                #there is no error[0]
            # print self.thetas[1].shape,self.thetas[2].shape

            for i in range(len(self.thetas)):
                layer = i+1
                # print error[layer].T.shape,self.nodes[i].shape
                if e == 0:
                    self.gradient[layer] =  np.dot(error[layer].T, self.nodes[i])
                else:
                    self.gradient[layer] =  self.gradient[layer] + np.dot(error[layer].T, self.nodes[i])
                regularization = np.concatenate((np.zeros([self.thetas[layer].shape[0], 1]), self.thetas[layer][:,1:]), axis = 1) * self._lambda
                self.gradient[layer] = self.gradient[layer]/ n + regularization 
                self.thetas[layer] = self.thetas[layer] - self.learningRate * self.gradient[layer]
            print e

    def predict(self, X):
        '''
        Used the model to predict values for each instance in X
        Arguments:
            X is a n-by-d numpy array
        Returns:
            an n-dimensional numpy array of the predictions
        '''
        self.forwardPropogation(X, self.thetas)
        lastIndex = len(self.nodes)-1
        # print self.nodes[lastIndex][:,1:]
        predicted = np.argmax(self.nodes[lastIndex][:,1:], axis=1)
        return predicted
    
    def visualizeHiddenNodes(self, filename):
        '''
        CIS 519 ONLY - outputs a visualization of the hidden layers
        Arguments:
            filename - the filename to store the image
        '''
        
    def forwardPropogation(self, X, thetas):
        X = np.c_[np.ones(X.shape[0]), X]
        n,d = X.shape
        


        self.nodes = {}

        self.nodes[0] = X
        # print X.shape, thetas[2].shape
        self.nodes[1] = sigmoid(np.dot(X, thetas[1].T))
        # print self.nodes[1].shape
        self.nodes[1] = np.c_[np.ones(self.nodes[1].shape[0]), self.nodes[1]] #bias
        # print self.nodes[1].shape
        for i in range(2,len(self.layers)+2):
            # print self.nodes[i-1].shape, thetas[i].T.shape
            self.nodes[i] = sigmoid(np.dot(self.nodes[i-1], thetas[i].T))
            # print self.nodes[i].shape[0]
            self.nodes[i] = np.c_[np.ones(self.nodes[i].shape[0]), self.nodes[i]] #bias

       



def sigmoid(z):
    return 1.0/(1.0+ np.exp(-z)) 