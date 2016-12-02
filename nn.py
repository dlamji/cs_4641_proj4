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

    def fit(self, X, y):
        '''
        Trains the model
        Arguments:
            X is a n-by-d numpy array
            y is an n-dimensional numpy array
        '''
        n,inputSize = X.shape
        self.layers = [self.layers[0]] + [10]
        hidlength = len(self.layers)
        ybinaryCst = preprocessing.LabelBinarizer()
        ybinary = ybinaryCst.fit_transform(y)
        # initialize weights
        # 1d: dict{layer}     2d: to    3d: from
        # print ybinary
        # print hidlength
        weightsmatrix = {}
        weightsmatrix[0] = np.array([[np.random.uniform(-self.epsilon,self.epsilon) for _ in range(inputSize+1)] for _ in range(self.layers[0])])
        for l in range(hidlength-1):
            weightsmatrix[l+1] = np.array([[np.random.uniform(-self.epsilon,self.epsilon) for _ in range(self.layers[l]+1)] for _ in range(self.layers[l+1])])
        # print weightsmatrix[0].shape,weightsmatrix[1].shape
        for epoch in range(self.numEpochs):
            # initialize delta matrix
            deltamatrix = {}
            deltamatrix[0] = np.zeros((self.layers[0],inputSize+1))      
            lastindex = 0
            for l in range(hidlength-1):
                lastindex = l+1
                deltamatrix[l+1] = np.zeros((self.layers[l+1],self.layers[l]+1))
            # print deltamatrix[0].shape,deltamatrix[1].shape
            # learn from the data
            # a = self.forwardPropagation(X,weightsmatrix)
            for i in range(len(X)):
                a = self.forwardPropagation(X[i],weightsmatrix)
                # print a[0].shape,a[1].shape,a[2].shape
                # print lastindex
                err = {}
                # print y[i]
                L = lastindex + 1
                err[L] = a[L] - ybinary[i]
                preverr = err[L]
                # print preverr
                for j in range(hidlength,1,-1):
                    gz = a[j-1]*(1-a[j-1])
                    err[j-1] = weightsmatrix[j-1][:,1:].T.dot(preverr)*gz
                    preverr = err[j-1]
                # print err
                # print np.insert(a[1],0,1).T,err[0]
                for l in range(hidlength):
                    # print err[l+1].T.shape,np.insert(a[l],0,1).shape,deltamatrix[l].shape
                    deltamatrix[l] += np.multiply([err[l+1]],np.array([np.insert(a[l],0,1)]).T).T
                    # for i in range(len(deltamatrix[l])):

            for l in range(hidlength):
                deltamatrix[l] /= n
                weightsmatrix[l] -= self.learningRate*deltamatrix[l]
            # print epoch

        self.weightsmatrix = weightsmatrix



    def predict(self, X):
        '''
        Used the model to predict values for each instance in X
        Arguments:
            X is a n-by-d numpy array
        Returns:
            an n-dimensional numpy array of the predictions
        '''
        n,inputSize = X.shape
        ypred = np.zeros((n))
        for i in range(n):
            tmp = self.forwardPropagation(X[i],self.weightsmatrix)
            # print tmp[len(self.layers)]
            ypred[i] = np.argmax(tmp[len(self.layers)])
            # print ypred[i]
        return ypred
    
    def visualizeHiddenNodes(self, filename):
        '''
        CIS 519 ONLY - outputs a visualization of the hidden layers
        Arguments:
            filename - the filename to store the image
        '''
        

    def sigmoid(self,matrix):
        return 1./(1+np.exp(-matrix))


    def forwardPropagation(self,firstinput,weightsmatrix):
        # weightsmatrix 3d
        # 1d: dict   2d: to   3d: from
        out = {}
        out[0] = firstinput
        prevout = firstinput
        # print weightsmatrix
        for l in range(len(self.layers)):
            if(type(prevout[0]) != np.float64):
                # print np.ones(conlen).shape
                conlen = len(prevout[0])
            else:
                conlen = 1
            # print weightsmatrix
            # print weightsmatrix[l].shape
            tmpout = weightsmatrix[l].dot(np.concatenate((np.ones(conlen),prevout),axis=0))
            prevout = self.sigmoid(tmpout)
            out[l+1] = prevout
        # out is a dict
        # 1d: layer   2d: a
        # print out
        return out



