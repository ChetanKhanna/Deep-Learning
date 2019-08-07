# import packages
import numpy as np


class LinearClassifier():

    def __init__(self):
        self.weights = None

    def train(self, X, y, learning_rate=1e-3, reg=1e-5, num_iters=100,
              batch_size=200, verbose=False):
        num_train, dim = X.shape
        num_labels = np.max(y) + 1
        # initializing weights randomly if not already done
        if self.weights == None:
            self.weights = 0.001 * np.random.randn(dim, num_labels)
        loss_hist = [] # variable to store history for all iterations
        for iter_ in range(num_iters):
            X_batch = None
            y_batch = None
            # making batches
            indices = np.random.choice(num_train, batch_size, replacement=True)
            X_batch = X[indices, :]
            y_batch = y[indices,]
            # getting loss and gradient from loss function
            loss, grad = self.loss(X_batch, y_batch, reg)
            loss_hist.append(loss)
            # updating weights
            self.weights = self.weights - (learning_rate * grad)
            # printing output if verbose == True
            if verbose and iter_ % 100 == 0:
                print('iteration %d / %d: loss %f' %(iter_, num_iters, loss))
        return loss_hist

    def predict(self, X):
        # initializing y_pred
        y_pred = np.zeros(X.shape[0])
        # getting weight matrix
        f = X.dot(self.weights)
        y_pred = np.argmax(f, axis=1)
        return y_pred

    def loss(self, X_batch, y_batch, reg):
        '''
        This is an abstract class meant to be overrider by subclass
        of LinearClassifier class.
        '''
        pass


class LinearSVM(LinearClassifier):

    def loss(self, X_batch, y_batch, reg):
        # getting the score matrix
        f = X_batch.dot(self.weights)
        correct_class_scores = np.choose(y_batch.T, f.T).reshape(X_batch.shape[0], 1)
        margin = np.maximum(0, f - correct_class_scores + 1)
        margin[range(margin.shape[0]), y_batch.T] = 0
        loss = margin.sum() / X_batch.shape[0]
        # regularizing
        loss += reg * np.sum(self.weights * self.weights)
        # calculating the new gradients
        