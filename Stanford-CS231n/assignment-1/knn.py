# importing required packages
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle
from pprint import pprint
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

class KNearestNeighbors():
    "KNN classifier"

    def __init__(self):
        pass

    def train(self, X, y):
        """
        The training in KNN is simply memorizing the given
        data along with its correct label
        params{
              X : array of input features
              y : array of labels
        }
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        """
        predicting the labels by first finding the distance
        matrix for each test instance with all training instances.
        This is L2 distance. The test instance is given the label of
        the training instance whose L2 distance is least.
        """
        num_test = X.shape[0] # stores the number of test cases
        y_pred = np.zeros(num_test, dtype=self.y_train.dtype)
        # looping over test cases
        for i in range(num_test):
            print(i)
            dist_mat = np.sum(np.sum((self.X_train - X[i,:])**2, axis=1), axis=1)
            dist_mat = np.sum(dist_mat, axis=1)
            min_index = np.argmin(dist_mat) # index of smallest distance
            y_pred[i] = self.y_train[min_index]
        return y_pred


# dataset loading and required functions
def unpickle(file):
    '''Load byte data from file'''
    with open(file, 'rb') as f:
        data = pickle.load(f, encoding='latin-1')
        return data

def load_cifar10_data(data_dir):
    '''Return train_data, train_labels, test_data, test_labels
    The shape of data is 32 x 32 x3'''
    train_data = None
    train_labels = []
    # unpickling train dataset
    for i in range(1, 6):
        data_dic = unpickle(data_dir + "/data_batch_{}".format(i))
        if i == 1:
            train_data = data_dic['data']
        else:
            train_data = np.vstack((train_data, data_dic['data']))
        train_labels += data_dic['labels']
    # unpickling test dataset
    test_data_dic = unpickle(data_dir + "/test_batch")
    test_data = test_data_dic['data']
    test_labels = test_data_dic['labels']
    # reshaping train dataset
    train_data = train_data.reshape((len(train_data), 3, 32, 32))
    train_data = np.rollaxis(train_data, 1, 4)
    train_labels = np.array(train_labels)
    # reshaping test dataset
    test_data = test_data.reshape((len(test_data), 3, 32, 32))
    test_data = np.rollaxis(test_data, 1, 4)
    test_labels = np.array(test_labels)
    # return train and test dataset and labels
    return train_data, train_labels, test_data, test_labels

data_dir = os.path.join(os.path.expanduser('~'), 'Deep-Learning',
                        'data', 'cifar-10-batches-py')
train_data, train_labels, test_data, test_labels = load_cifar10_data(data_dir)
# printing dataset shape
print(train_data.shape)
print(train_labels.shape)
print(test_data.shape)
print(test_labels.shape)
# In order to check where the data shows an image correctly
plt.imshow(train_data[5])
plt.show()
# model training
model = KNearestNeighbors()
print('training..')
model.train(train_data, train_labels)
print('Done.')
print('Predicting..')
y_pred = model.predict(test_data)
print('Done.')
print('Accuracy:', accuracy_score(test_labels, y_pred))
# training on inbuilt classifier
model_2 = KNeighborsClassifier()
print('Training..')
model_2.fit(np.sum(np.sum(train_data, axis=1), axis=1), train_labels)
print('Done.')
print('Predicting labels..')
y_pred_2 = model_2.predict(np.sum(np.sum(test_data, axis=1), axis=1))
print('Done.')
print('Accuracy:', accuracy_score(test_labels, y_pred_2))
