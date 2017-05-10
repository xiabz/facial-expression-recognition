import os
import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np

from sklearn.utils import shuffle
from time import time
from sklearn.metrics import mean_squared_error

from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import DropoutLayer
from lasagne.layers import Conv2DLayer
from lasagne.layers import MaxPool2DLayer
from lasagne.layers import get_all_params

from lasagne.nonlinearities import softmax
from lasagne.updates import adam

from nolearn.lasagne import NeuralNet
from nolearn.lasagne import TrainSplit
from nolearn.lasagne import objective
from nolearn.lasagne import PrintLayerInfo

from nolearn.lasagne.visualize import plot_loss


def loadData(path):
    '''
        Load the data from a csv file. The images are 48x48 = 2304 size vectors.

        Input is:
            -path: the path where input data is stored.

        The return values are:
            -X: A numpy array containing pixels of each picture.
            -y: A numpy array containing the labels of each picture.
    '''
    y = []
    X = []

    with open(path, 'rb') as lines:
        next(lines)  # just skip the header
        for line in lines:
            row = line.split(',')
            y.append(int(row[0]))
            X.append([int(p) for p in row[1].split()])

    X = np.array(X) / 255.0  # scale pixel values to [0, 1]
    y = np.array(y)

    X = X.astype(np.float32)  # Theano works with fp32 precision
    y = y.astype(np.int32)

    N, D = X.shape
    d = int(np.sqrt(D))
    X = X.reshape(-1, 1, d, d)

    return X, y


def design_net(X, epochs, update, learning_rate, obj_l2, eval_size, verbose):
    '''
        Design a neural net with different layers.

        Inputs are:
            -X: A numpy array containing pixels of each picture.
            -epochs: The number of epochs the net learns with each call to fit.
            -update: A good first choice as updateing rule.
            -learning_rate: Learning rate of the update parameter.
            -obj_l2: The magnitude of L2 regularization to prevent overfitting.
            -eval_size: Hold out some of the training data for validation.
            -verbose: Print some useful information.

        The return value is:
            -the designed net.
    '''
    layers = [
        # layer dealing with the input data
        (InputLayer, {'shape': (None, X.shape[1], X.shape[2], X.shape[3])}),

        # first stage of the convolutional layers
        (Conv2DLayer, {'num_filters': 48, 'filter_size': (3, 3)}),
        (Conv2DLayer, {'num_filters': 48, 'filter_size': (3, 3)}),
        (Conv2DLayer, {'num_filters': 48, 'filter_size': (3, 3)}),
        # (Conv2DLayer, {'num_filters': 48, 'filter_size': (3, 3)}),
        # (Conv2DLayer, {'num_filters': 48, 'filter_size': (3, 3)}),
        (MaxPool2DLayer, {'pool_size': (2, 2)}),

        # second stage of the convolutional layers
        (Conv2DLayer, {'num_filters': 96, 'filter_size': (3, 3)}),
        (Conv2DLayer, {'num_filters': 96, 'filter_size': (3, 3)}),
        # (Conv2DLayer, {'num_filters': 96, 'filter_size': (3, 3)}),
        (MaxPool2DLayer, {'pool_size': (2, 2)}),

        # two dense layers with a dropout layer
        (DenseLayer, {'num_units': 32}),
        (DropoutLayer, {}),
        (DenseLayer, {'num_units': 32}),

        # the output layer
        (DenseLayer, {'num_units': 7, 'nonlinearity': softmax}),
    ]

    net = NeuralNet(
        layers = layers,
        max_epochs = epochs,
        update = update,
        update_learning_rate = learning_rate,
        objective_l2 = obj_l2,
        train_split = TrainSplit(eval_size = eval_size),
        verbose = verbose,
    )

    return net


def train_net(net, X, y):
    '''
        Train the neural net. 

        Inputs are:
            -net: the neural net waiting to be trained.
            -X: a numpy array containing pixels of each picture.
            -y: a numpy array containing the labels of each picture.
    '''
    net.fit(X, y)


def print_layers_info(net):
    '''
        Print layers infomation.
        Input is:
            -net: The initialized net.
    '''
    net.initialize()
    layers_info = PrintLayerInfo()
    layers_info(net)


def cal_RMSE(net, X, y):
    '''
        Calculate the mean square error of the outcome.

        Inputs are:
            -net: the trained net
            -X: a numpy array containing pixels of each picture.
            -y: a numpy array containing the labels of each picture.

        The return value is:
            -MSE
    '''
    return np.sqrt(mean_squared_error(net.predict(X), y))


def plot_loss_pic(net):
    '''
        Plot the loss.
        Input is:
            -net: The trained net.
    '''
    train_loss = [row['train_loss'] for row in net.train_history_]
    valid_loss = [row['valid_loss'] for row in net.train_history_]
    plt.plot(train_loss, label='train loss')
    plt.plot(valid_loss, label='valid loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc='best')
    plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = 'Facial Expression Recognition by CNN.')
    parser.add_argument('data', help = 'File where input data is stored.')
    parser.add_argument('--max_epochs', default = 100, help = 'The number of epochs the net learns with each call to fit.')
    parser.add_argument('--update', default = adam, help = 'A good first choice as updateing rule.')
    parser.add_argument('--update_learning_rate', default = 0.0002, help = 'Defines how large the steps of the gradient descent updates to be.')
    parser.add_argument('--objective_l2', default = 0.0025, help = 'The magnitude of L2 regularization to prevent overfitting.')
    parser.add_argument('--eval_size', default = 0.25, help = 'Hold out some of the training data for validation.')
    parser.add_argument('--verbose', default = 2, help = 'Print some useful information.')

    args = parser.parse_args()
    data = args.data
    epochs = args.max_epochs
    update = args.update
    learning_rate = args.update_learning_rate
    obj_l2 = args.objective_l2
    eval_size = args.eval_size
    verbose = args.verbose

    path = os.path.join(os.path.expanduser('~'), data)
    X, y = loadData(path)

    # cross validation, the size of validation data depends on the argument 'eval_size'
    k = 4
    start = time()
    RMSE = []
    for i in range(k):
        X, y = shuffle(X, y)

        net = design_net(X, epochs, update, learning_rate, obj_l2, eval_size, verbose)

        sys.setrecursionlimit(10000)

        train_net(net, X, y)

        RMSE.append(cal_RMSE(net, X, y))

    now = time()-start

    print "Time: ", now

    print "RMSE: ", np.mean(RMSE)

    # used to see if the layers built are good in performance
    print_layers_info(net)

    # plot the MSE of trained data and validation data in each epoch
    plot_loss_pic(net)






