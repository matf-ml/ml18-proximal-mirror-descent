from __future__ import print_function

import numpy as np
from linear_svm import *


class LinearClassifier(object):

  def __init__(self):
    self.W = None

  def train(self, X, y, learning_rate=1e-3, reg=1e-5, norm="l2", num_iters=100,
            batch_size=200, verbose=False):
    """
    Inputs:
    - X: A numpy array of shape (N, D) containing training data
    - y: A numpy array of shape (N,) containing training labels
    - learning_rate: (float) learning rate for optimization
    - reg: (float) regularization strength
    - norm: (str) type of regularization
    - num_iters: (integer) number of steps to take when optimizing
    - batch_size: (integer) number of training examples to use at each step
    - verbose: (boolean) If true, print progress during optimization
    Outputs:
    A list containing the value of the loss function at each training iteration
    """
    num_train, dim = X.shape
    num_classes = np.max(y) + 1
    if self.W is None:
      self.W = np.random.randn(dim, num_classes)

    loss_history = []
    for it in range(num_iters):
      batch_idx = np.random.choice(num_train, batch_size)
      X_batch = X[batch_idx, :]
      y_batch = y[batch_idx]

      loss, grad = self.loss(X_batch, y_batch, reg, norm)
      loss_history.append(loss)

      self.W -= learning_rate * grad

      if verbose and it % 100 == 0:
        print('iteration %d / %d: loss %f' % (it, num_iters, loss))

    return loss_history

  def predict(self, X):
    """
    Inputs:
    - X: A numpy array of shape (N, D) containing training data
    Returns:
    - y_pred: Predicted labels for the data in X
    """
    h = np.dot(X, self.W)
    y_pred = np.argmax(h, axis=1)

    return y_pred

  def loss(self, X_batch, y_batch, reg, norm):
    """
    Subclasses will override this
    Inputs:
    - X_batch: A numpy array of shape (N, D) containing a minibatch of N
      data points
    - y_batch: A numpy array of shape (N,) containing labels for the minibatch
    - reg: (float) regularization strength
    - norm: (str) type of regularization
    Returns: A tuple containing:
    - loss as a single float
    - gradient with respect to self.W; an array of the same shape as W
    """
    pass


class LinearSVM(LinearClassifier):
  """ A subclass that uses the Multiclass SVM loss function """

  def loss(self, X_batch, y_batch, reg, norm):
    return svm_loss_vectorized(self.W, X_batch, y_batch, reg, norm)