import numpy as np
from random import shuffle

def svm_loss_vectorized(W, X, y, reg, norm="l2", delta=1.0):
  """
  Inputs:
  - W: A numpy array of shape (D, C) containing weights
  - X: A numpy array of shape (N, D) containing a minibatch of data
  - y: A numpy array of shape (N,) containing training labels
  - reg: (float) regularization strength
  - norm: (str) type of regularization
  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  num_train = X.shape[0]

  scores = np.dot(X, W)
  correct_class_scores = scores[range(len(y)), y].reshape(-1, 1)

  margins = np.maximum(scores - correct_class_scores + delta, 0.0)
  # We currently have num_train * delta error in our matrix, since we should
  # not add delta to the correct class indices. For all i=[0..num_train],
  # j=y[i] set margins[i,j] = 0.0
  margins[np.arange(num_train), y] = 0.0

  loss = np.sum(margins) / float(num_train)

  norm = float(norm[1:])
  loss += 1/norm * reg * np.sum(np.absolute(W) ** norm)

  grad_mask = (margins > 0).astype(int)
  grad_mask[np.arange(y.shape[0]), y] = - np.sum(grad_mask, axis=1)
  dW = np.dot(X.T, grad_mask)

  dW /= float(num_train)
  dreg = reg * np.sign(W) * np.absolute(W) ** (norm - 1)
  dW += dreg

  return loss, dW