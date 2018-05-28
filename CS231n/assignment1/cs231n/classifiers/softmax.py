import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_samples = X.shape[0]
  num_classes = W.shape[1]
  score_row_i = np.zeros(num_classes)
  for i in range(num_samples):
    score_row_i = X[i].dot(W)
    score_row_i -= np.max(score_row_i)  # prevent numeric instability
    loss -= np.log( np.exp(score_row_i[y[i]]) / np.sum(np.exp(score_row_i)) )
    for j in range(num_classes):
      P_ij = np.exp(score_row_i[j]) / np.sum(np.exp(score_row_i))
      if(j == y[i]):
        dW[:, j] += X[i, :].T * (P_ij - 1)
      else:
        pass
        dW[:, j] += X[i, :].T * P_ij

  loss /= num_samples
  dW /= num_samples

  loss += reg * np.sum(W * W)
  dW += 2 * reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_samples = X.shape[0]
  num_classes = W.shape[1]
  score = X.dot(W)                                # N by C
  prob = np.exp(score)
  prob_de = np.sum(np.exp(score), axis=1)
  prob_de = np.reshape(prob_de, (num_samples,1))
  prob = prob / prob_de                           # N by C, Mat of P
  loss =  -1 * np.sum( np.log( prob[range(num_samples),y] ) )
  prob[range(num_samples),y] -= 1                 # j == y[i] , dw = (P_ij - 1)Xi
  dW = X.T.dot(prob)                              # (D by N)(N by C) = D by C

  loss /= num_samples
  dW /= num_samples

  loss += reg * np.sum(W * W)
  dW += 2 * reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

