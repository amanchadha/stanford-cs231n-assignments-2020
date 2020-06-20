from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_classes = W.shape[1]
    num_train = X.shape[0]
    for i in range(num_train):
        scores = X[i].dot(W) # scores.shape is N x C

        # shift values for 'scores' for numeric reasons (over-flow cautious)
        scores -= scores.max()

        probs = np.exp(scores)/np.sum(np.exp(scores))

        loss += -np.log(probs[y[i]])

        # since dL(i)/df(k) = p(k) - 1 (if k = y[i]), where f is a vector of scores for the given example
        # i is the training sample and k is the class
        dscores = probs.reshape(1,-1)
        dscores[:, y[i]] -= 1

        # since scores = X.dot(W), iget dW by multiplying X.T and dscores
        # W is D x C so dW should also match those dimensions
        # X.T x dscores = (D x 1) x (1 x C) = D x C
        dW += np.dot(X[i].T.reshape(X[i].shape[0], 1), dscores)

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)

    # Add regularization loss to the gradient
    dW += 2 * reg * W    

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_classes = W.shape[1]
    num_train = X.shape[0]

    scores = X.dot(W) # scores.shape is N x C

    # shift values for 'scores' for numeric reasons (over-flow cautious)
    scores -= scores.max(axis = 1, keepdims = True)

    probs = np.exp(scores)/np.sum(np.exp(scores), axis = 1, keepdims = True)

    loss = -np.log(probs[np.arange(num_train), y])

    # loss is a single number
    loss = np.sum(loss)

    # since dL(i)/df(k) = p(k) - 1 (if k = y[i]), where f is a vector of scores for the given example
    # i is the training sample and k is the class
    dscores = probs.reshape(num_train, -1)
    dscores[np.arange(num_train), y] -= 1

    # since scores = X.dot(W), iget dW by multiplying X.T and dscores
    # W is D x C so dW should also match those dimensions
    # X.T x dscores = (D x 1) x (1 x C) = D x C
    dW = np.dot(X.T.reshape(X.shape[1], num_train), dscores)

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)

    # Add regularization loss to the gradient
    dW += 2 * reg * W    
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
