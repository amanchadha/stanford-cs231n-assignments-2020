from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

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
    dW = np.zeros(W.shape) # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1 # note delta = 1
            if margin > 0:
                loss += margin

                # for incorrect classes (j != y[i]), gradient for class j is x * I(margin > 0) 
                # the transpose on the extracted input sample X[i] transforms it into a column vector
                # for dw[:, j]
                dW[:, j] += X[i].T
                
                # for correct class (j = y[i]), gradient for class j is -x * I(margin > 0) 
                # the transpose on the extracted input sample X[i] transforms it into a column vector
                # for dw[:, j]
                dW[:, y[i]] += -X[i].T
            
    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather than first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Add regularization loss to the gradient
    dW += 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    return loss, dW



def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.

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
    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    scores = X.dot(W) #dims: N x C
    correct_class_score = scores[:, y]

    # convert correct_class_scores to a column vector
    correct_class_scores = scores[np.arange(num_train), y].reshape(-1, 1)
    
    # hinge loss
    margin = np.maximum(0, scores - correct_class_scores + 1) # note delta = 1

    # set loss of the correct class to be 0
    margin[np.arange(num_train), y] = 0 
    
    # loss is summed over all examples and classes since loss is one number
    loss = np.sum(margin)
    
    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # compute input mask
    valid_margin_mask = np.zeros(margin.shape) # margin.shape is N x C
    valid_margin_mask[margin > 0] = 1 # if margin is positive, set a positive mask

    # subtract in correct class (-s_y) aka valid_margin_count
    valid_margin_mask[np.arange(num_train), y] = -np.sum(valid_margin_mask, axis=1)

    # since scores = X.dot(W), and valid_margin_mask is a function of margin,
    # which is a function of scores, we get dW by multiplying X.T and valid_margin_mask
    # W is D x C so dW should also match those dimensions
    # X.T x valid_margin_mask = (D x N) x (N x C) = D x C
    dW = X.T.dot(valid_margin_mask) # X.T.shape is D x N and valid_margin_mask.shape is N x C

    # Right now the gradient is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    dW /= num_train

    # Add regularization loss to the gradient
    dW += 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
