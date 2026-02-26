from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):

    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    for i in range(num_train):
        scores = X[i].dot(W)

        # compute the probabilities in numerically stable way
        scores -= np.max(scores)
        p = np.exp(scores)
        p /= p.sum()  # normalize
        logp = np.log(p)

        loss -= logp[y[i]]  # negative log probability is the loss
        p[y[i]] -= 1
        dW += X[i][:,None] * p[None,:]
    # normalized hinge loss plus regularization
    loss = loss / num_train + reg * np.sum(W * W)
    dW = dW/num_train + 2*reg*W
    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
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
    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    scores = X @ W # (N,D)*(D,C) -> (N,C)

    # compute the probabilities in numerically stable way
    scores -= np.max(scores, axis = 1)[:,None] # (N)
    p = np.exp(scores) # (N,C)
    p /= p.sum(axis = 1)[:,None]  # normalize
    logp = np.log(p)
    arr = np.arange(num_train)
    loss -= logp[arr,y].sum()  # negative log probability is the loss
    p[arr,y] -= 1
    dW += X.T @ p
    loss = loss / num_train + reg * np.sum(W * W)
    dW = dW/num_train + 2*reg*W


    return loss, dW

