#!/usr/bin/env python

import numpy as np
import random

from q1_softmax import softmax
from q2_sigmoid import sigmoid, sigmoid_grad
from q2_gradcheck import gradcheck_naive


def forward_backward_prop(data, labels, params, dimensions):
    """
    Forward and backward propagation for a two-layer sigmoidal network

    Compute the forward propagation and for the cross entropy cost,
    and backward propagation for the gradients for all parameters.

    Arguments:
    data -- M x Dx matrix, where each row is a training example.
    labels -- M x Dy matrix, where each row is a one-hot vector.
    params -- Model parameters, these are unpacked for you.
    dimensions -- A tuple of input dimension, number of hidden units
                  and output dimension
    """

    ### Unpack network parameters (do not modify)
    ofs = 0
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])
    W1 = np.reshape(params[ofs:ofs+ Dx * H], (Dx, H))
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H))
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))

    ### YOUR CODE HERE: forward propagation
    z1 = np.dot(data, W1) + b1
    h = sigmoid(z1)
    z2 = np.dot(h, W2) + b2
    yhat = softmax(z2)
    cost = np.sum(-labels*np.log(yhat))
    ### END YOUR CODE

    ### YOUR CODE HERE: backward propagation
#     delta1 = yhat - labels   #partial(CE)/partial(z2)
#     gradW2 = []
#     for i in range(data.shape[0]):
#         delta2 = np.zeros((Dy,H,Dy),dtype=np.float32) #partial(z2)/partial(W2)
#         for j in range(Dy):
#             delta2[j,:,j] = h[i]
#         delta2.shape = (Dy,H*Dy)
#         gradW2.append(np.dot(delta1[i], delta2))
#     gradW2 = np.sum(gradW2,axis=0)
#     gradb2 = np.sum(delta1,axis=0)
    
#     delta3 = np.dot(delta1, W2.T)*sigmoid_grad(h) #partial(CE)/partial(z1)
#     gradW1 = []
#     for i in range(data.shape[0]):
#         delta4 = np.zeros((H,Dx,H),dtype=np.float32) #partial(z1)/partial(W1)
#         for j in range(H):
#             delta4[j,:,j] = data[i]
#         delta4.shape = (H,H*Dx)
#         gradW1.append(np.dot(delta3[i], delta4))
#     gradW1 = np.sum(gradW1,axis=0)
#     gradb1 = np.sum(delta3,axis=0)
    ### END YOUR CODE
    
    M = data.shape[0]
    delta1 = yhat - labels   #partial(CE)/partial(z2)
    #partial(z2)/partial(W2) change to use h.reshape((M,H,1)) in left
    gradW2 = h.reshape((M,H,1))*delta1.reshape((M,1,Dy))
    gradW2 = np.sum(gradW2,axis=0)
    gradb2 = np.sum(delta1,axis=0)
    
    delta3 = np.dot(delta1, W2.T)*sigmoid_grad(h) #partial(CE)/partial(z1)
    gradW1 = data.reshape((M,Dx,1))*delta3.reshape((M,1,H))
    gradW1 = np.sum(gradW1,axis=0)
    gradb1 = np.sum(delta3,axis=0)

    ### Stack gradients (do not modify)
    grad = np.concatenate((gradW1.flatten(), gradb1.flatten(),
        gradW2.flatten(), gradb2.flatten()))

    return cost, grad


def sanity_check():
    """
    Set up fake data and parameters for the neural network, and test using
    gradcheck.
    """
    print "Running sanity check..."

    N = 20
    dimensions = [10, 5, 10]
    data = np.random.randn(N, dimensions[0])   # each row will be a datum
    labels = np.zeros((N, dimensions[2]))
    for i in xrange(N):
        labels[i, random.randint(0,dimensions[2]-1)] = 1
    params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (
        dimensions[1] + 1) * dimensions[2], )

    gradcheck_naive(lambda params:
        forward_backward_prop(data, labels, params, dimensions), params)


def your_sanity_checks():
    """
    Use this space add any additional sanity checks by running:
        python q2_neural.py
    This function will not be called by the autograder, nor will
    your additional tests be graded.
    """
    print "Running your sanity checks..."
    ### YOUR CODE HERE
    raise NotImplementedError
    ### END YOUR CODE


if __name__ == "__main__":
    sanity_check()
#     your_sanity_checks()
