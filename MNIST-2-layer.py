# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 12:46:28 2017

@author: Asmita
"""

import numpy as np
import random

import time
start_time = time.time()

def softmax(z):
    z = z.T
    e_z = np.exp(z - np.max(z, axis = 0))
    op = (e_z / np.sum(e_z, axis = 0))
    return np.array(op).T

def relu(activation):
    return np.array([[max(0, c) for c in i] for i in activation])
    
def deriv_sigmoid(x):
    return x * (1.0 - x)

def deriv_relu(z):
    z[z<=0] = 0
    z[z>0] = 1
    return z

def oneHotEncoding(data):
    result = np.array(np.zeros((10), float))
    result[data] = 1
    return result

#Load data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/home/asmita/data")

X_train = mnist.train.images
y_train = mnist.train.labels.astype("int")
y_train = (np.expand_dims(y_train, axis = 1))
y_train_one_hot = []
for yi in y_train:
    y_train_one_hot.append(oneHotEncoding(yi))
y_train_one_hot = (np.array(y_train_one_hot))
#X_train = X_train[:20,:]
#y_train_one_hot = y_train_one_hot[:20,:]
print("X_train.shape=",X_train.shape,"  y_train.shape=",y_train.shape)

np.random.seed(1)
n_inputs = X_train.shape[1]
n_hidden = np.array([300]) #array of neurons in each layer
n_layers = len(n_hidden)
n_outputs = 10
epoch = 30
eta = 0.01

#Build weight vectors
weights_hidden =list()
bias_hidden = list()
z = n_inputs
for i in n_hidden:
    arr = np.array(2 * np.random.random((z , i)) -1)
    arr2 = np.array(2 * np.random.random((1, i)) -1)
    weights_hidden.append(arr)
    bias_hidden.append(arr2)
    z = i

weights_output = np.array(2 * np.random.random((n_hidden[-1], n_outputs)) -1)
bias_output = np.array(2 * np.random.random((1, n_outputs)) -1)

print ("-----Training", n_layers+1, "-layer Neural Net without Momentum-----")
for e in range(epoch):
    #Forward propagation
    activation = list()
    activation.append(X_train)
    for i in range(n_layers):
        weights_hidden[i] = np.array(weights_hidden[i])
        a = (np.dot(weights_hidden[i].T, activation[-1].T)) + bias_hidden[i].T
        ReLU = relu(a) 
        activation.append(ReLU.T)
    activation_2 = np.dot(weights_output.T, activation[-1].T) + bias_output.T
    output = softmax(activation_2.T)
    output_class = np.expand_dims(np.argmax(output, axis = 1), axis = 1)
#Error calculation
    error_calc =0.0
    for o, yi in zip(output_class,y_train):
        if o != yi:
            error_calc += 1
    if (e % 1) ==0:
        print ("Epoch: ", e, "Error: ", error_calc/len(y_train))
#Back propagation
    error = output - y_train_one_hot
    output_derivative = error 
    #Error for first hidden layer w.r.t. output
    hidden_error = weights_output.dot(output_derivative.T)
    weights_output -= eta * activation[-1].T.dot(output_derivative)
    bias_output = bias_output - eta * output_derivative
    #Updating hidden weights
    for i in range(n_layers):
        hidden_derivative = hidden_error * deriv_relu(activation[-i-1].T)
        hidden_error = weights_hidden[-i-1].dot(hidden_derivative)
        bias_hidden[-i-1] = bias_hidden[-i-1] - eta * hidden_derivative.T #np.sum(, axis =0, keepdims =True)
        weights_hidden[-i-1] -= eta * ((activation[-i-2]).T.dot(hidden_derivative.T))
        #print (bias_output.shape, bias_hidden[-i-1].shape)

for i in range(n_layers):
    bias_hidden[i] = np.sum(bias_hidden[i], axis=0, keepdims=True)
bias_output = np.sum(bias_output, axis =0, keepdims=True)
        
#Predicting on test data
X_test = mnist.test.images
y_test = mnist.test.labels.astype("int")
y_test = (np.expand_dims(y_test, axis = 1))
y_test_one_hot = []
for yi in y_test:
    y_test_one_hot.append(oneHotEncoding(yi))
y_test_one_hot = (np.array(y_test_one_hot))
activation = list()
activation.append(X_test)
for i in range(n_layers):
    weights_hidden[i] = np.array(weights_hidden[i])
    a = (np.dot(weights_hidden[i].T, activation[-1].T)) + bias_hidden[i].T
    ReLU = relu(a) 
    activation.append(ReLU.T)
activation_2 = np.dot(weights_output.T, activation[-1].T) + bias_output.T
output = softmax(activation_2.T)
output_class = np.expand_dims(np.argmax(output, axis = 1), axis = 1)
error_test =0.0
for o, yi in zip(output_class,y_train):
    if o != yi:
        error_test += 1
print ("Error for test Data: ", error_test/len(y_test))

print("--- %s seconds ---" % (time.time() - start_time))