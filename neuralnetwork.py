# BACKPROPAGATION ALGORITHM GB430
# Using:
# One input layer with two nodes: X1 and X2
# One hidden layer with two nodes
# One output layer with one node.
# Sigmoid neuron as node

# Run the program by running this class

import numpy as np
#import pandas as pd
#from pandas import DataFrame


# Sigmoid function (used in forward propagation)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Derivative of Sigmoid function (used in backward propagation)
def sigmoid_derivative(x):
    return x * (1 - x)


# Desired inputs of XOR-gate
xor_input = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# Desired output of XOR-gate
xor_output = np.array([[0], [1], [1], [0]])

# Can adjust number of epochs
# One epoch: when the entire dataset is passed forward and backward through the neural network
# The more epochs the closer we get to the desired XOR-gate output (but takes more time)
epochs = 20000
learning_rate = 0.1
inputLayerNeurons, hiddenLayerNeurons, outputLayerNeurons = 2, 2, 1

# Random weights and biases
hidden_weights = np.random.uniform(size=(inputLayerNeurons, hiddenLayerNeurons))
hidden_bias = np.random.uniform(size=(1, hiddenLayerNeurons))
out_weights = np.random.uniform(size=(hiddenLayerNeurons, outputLayerNeurons))
out_bias = np.random.uniform(size=(1, outputLayerNeurons))

print("Start backpropagation hidden weights:   \n", end='')
print(*hidden_weights)
print("Start backpropagation output weights:   \n", end='')
print(*out_weights)


# Training algorithm
for _ in range(epochs):
    # Forward Propagation
    hidden_layer_activation = np.dot(xor_input, hidden_weights)
    hidden_layer_activation += hidden_bias
    hidden_layer_output = sigmoid(hidden_layer_activation)

    out_layer_activation = np.dot(hidden_layer_output, out_weights)
    out_layer_activation += out_bias
    predicted_out = sigmoid(out_layer_activation)

    # Backpropagation
    error = xor_output - predicted_out
    b_predicted_output = error * sigmoid_derivative(predicted_out)

    error_hidden_layer = b_predicted_output.dot(out_weights.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)

    # Updating weights and biases
    out_weights += hidden_layer_output.T.dot(b_predicted_output) * learning_rate
    out_bias += np.sum(b_predicted_output, axis=0, keepdims=True) * learning_rate
    hidden_weights += xor_input.T.dot(d_hidden_layer) * learning_rate
    hidden_weights += xor_input.T.dot(d_hidden_layer) * learning_rate
    hidden_bias += np.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate


# Function for printing out results in a XOR-matrix
def draw_xor_results():
    output_format = np.around(predicted_out, decimals=0)
    output_format = output_format.flatten()
    print("Formatted and rounded answer from the neural network: ")
    print('\u2554' + "                                           " + u'\u2557')
    print('\u2551' + "     X1     " + "     X2     " + "           Y       " + '\u2551')
    print('\u2551' + "                                           " + u'\u2551')
    print('\u2551' + "     0      " + "     0               ", output_format[0], "     "+'\u2551')
    print('\u2551' + "                                           " + u'\u2551')
    print('\u2551' + "     0      " + "     1               ", output_format[1], "     "+'\u2551')
    print('\u2551' + "                                           " + '\u2551')
    print('\u2551' + "     1      " + "     0               ", output_format[2], "     "+'\u2551')
    print('\u2551' + "                                           " + u'\u2551')
    print('\u2551' + "     1      " + "     1               ", output_format[3], "     "+'\u2551')
    print('\u255a' + "                                           " + '\u255d')


print("\nEnd backpropagation hidden weights: \n", end='')
print(*hidden_weights)
print("End backpropagation output weights: \n", end='')
print(*out_weights)

print("\nNon-formatted output from neural network: \n", end='')
print(*predicted_out)

draw_xor_results()

