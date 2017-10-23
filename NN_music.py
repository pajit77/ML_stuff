# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 14:43:02 2017

@author: Pankaj
"""
'''
REFERENCES 

https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw
In sum, for most problems, one could probably get decent performance (even without a second optimization step) by setting the hidden layer configuration using just two rules: 
(i) number of hidden layers equals one; and 
(ii) the number of neurons in that layer is the mean of the neurons in the input and output layers

Here -
Hidden layers = 1
Inputs = 5 
and Outputs = 1
Mean = 3 (neurons in hidden layer)

https://stackoverflow.com/questions/3985619/how-to-calculate-a-logistic-sigmoid-function-in-python
'''
'''
0

0 0 0 B 

0 0 0 0 0  B  
''' 


import re
import operator
from random import seed
seed(77)
from random import random
import numpy as np 
import math

#WIPE out after making dynamic
import os 
os.chdir('C:\\Users\\Pankaj\\Documents\\10601\\hw6')
#os.getcwd()

train_file = 'music_train.csv'
dev_file = 'music_dev.csv'
trainlabel_file = 'music_train_keys.txt'

devlabel_file = 'music_dev_keys.txt'


# Get music file

def tokenizeDoc(cur_doc):
    cur_doc = cur_doc.lower()
    return re.findall('\w+',cur_doc)


def get_data(fname):    
    data = []
    counter = 0
    with open(fname,'r') as fhandle:
        for line in fhandle:
            if counter ==0:
                counter+=1
                pass
            else:
                line = tokenizeDoc(line)
                #print line
                for i in [4,5]:
                    if line[i] =='no':
                        line[i]=0
                    else:
                        line[i]=1
                line = [int(i) for i in line]
                data.append(line)
    return data

def get_labels(fname):
    with open(fname,'r') as fhandle:
        data = []
        for line in fhandle:
            line = tokenizeDoc(line)
            if line[0] == 'no':
                data.append(0)
            else:
                data.append(1)
    return data

def normalizer(data):
    data2=[]
    for datapoint in data:
        dif = map(operator.sub, datapoint, min_list)
        x_scaled = [0,0,0,0,0,0]
        for i in range(0,6,1):
            try:
                x_scaled[i] = float(dif[i])/maxdif[i]
            except:
                x_scaled[i] = 0
        del x_scaled[1]
        data2.append(x_scaled)
    return data2

def layers(neurons,connections):
    layer = []
    #Loop for number of neurons
    for i in range(0,neurons,1):
        weight_dict = {}
        neuron_weights = []
        #Loop for number of connections (input neurons + bias)
        for i in range (0,connections,1):
            neuron_weights.append(random())
        weight_dict['w']  = neuron_weights
        layer.append(weight_dict)
    return layer

def safe_sigmoid(x):
    if x >= 0:
        z = math.exp(-x)
        return 1 / (1 + z)
    else:
        z = math.exp(x)
        return z / (1 + z)

# Feed Forward the inputs
def feed_forward(topology, Xi):
    rec_input = list(Xi)
    rec_input.append(1)
    for layer in topology:
        sig_output = []
        for neuron in layer:
            neuron['sig_out'] = safe_sigmoid(np.dot(rec_input,neuron['w']))
            sig_output.append(neuron['sig_out'])
        rec_input = sig_output
	return rec_input

def activate(weights, inputs):
	activation = weights[-1]
	for i in range(len(weights)-1):
		activation += weights[i] * inputs[i]
	return activation
 
# Transfer neuron activation
def transfer(activation):
	return 1.0 / (1.0 + math.exp(-activation))

def forward_propagate(network, row):
	inputs = row
	for layer in network:
		new_inputs = []
		for neuron in layer:
			activation = activate(neuron['w'], inputs)
			neuron['output'] = transfer(activation)
			new_inputs.append(neuron['output'])
		inputs = new_inputs
	return inputs

Xi=train[0]
out = forward_propagate(topology,Xi)

# Update network weights with error
def update_weights(network, row, l_rate):
	for i in range(len(network)):
		inputs = row[:-1]
		if i != 0:
			inputs = [neuron['output'] for neuron in network[i - 1]]
		for neuron in network[i]:
			for j in range(len(inputs)):
				neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
			neuron['weights'][-1] += l_rate * neuron['delta']
 
# Train a network for a fixed number of epochs
def train_network(network, train, l_rate, n_epoch, n_outputs):
	for epoch in range(n_epoch):
		sum_error = 0
		for row in train:
			outputs = feed_forward(network, row)
			expected = [0 for i in range(n_outputs)]
			expected[row[-1]] = 1
			sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
			back_prop(network, expected)
			update_weights(network, row, l_rate)
		print sum_error



train=[]
dev=[]
train = get_data(train_file)
dev = get_data(dev_file)
train_truth = get_labels(trainlabel_file)
abcde = zip(*train)
max_list = map(max,abcde)
min_list = map(min,abcde)
maxdif = map(operator.sub, max_list, min_list)

train = normalizer(train)
dev = normalizer(dev)
 
# Create Network Topology
topology = []
topology.append(layers(3,6))
topology.append(layers(1,7))

Xi = train[0]
for Xi in train:
    out = feed_forward(topology, Xi)


def transfer_derivative(output):
	return output * (1.0 - output)
 
# Backpropagate error and store in neurons
def back_prop(network, truth):
	for i in layer_counts:
		layer = topology[i]
		errors = list()
		if i != len(topology)-1:
			for j in range(len(layer)):
				error = 0.0
				for neuron in network[i + 1]:
					error += (neuron['weights'][j] * neuron['delta'])
				errors.append(error)
		else:
			for j in range(len(layer)):
				neuron = layer[j]
				errors.append(truth[j] - neuron['sig_out'])
		for j in range(len(layer)):
			neuron = layer[j]
			neuron['delta'] = errors[j] * transfer_derivative(neuron['sig_out'])
 

truth = train_truth[0]

backward_propagate_error(network, expected)


layer_numbers = [1,0]
layer_num=1
for layer_num in layer_numbers:
    layer = topology[layer_num]
    output_layer_flag = 1
    num_neurons = len(layer)
    errors = list()
    if output_layer_flag == 1 :
        output_layer_flag+=1
        neuron = layer[0]
        errors.append(truth - neuron['output'])
        neuron['delta'] = errors[0] * transfer_derivative(neuron['output'])
    else :
        for j in range(num_neurons):
            err = float(0.0)
            for neuron in topology[layer_num+1]:
                print neuron['weights'][j]
        

