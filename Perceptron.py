import numpy as np


def sigmoid(x):
    return 1/(1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1-x)

training_inp = np.array([[0,0,1],
                         [1,1,1],
                         [1,0,1],
                         [0,1,1]])
    
training_oup = np.array([[0,1,1,0]]).T

np.random.seed(1)

synaptic_weights = 2* np.random.random((3,1)) - 1

print(' Random Starting Synaptic Weights: ')
print(synaptic_weights)

for i in range(100000):
    input_layer = training_inp
    
    outputs = sigmoid(np.dot(input_layer, synaptic_weights))
    
    error = training_oup - outputs
    adjustment = error * sigmoid_derivative(outputs)
    synaptic_weights+= np.dot(input_layer.T, adjustment)
    
print("Synaptic weights after traiing:")
print(synaptic_weights)
    
    
print("Outputs after training:" )
print(outputs)

