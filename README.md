# Neural-Network
This repository contains Python code for a simple implementation of a Neural Network and a Perceptron using the Numpy library.

## NeuralNetwork.py

### Description
The `NeuralNetwork.py` file contains the implementation of a basic neural network. The network is initialized with random synaptic weights and can be trained using training data. It utilizes the sigmoid activation function for forward propagation and backpropagation to adjust weights during training.

### Usage
To create and train the neural network, you can run the code as follows:

```python
if __name__ == "__main__":
    neural_network = NeuralNetwork()
    print("Random synaptic weights:")
    print(neural_network.synaptic_weights) 
    
    # Define your training data and expected outputs
    training_inputs = np.array([[0,0,1],
                                [1,1,1],
                                [1,0,1],
                                [0,1,1]])
    training_outputs = np.array([[0,1,1,0]]).T
    
    # Train the neural network with your data and desired iterations
    neural_network.train(training_inputs, training_outputs, 10000)
    print("Synaptic weights after training:")
    print(neural_network.synaptic_weights)
    
    # Input new data to get predictions
    B =  str(input("Input 1: "))
    A =  str(input("Input 2: "))
    C =  str(input("Input 3: "))    
    
    print("New input data:", B, A, C)
    print("Output data:")
    print(neural_network.thx(np.array([B, A, C]))
