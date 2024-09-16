from nn.linear_layer import LinearLayer
from activation_functions.relu import ReLUActivation
from nn.module import Module

class SimpleModel(Module):
    def __init__(self, input_size, hidden_size, output_size):
        self.layer1 = LinearLayer(input_size, hidden_size)
        self.layer2 = LinearLayer(hidden_size, output_size)
        self.relu = ReLUActivation()

    def __call__(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x
