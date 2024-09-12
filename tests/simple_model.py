from nn.linear_layer import LinearLayer
from activation_functions.activation import ReLUActivation

class SimpleModel:
    def __init__(self, input_size, hidden_size, output_size):
        self.layer1 = LinearLayer(input_size, hidden_size)
        self.layer2 = LinearLayer(hidden_size, output_size)
        self.relu = ReLUActivation()

    def __call__(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

    def parameters(self):
        return self.layer1.parameters() + self.layer2.parameters()