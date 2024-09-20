from activation_functions.activation import ActivationFunction
import math

class Softmax(ActivationFunction):
    def _softmax(self, x):
        e_x = [math.exp(i.value) for i in x]
        return [i / sum(e_x) for i in e_x]
    
    def forward(self, input):
        softmax_values = self._softmax(input)
        for i in range(len(input)):
            input[i].value = softmax_values[i]
        return input

    def _build_backward_function(self, input, out):
        def _backward():
            for i in range(len(input)):
                input[i].grad += out[i] * (1 - out[i]) if out.requires_grad else 0
        return _backward
    