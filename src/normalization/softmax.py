from normalization.normalization_function import NormalizationFunction
import math

class Softmax(NormalizationFunction):
    def _softmax(self, x):
        max_value = max([i.value for i in x])
        e_x = [math.exp(i.value - max_value) for i in x]
        return [i / sum(e_x) for i in e_x]
    
    def forward(self, input):
        softmax_values = self._softmax(input)
        return softmax_values

    def _build_backward_function(self, input, out):
        def _backward():
            input.grad += out.value * (1 - out.value) if out.requires_grad else 0
        return _backward
    