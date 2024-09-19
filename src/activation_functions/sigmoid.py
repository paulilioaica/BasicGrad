from activation_functions.activation import ActivationFunction
import math

class SigmoidActivation(ActivationFunction):
    def _simgoid(self, x):
        return 1 / (1 + math.exp(-x))
    
    def forward(self, input):
        return self._simgoid(input.value)
    
    def _build_backward_function(self, input, out):
        def _backward():
            input.grad += out.grad * out.value * (1 - out.value) if out.requires_grad else 0
        return _backward