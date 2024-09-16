from activation_functions.activation import ActivationFunction

class ReLUActivation(ActivationFunction):
    def forward(self, input):
        return max(0, input.value)
    
    def _build_backward_function(self, input, out):
        def _backward():
            input.grad += (out.value > 0) * out.grad
        return _backward