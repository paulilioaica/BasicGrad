from activation_function import ActivationFunction

class ReLUActivation(ActivationFunction):
    def forward(self, input):
        return max(0, input.value), input
    
    def _build_backward_function(self, input, out):
        # Derivative of ReLU: 1 if input > 0 else 0
        def _backward():
            input.grad += (out.value > 0) * out.grad
        return _backward