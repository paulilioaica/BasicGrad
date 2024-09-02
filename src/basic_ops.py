from operation import Operation

class AddOperation(Operation):
    def forward(self, a, b):
        return a.value + b.value, (a, b)

    def _build_backward_function(self, a, b, out):
        def _backward():
            a.grad += out.grad
            b.grad += out.grad
        return _backward


class MulOperation(Operation):
    def forward(self, a, b):
        return a.value * b.value, (a, b)

    def _build_backward_function(self, a, b, out):
        def _backward():
            a.grad += b.value * out.grad
            b.grad += a.value * out.grad
        return _backward
    