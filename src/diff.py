class Variable:
    def __init__(self, data, _children=(), _op=None) -> None:
        self.data = data
        self.grad = 0
        self._op = _op
        self._prev = set(_children)
        self._backward = lambda: None

    def __add__(self, other):
        return self._apply(AddOperation(), other)

    def __mul__(self, other):
        return self._apply(MulOperation(), other)

    def _apply(self, operation, other):
        other = self._ensure_variable(other)
        out = operation.apply(self, other)
        return out

    @staticmethod
    def _ensure_variable(other):
        if not isinstance(other, Variable):
            other = Variable(other)
        return other

    def backward(self):
        self.grad = 1
        topological_order = self._build_topological_order()

        for node in reversed(topological_order):
            node._backward()

    def _build_topological_order(self):
        order = []
        visited = set()

        def dfs(node):
            if node not in visited:
                visited.add(node)
                for child in node._prev:
                    dfs(child)
                order.append(node)

        dfs(self)
        return order

    def __repr__(self):
        return f"Variable(data={self.data}, grad={self.grad})"


class Operation:
    def apply(self, *args):
        out = self.forward(*args)
        out._backward = self._build_backward_function(*args, out)
        return out

    def forward(self, *args):
        raise NotImplementedError

    def _build_backward_function(self, *args, out):
        raise NotImplementedError


class AddOperation(Operation):
    def forward(self, a, b):
        return Variable(a.data + b.data, (a, b))

    def _build_backward_function(self, a, b, out):
        def _backward():
            a.grad += out.grad
            b.grad += out.grad
        return _backward


class MulOperation(Operation):
    def forward(self, a, b):
        return Variable(a.data * b.data, (a, b))

    def _build_backward_function(self, a, b, out):
        def _backward():
            a.grad += b.data * out.grad
            b.grad += a.data * out.grad
        return _backward
    

class ActivationFunction:
    def apply(self, *args):
        out = self.forward(*args)
        out._backward = self._build_backward_function(*args, out)
        return out

    def forward(self, *args):
        raise NotImplementedError

    def _build_backward_function(self, *args, out):
        raise NotImplementedError
    

class ReLUActivation(ActivationFunction):
    def forward(self, input):
        out = Variable(max(0, input.data), (input,))
        return out
    
    def _build_backward_function(self, input, out):
        # Derivative of ReLU: 1 if input > 0 else 0
        def _backward():
            input.grad += (out.data > 0) * out.grad
        return _backward