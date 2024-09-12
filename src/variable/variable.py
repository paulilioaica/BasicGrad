from ops.basic_ops import *

class Variable:
    def __init__(self, value, _children=(), _op=None, _name=None) -> None:
        self.value = value
        self.grad = 0
        self._op = _op
        self._prev = set(_children)
        self._backward = lambda: None
        self._name = _name

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
        return f"Variable(data={self.value}, grad={self.grad}), _op={self._op}, _prev={self._prev}"




