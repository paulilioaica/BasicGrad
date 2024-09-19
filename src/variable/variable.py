from ops.basic_ops import *

class Variable:
    def __init__(self, value, _children=(), _op=None, _name=None, requires_grad=True):
        self.value = value
        self.grad = 0
        self._op = _op
        self._prev = set(_children)
        self._backward = lambda: None
        self._name = _name
        self.requires_grad = requires_grad

    def __new__(cls, value, _children=(), _op=None, _name=None, requires_grad=True):
        if isinstance(value, (int, float)):
            instance = super().__new__(cls)
            return instance
        elif isinstance(value, list):
            # go through the list and wrap all the ints or floats in Variables
            converted_list = Variable._convert(value, _children, _op, _name, requires_grad)
            return converted_list
        else:
            raise TypeError("Unsupported type for Variable, must be int, float or list")
        
    def __add__(self, other):
        return self._apply(AddOperation(), other)

    def __mul__(self, other):
        return self._apply(MulOperation(), other)

    def _apply(self, operation, other):
        out = operation.apply(self, other)
        return out


    @staticmethod
    def _convert(value, _children, _op, _name, requires_grad):
        if isinstance(value, (int, float)):
            return Variable(value, _children, _op, _name, requires_grad)
        elif isinstance(value, list):
            return [Variable._convert(item, _children, _op, _name, requires_grad) for item in value]
        else:
            raise TypeError("Unsupported type for Variable")

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
        return f"Variable(data={self.value}, grad={self.grad}, _op={self._op}, name={self._name}, requires_grad={self.requires_grad}, _prev={self._prev})"