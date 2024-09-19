from .operation import Operation
import math


class AddOperation(Operation):
    def forward(self, a, b):
        return a.value + b.value

    def _build_backward_function(self, a, b, out):
        def _backward():
            a.grad += out.grad  if a.requires_grad else 0
            b.grad += out.grad  if b.requires_grad else 0
        return _backward

class MulOperation(Operation):
    def forward(self, a, b):
        return a.value * b.value

    def _build_backward_function(self, a, b, out):
        def _backward():
            a.grad += b.value * out.grad if a.requires_grad else 0
            b.grad += a.value * out.grad if b.requires_grad else 0
        return _backward
    
class SubOperation(Operation):
    def forward(self, a, b):
        return a.value - b.value
    
    def _build_backward_function(self, a, b, out):
        def _backward():
            a.grad += out.grad  if a.requires_grad else 0
            b.grad -= out.grad  if b.requires_grad else 0
        return _backward
    
class PowOperation(Operation):
    def forward(self, a, b):
        return a.value**b.value
    
    def _build_backward_function(self, a, b, out):
        def _backward():
            a.grad += b.value * (a.value ** (b.value - 1)) * out.grad  if a.requires_grad else 0
            b.grad += (a.value ** b.value) * math.log(a.value) * out.grad  if b.requires_grad else 0
        return _backward
    
class DivOperation(Operation):
    def forward(self, a, b):
        return a.value / b.value
    
    def _build_backward_function(self, a, b, out):
        def _backward():
            a.grad += (1/b.value) * out.grad  if a.requires_grad else 0
            b.grad -= (a.value / (b.value **2)) * out.grad  if b.requires_grad else 0
        return _backward
    
