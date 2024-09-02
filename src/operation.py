class Operation:
    def apply(self, *args):
        a_op_b, (a, b) = self.forward(*args)
        out = a.__class__(a_op_b, (a, b), _op=self)
        out._backward = self._build_backward_function(*args, out)
        return out

    def forward(self, *args):
        raise NotImplementedError

    def _build_backward_function(self, *args, out):
        raise NotImplementedError



