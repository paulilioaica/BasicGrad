class Operation:
    def apply(self, *args):
        a_op_b = self.forward(*args)
        out = args[0].__class__(a_op_b, (args), _op=self)
        out._backward = self._build_backward_function(*args, out)
        return out

    def forward(self, *args):
        raise NotImplementedError

    def _build_backward_function(self, *args, out):
        raise NotImplementedError



