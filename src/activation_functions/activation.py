class ActivationFunction:

    def __call__ (self, *args):
        value, variable = self.forward(*args)
        out = variable.__class__(value, (variable,) , _op=self)
        out._backward = self._build_backward_function(*args, out)
        return out

    def _build_backward_function(self, *args, out):
        raise NotImplementedError
