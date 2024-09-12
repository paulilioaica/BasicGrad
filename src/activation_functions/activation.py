class ActivationFunction:
    def __call__ (self, *args):
        if isinstance(args[0], list):
            out = []
            for batch in args[0]:
                current_batch = []
                for arg in batch:
                    value, variable = self.forward(arg)
                    current_batch.append(variable.__class__(value, (variable,) , _op=self))
                out.append(current_batch)
        else:    
            value, variable = self.forward(*args)
            out = variable.__class__(value, (variable,) , _op=self)
            out._backward = self._build_backward_function(*args, out)
        return out

    def _build_backward_function(self, *args, out):
        raise NotImplementedError
