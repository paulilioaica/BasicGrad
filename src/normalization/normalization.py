class NormalizationFunction:
    def __call__ (self, *args):
        if isinstance(args[0], list):
            out = []
            for batch in args[0]:
                current_batch = []
                for arg in batch:
                    value = self.forward(arg)
                    variable = arg.__class__(value, (arg,), _op=self)
                    variable._backward = self._build_backward_function(arg, variable)
                    current_batch.append(variable)
                out.append(current_batch)
        else:    
            value, variable = self.forward(*args)
            out = variable.__class__(value, args, _op=self)
            out._backward = self._build_backward_function(*args, out)
        return out

    def _build_backward_function(self, *args, out):
        raise NotImplementedError
