class ActivationFunction:
    def __call__(self, *args):
        if isinstance(args[0], list):
            out = []
            for batch in args[0]:
                current_batch = []
                for entry in batch:
                    current_entry = []
                    for arg in entry:
                        value = self.forward(arg)
                        variable = arg.__class__(value, (arg,), _op=self)
                        variable._backward = self._build_backward_function(arg, variable)
                        current_entry.append(variable)
                    current_batch.append(current_entry)
                out.append(current_batch)
        else:
            value = self.forward(*args)
            out = args[0].__class__(value, args, _op=self)
            out._backward = self._build_backward_function(*args, out)

        return out

    def _build_backward_function(self, *args, out):
        raise NotImplementedError

    def forward(self, *args):
        raise NotImplementedError
