class NormalizationFunction:
    def __call__(self, *args):
        if isinstance(args[0], list):
            out = []
            for batch in args[0]:
                current_batch = []
                for sample in batch:
                    current_sample = []
                    normalized_sample = self.forward(sample)
                    for i, output in enumerate(sample):
                        variable = output.__class__(normalized_sample[i], (output,), _op=self)
                        variable._backward = self._build_backward_function(output, variable)
                        current_sample.append(variable)
                    current_batch.append(current_sample)
                out.append(current_batch)
        else:
            normalized_value = self.forward(*args)
            out = args[0].__class__(normalized_value, args, _op=self)
            out._backward = self._build_backward_function(*args, out)

        return out

    def forward(self, output):
        raise NotImplementedError

    def _build_backward_function(self, *args, out):
        raise NotImplementedError
