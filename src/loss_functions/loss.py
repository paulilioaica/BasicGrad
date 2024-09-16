class LossFunction:
    def __init__(self) -> None:
        raise NotImplementedError
    
    def __call__(self, *args):
        raise NotImplementedError
    
    def _build_backward_function(self, *args, out):
        raise NotImplementedError
    