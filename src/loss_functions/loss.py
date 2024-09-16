class LossFunction:
    def __init__(self) -> None:
        raise NotImplementedError
    
    def __call__(self, *args):
        raise NotImplementedError
    
    def backward(self):
        raise NotImplementedError