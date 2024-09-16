class Module:
    def __init__(self) -> None:
        raise NotImplementedError
    
    def __call__(self, *args):
        raise NotImplementedError

    def parameters(self):
            for attr in self.__dict__.values():
                if isinstance(attr, Module):  
                    yield from attr.parameters()
                elif hasattr(attr, 'parameters'):
                    yield from attr.parameters()