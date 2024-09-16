class Module:
    def __init__(self) -> None:
        raise NotImplementedError
    
    def __call__(self, *args):
        raise NotImplementedError

    def parameters(self):
            param_list = []
            for attr in self.__dict__.values():
                if isinstance(attr, Module):
                    param_list.extend(attr.parameters())
                elif hasattr(attr, 'parameters'):
                    param_list.extend(attr.parameters())
            return param_list