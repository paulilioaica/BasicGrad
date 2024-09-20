class SGDOptimizer:
    def __init__(self, parameters, learning_rate=0.01):
        self.parameters = parameters
        self.learning_rate = learning_rate

    def step(self):
        for param in self.parameters:
            param.value -= min(self.learning_rate * param.grad, 1.0)

    def zero_grad(self):
        for param in self.parameters:
            param.grad = 0