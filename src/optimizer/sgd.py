class SGDOptimizer:
    def __init__(self, parameters, learning_rate=0.01):
        self.parameters = parameters
        self.learning_rate = learning_rate

    def step(self):
        for param in self.parameters:
            # Update each parameter based on its gradient
            param.value -= self.learning_rate * param.grad

    def zero_grad(self):
        for param in self.parameters:
            param.grad = 0