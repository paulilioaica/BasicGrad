from variable.variable import Variable
from nn.module import Module
import random

class LinearLayer(Module):
    def __init__(self, input_size, output_size, dropout=False):
        self.input_size = input_size
        self.output_size = output_size
        self.weight = Variable([[random.random() for _ in range(self.input_size)] for _ in range(self.output_size)])
        self.bias = Variable([random.random() for _ in range(self.output_size)])

    def __call__(self, input):
        batch_size = len(input)
        output = Variable([[[0 for _ in range(self.output_size)] for _ in range(len(input[batch]))] for batch in range(batch_size)])

        for batch in range(batch_size):
            for i in range(self.output_size):
                for j in range(len(input[batch])):
                    sum_value = Variable(value=0, _name=f'sum_value_{batch}_{i}_{j}')
                    
                    for k in range(self.input_size):
                        sum_value += self.weight[i][k] * input[batch][j][k]
                    
                    output[batch][j][i] = sum_value + self.bias[i]

        return output
    
    def parameters(self):
        params = []
        for row in self.weight:
            params.extend(row)
        params.extend(self.bias)
        return params
