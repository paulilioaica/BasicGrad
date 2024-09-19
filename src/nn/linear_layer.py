from variable.variable import Variable
from nn.module import Module
import random


class LinearLayer(Module):
    def __init__(self, input_size, output_size, dropout=False):
        self.input_size = input_size
        self.output_size = output_size
        self.weight = Variable([[random.random() for i in range(self.input_size)] for j in range(self.output_size)])
        self.bias = Variable([random.random() for i in range(self.output_size)])

    def __call__(self, input):
        batch_size = len(input)

        output = Variable([[0 for i in range(self.output_size)] for batch in range(batch_size)])

        for batch in range(batch_size):
            for i in range(self.output_size):

                sum_value = Variable(value=0, _name=f'sum_value_{batch}_{i}')
                
                for j in range(self.input_size):
                    sum_value += self.weight[i][j] * input[batch][j]
                
                output[batch][i] = sum_value + self.bias[i]

        return output
    
    def parameters(self):
        params = []
        for row in self.weight:
            params.extend(row)
        params.extend(self.bias)
        return params