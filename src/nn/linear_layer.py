from variable.variable import Variable
import random


class LinearLayer:
    def __init__(self, input_size, output_size, dropout=False):
        self.input_size = input_size
        self.output_size = output_size
        self.weight = [[Variable(value=random.random()) for i in range(self.input_size)] 
                       for j in range(self.output_size)]

    def __call__(self, input):
        batch_size = len(input)

        output = [[[0] for i in range(self.output_size)] 
                       for j in range(batch_size)]

        for batch in range(batch_size):
            for i in range(self.output_size):

                current_value = Variable(value=0)
                
                for j in range(self.input_size):
                  current_value += self.weight[i][j]
                
                output[batch][i] = current_value

        return output
