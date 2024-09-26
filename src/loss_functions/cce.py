import math
from loss_functions.loss import LossFunction

class CategoricalCrossEntropy(LossFunction):
    def __init__(self) -> None:
        self.loss = 0
        self.y_pred = []
        self.y_true = []
        self.epsilon = 1e-12

    def __call__(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true
        batch_size = len(y_pred) 
        batch_loss = 0

        for batch in range(batch_size):
            sample_size = len(y_pred[batch])
            for sample in range(sample_size):
                for i in range(len(y_pred[batch][sample])):
                    pred_value = max(y_pred[batch][sample][i].value, self.epsilon)
                    loss = y_true[batch][sample][i].value * math.log(pred_value)
                    batch_loss += loss

        self.loss = - batch_loss / batch_size 
        return self

    def backward(self):
        batch_size = len(self.y_pred)
        for batch in range(batch_size):
            sample_size = len(self.y_pred[batch])
            for sample in range(sample_size):
                for i in range(len(self.y_pred[batch][sample])):
                    self.y_pred[batch][sample][i].grad = (self.y_pred[batch][sample][i].value - self.y_true[batch][sample][i].value)
                    self.y_pred[batch][sample][i].backward()

    def __repr__(self):
        return f"Loss: {self.loss}"
