from loss_functions.loss import LossFunction
import math

class CategoricalCrossEntropy(LossFunction):
    def __init__(self) -> None:
        self.loss = 0
        self.y_pred = []
        self.y_true = []

    def __call__(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true
        batch_size = len(y_pred)
        batch_loss = 0

        for batch in range(batch_size):
            for i in range(len(y_pred[batch])):
                loss = -y_true[batch][i].value * math.log(y_pred[batch][i].value)
                batch_loss += loss

        self.loss = batch_loss / batch_size
        return self

    def backward(self):
        batch_size = len(self.y_pred)
        for batch in range(batch_size):
            for i in range(len(self.y_pred[batch])):
                self.y_pred[batch][i].grad += -self.y_true[batch][i].value / self.y_pred[batch][i].value / batch_size
                self.y_pred[batch][i].backward()

    def __repr__(self):
        return f"Loss: {self.loss}"