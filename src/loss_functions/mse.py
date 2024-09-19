from loss_functions.loss import LossFunction


class MSELoss(LossFunction):
    def __init__(self) -> None:
        self.loss = 0
        self.diff_sum = []
        self.y_pred = []

    def __call__(self, y_pred, y_true):
        self.y_pred = y_pred
        batch_size = len(y_pred)
        batch_loss = 0
        self.diff_sum = []

        for batch in range(batch_size):
            differences = [self.y_pred[batch][i].value - y_true[batch][i].value for i in range(len(y_pred[batch]))]
            self.diff_sum.append(differences)
            loss = sum([i**2 for i in differences])
            batch_loss += loss

        self.loss = batch_loss / batch_size
        return self

    def backward(self):
        batch_size = len(self.y_pred)
        for batch in range(batch_size):
            for i in range(len(self.y_pred[batch])):
                self.y_pred[batch][i].grad += 2 * self.diff_sum[batch][i] / batch_size
                self.y_pred[batch][i].backward()
        
    def __repr__(self):
        return f"Loss: {self.loss}"
