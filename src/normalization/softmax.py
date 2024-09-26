from normalization.normalization_function import NormalizationFunction
import math

class Softmax(NormalizationFunction):
    def __init__(self, temperature=3, clip_value=50):
        self.temperature = temperature
        self.clip_value = clip_value

    def _softmax(self, x):
        max_value = max([i.value for i in x])
        
        clipped_logits = [(i.value - max_value) / self.temperature for i in x]
        clipped_logits = [min(self.clip_value, max(-self.clip_value, logit)) for logit in clipped_logits]
        
        log_sum_exp = math.log(sum([math.exp(logit) for logit in clipped_logits]))
        softmax_values = [math.exp(logit - log_sum_exp) for logit in clipped_logits]

        return softmax_values

    def forward(self, input):
        softmax_values = self._softmax(input)
        return softmax_values

    def _build_backward_function(self, input, out):
        def _backward():
            if out.requires_grad:
                input.grad += out.value * (1 - out.value) * out.grad / self.temperature
        return _backward
