from torch.autograd import Function
import torch.nn as nn
import numpy as np
import torch




class MyFunction2(Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, weight = None, bias=None):

        input = input.detach().numpy()
        print(input, input.shape)
        np.random.shuffle(input)
        output = input

        return torch.autograd.Variable(torch.tensor(output))

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):

        return grad_output




class LinearAlt(nn.Module):
    def __init__(self):
        super(LinearAlt, self).__init__()
    def forward(self, input):
        # See the autograd section for explanation of what happens here.
        return MyFunction2.apply(input)
