from torch.autograd import Function
import torch.nn as nn
import numpy as np
import torch
from SPController import SPController

class MyFunction2(Function):
    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, weight = None, bias=None, spc = None):

        #actual forward pipeline:
        #input comes in already encrypted
        #weights are masked
        #pytorch does masked input @ masked weights
        #that is sent to sgx to check if modified output is correct after unmodifying it
        #remask output and send it back


        #saves masked input and masked weights for fast backwards gpu pass
        ctx.save_for_backward(input, weight, bias)
        
        rand_mask = torch.ones(input.shape)

        input = input + rand_mask
        #weight = weight + weight_rand_mask

        output = input.mm(weight.t())
        
        rand_mask = rand_mask.mm(weight.t())
        output = output - rand_mask #- weight_rand_mask

        # masked_ouput = masked_input @ masked_weights
        # decrypted_output = maksed_output - random_matrix @ true_weights + (not so) extreme foiling

        sgx_freevalt()

        

        #lalala sgx stuff
        # input = input.detach().numpy()
        # print(input, input.shape)
        # np.random.shuffle(input)
        # w = weight.detach().numpy()

        # ret = spc.query_enclave([input, w, input @ w])
        # if ret is None:
        #     print("Verification failed!")
        # else:
        #     print("Response: " + str(ret))
  
        # output = ret
  
        #return masked output
        # return torch.autograd.Variable(torch.tensor(output)) 

        return output
    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):

        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        #rewrite all these functions to page to SGX and send back error with random noise
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        return grad_output, grad_weight, grad_bias, None 




class LinearAlt(nn.Module):
    def __init__(self, input_features, output_features, bias=True):
        super(LinearAlt, self).__init__()

        # self.spc = SPController()
        # self.spc.start(verbose=True)

        self.input_features = input_features
        self.output_features = output_features

        #technically masked weights
        self.weight = nn.Parameter(torch.Tensor(output_features, input_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(output_features))
        else:
            self.register_parameter('bias', None)

        self.weight.data.uniform_(-0.1, 0.1)
        if bias is not None:
            self.bias.data.uniform_(-0.1, 0.1)


    def forward(self, input):
        # See the autograd section for explanation of what happens here.
        return MyFunction2.apply(input, self.weight, self.bias, self.spc)