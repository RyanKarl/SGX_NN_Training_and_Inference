from torch.autograd import Function
import torch.nn as nn
import numpy as np
import torch
<<<<<<< HEAD
from SPController import SPController
=======

>>>>>>> b35002db8924417f854818912f1783de378b3a06



class MyFunction2(Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
<<<<<<< HEAD
    def forward(ctx, input, weight = None, bias=None, spc = None):

        input = input.detach().numpy()
        w = weight.detach().numpy()
        
        #print(input, input.shape)
        np.random.shuffle(input)
        
        #Debug
        a = [ [ 1.0, 1.0 ], [ 1.0, 1.0 ] ] 
        b = [ [ 1.0, 1.0 ], [ 1.0, 1.0 ] ] 
        c = [ [ 2.0, 2.0 ], [ 2.0, 2.0 ] ] 
        arrs = [np.asarray(x) for x in [a, b, c]]


        #Do IPC
        ret = spc.query_enclave(arrs)        
        if ret is None:
            print("Verification failed!")
        else:  
            print("Response: " + str(ret))
        
        output = input
        
=======
    def forward(ctx, input, weight = None, bias=None):

        input = input.detach().numpy()
        print(input, input.shape)
        np.random.shuffle(input)
        output = input
>>>>>>> b35002db8924417f854818912f1783de378b3a06

        return torch.autograd.Variable(torch.tensor(output))

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):

<<<<<<< HEAD
        return grad_output, None, None, None 
=======
        return grad_output
>>>>>>> b35002db8924417f854818912f1783de378b3a06




class LinearAlt(nn.Module):
    def __init__(self):
        super(LinearAlt, self).__init__()
<<<<<<< HEAD
    
        self.spc = SPController()
        self.spc.start(verbose=False)
    

    
    def forward(self, input, weight):
        # See the autograd section for explanation of what happens here.
        return MyFunction2.apply(input, weight, None, self.spc)
=======
    def forward(self, input):
        # See the autograd section for explanation of what happens here.
        return MyFunction2.apply(input)
>>>>>>> b35002db8924417f854818912f1783de378b3a06
