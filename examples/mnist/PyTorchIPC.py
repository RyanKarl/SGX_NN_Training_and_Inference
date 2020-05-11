from torch.autograd import Function, Variable
import torch.nn as nn
import torch
import numpy as np
import torch
import torch.nn.functional as F
from SPController import SPController
import time
input_ = input

torch.set_default_dtype(torch.float32)
super_mega_mask = torch.rand(10000,10000, device = "cuda:0") * 1

def my_cross_entropy(x, y):
    #x = x - 1
    log_prob = -1.0 * F.log_softmax(x, 1)
    loss = log_prob.gather(1, y.unsqueeze(1))
    loss = loss.mean()
    return loss


def softmax(x):
    x -= torch.max(x, axis = 1)[0][:, None]
    out = torch.exp(x).T / torch.sum(torch.exp(x), axis = 1)[:, None].T
    return out.T


# tensor([[ 0.0819,  0.0680,  0.0301],
#         [ 0.0680,  0.0301, -0.0728],
#         [ 0.0301, -0.0728, -0.3525]])


def softmax_der(s):
    return torch.diag_embed(s) - (s[:,:,None] @ s[:,:,None].permute([0,2,1]).contiguous())


def SGXFL(input, weight):
    rand_mask = super_mega_mask[0:input.shape[0], 0:input.shape[1]]
    weight_rand_mask = super_mega_mask[0:weight.shape[0], 0:weight.shape[1]]

    # print(input.shape)

    a = input - rand_mask
    b = weight - weight_rand_mask

    c = a @ b.t()
    # print(c)

    rand_mask = torch.ones(c.shape, device = "cuda:0")
    # print(c.shape)
    out = softmax(c) #+ rand_mask

    try:
        return out
    except:
        return out


def SGXBL(grad_output, input, weight):
    
    # print(grad_output.shape)
    rand_mask = super_mega_mask[0:input.shape[0], 0:input.shape[1]]
    weight_rand_mask = super_mega_mask[0:weight.shape[0], 0:weight.shape[1]]
    grad_rand_mask = torch.ones(grad_output.shape, device = "cuda:0")

    a = input - rand_mask
    b = weight - weight_rand_mask
    try:
        c = grad_output.clone() # - grad_rand_mask
    except:
        c = grad_output.clone()

    # print(c.shape, a.shape, b.shape)

    # c = c * (1-torch.tanh(a @ b.t())**2)
    # c = c * (torch.sigmoid(a @ b.t()) * (1 - torch.sigmoid(a @ b.t())))
    pre_shape = c.shape
    c = c[:,None,:]
    c = c @ softmax_der(softmax(a @ b.t()))
    # c[(a @ b.t()) < 0] = 0
    # print(c)
    # a = 1-torch.tanh(a)**2
    # print(c)
    c = c.reshape(pre_shape)
    d = c @ b

    e = c.t().mm(a)
    # print(e)

    rand_mask = torch.ones(d.shape, device = "cuda:0")
    weight_rand_mask = torch.ones(e.shape, device = "cuda:0")

    return d + super_mega_mask[0:d.shape[0], 0:d.shape[1]], e + weight_rand_mask

class MyFunction3(Function):
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
        
        # rand_mask = torch.ones(input.shape)

        # input = input + rand_mask
        #weight = weight + weight_rand_mask

        # output = input.mm(weight.t())
        
        # rand_mask = rand_mask.mm(weight.t())
        # output = output - rand_mask #- weight_rand_mask

        # masked_ouput = masked_input @ masked_weights
        # decrypted_output = maksed_output - random_matrix @ true_weights + (not so) extreme foiling

        output = SGXFL(input, weight)

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

        
        # print(grad_output)
        # time.sleep(1)


        #rewrite all these functions to page to SGX and send back error with random noise
        # if ctx.needs_input_grad[0]:
        #     grad_input = grad_output.mm(weight)
        # if ctx.needs_input_grad[1]:
        #     grad_weight = grad_output.t().mm(input)
        # if bias is not None and ctx.needs_input_grad[2]:
        #     grad_bias = grad_output.sum(0)
        a,b = SGXBL(grad_output, input, weight)
        return a, b, grad_bias, None 




class LinearAltLast(nn.Module):
    def __init__(self, input_features, output_features, bias=True):
        super(LinearAltLast, self).__init__()

        # self.spc = SPController()
        # self.spc.start(verbose=True)
        self.spc = None

        self.input_features = input_features
        self.output_features = output_features

        #technically masked weights
        self.weight = nn.Parameter(torch.Tensor(output_features, input_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(output_features))
        else:
            self.register_parameter('bias', None)

        self.weight.data.uniform_(-0.1, 0.1)
        self.weight.data += super_mega_mask[0:self.weight.shape[0], 0:self.weight.shape[1]].to("cpu")
        if bias is not None:
            self.bias.data.uniform_(-0.1, 0.1)


    def forward(self, input):
        # See the autograd section for explanation of what happens here.
        return MyFunction3.apply(input, self.weight, self.bias, self.spc)


def SGXF(input, weight):
    rand_mask = super_mega_mask[0:input.shape[0], 0:input.shape[1]]
    

    weight_rand_mask = super_mega_mask[0:weight.shape[0], 0:weight.shape[1]]



    # print(input.shape)

    #128,784
    a = input #- rand_mask

    #500,784
    b = weight #- weight_rand_mask

    # print(a.shape,b.shape)
    c = a @ b.t()
    # print(c)

    #128, 500
    diff = rand_mask @ b.t()

    #128, 500
    diff2 = a @ weight_rand_mask.t()

    diff3 = rand_mask @ weight_rand_mask.t()
    
    # print((a - rand_mask) @ (b - weight_rand_mask).t())
    # print((c - diff - diff2 + diff3) - ((a - rand_mask) @ (b - weight_rand_mask).t()) )

    rand_mask = super_mega_mask[0:c.shape[0], 0:c.shape[1]]
    out = (torch.tanh(c - diff - diff2 + diff3) + rand_mask)

    

    try:
        return out
    except:
        return out


def SGXB(grad_output, input, weight, output):
    
    # print(grad_output.shape)
    rand_mask = super_mega_mask[0:input.shape[0], 0:input.shape[1]]
    weight_rand_mask = super_mega_mask[0:weight.shape[0], 0:weight.shape[1]]
    


    #128,784
    a = input #- rand_mask

    #500,784
    b = weight #- weight_rand_mask



    
    
    c = grad_output.clone() #- 1

    grad_rand_mask = super_mega_mask[0:c.shape[0], 0:c.shape[1]]

    # print(c.shape, a.shape, b.shape)

    #tanh(c) + mask
    g = output.clone()


    f = a @ b.t()
    diff = rand_mask @ b.t()
    diff2 = a @ weight_rand_mask.t()
    diff3 = rand_mask @ weight_rand_mask.t()
    #128,500
    c = c * (1-torch.tanh(g  - diff - diff2 + diff3)**2)

    # c = c * (torch.sigmoid(a @ b.t()) * (1 - torch.sigmoid(a @ b.t())))

    # ahh = (torch.relu(a @ b.t()) / (a @ b.t())) * 0

    # ahh[torch.isnan(ahh)] = 0

    # c *= ahh
    # try:
    #     c[(a @ b.t()) <= 0] = 0
    # except:
    #     c[(a @ b.t()) <= 0] = 0
    # a = 1-torch.tanh(a)**2


    #128, 500 x 500, 784
    d = c @ b

    diffa = (1-torch.tanh(grad_rand_mask - diff - diff2 + diff3)**2) @ b
    diffb = c @ weight_rand_mask

    diffc = (1-torch.tanh(grad_rand_mask- diff - diff2 + diff3)**2) @ weight_rand_mask

    d = d - diffa - diffb + diffc

    #500, 128 x 128, 784
    e = c.t().mm(a)

    diffa = c.t() @ rand_mask
    diffb = (1-torch.tanh(grad_rand_mask - diff - diff2 + diff3)**2).t() @ a 

    diffc = (1-torch.tanh(grad_rand_mask - diff - diff2 + diff3)**2).t() @ rand_mask

    e = e - diffa - diffb + diffc    

    
    # print(e)

    rand_mask = torch.ones(d.shape, device = "cuda:0")
    weight_rand_mask = torch.ones(e.shape, device = "cuda:0")

    return d + super_mega_mask[0:d.shape[0], 0:d.shape[1]], e + weight_rand_mask 

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
        if input.shape[1] == 784:
            a = input + super_mega_mask[0:input.shape[0], 0:input.shape[1]]

        #saves masked input and masked weights for fast backwards gpu pass
        
        
        # rand_mask = torch.ones(input.shape)

        # input = input + rand_mask
        #weight = weight + weight_rand_mask

        # output = input.mm(weight.t())
        
        # rand_mask = rand_mask.mm(weight.t())
        # output = output - rand_mask #- weight_rand_mask

        # masked_ouput = masked_input @ masked_weights
        # decrypted_output = maksed_output - random_matrix @ true_weights + (not so) extreme foiling
        output = SGXF(input, weight)

        ctx.save_for_backward(input, weight, bias, output)

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

        input, weight, bias, output = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        
        # print(grad_output)
        # time.sleep(1)


        #rewrite all these functions to page to SGX and send back error with random noise
        # if ctx.needs_input_grad[0]:
        #     grad_input = grad_output.mm(weight)
        # if ctx.needs_input_grad[1]:
        #     grad_weight = grad_output.t().mm(input)
        # if bias is not None and ctx.needs_input_grad[2]:
        #     grad_bias = grad_output.sum(0)
        a,b = SGXB(grad_output, input, weight, output)
        return a, b, grad_bias, None




class LinearAlt(nn.Module):
    def __init__(self, input_features, output_features, bias=True):
        super(LinearAlt, self).__init__()

        # self.spc = SPController()
        # self.spc.start(verbose=True)
        self.spc = None

        self.input_features = input_features
        self.output_features = output_features

        #technically masked weights
        self.weight = nn.Parameter(torch.Tensor(output_features, input_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(output_features))
        else:
            self.register_parameter('bias', None)

        self.weight.data.uniform_(-0.1, 0.1)
        self.weight.data += super_mega_mask[0:self.weight.shape[0], 0:self.weight.shape[1]].to("cpu")
        if bias is not None:
            self.bias.data.uniform_(-0.1, 0.1)


    def forward(self, input):
        # See the autograd section for explanation of what happens here.
        return MyFunction2.apply(input, self.weight, self.bias, self.spc)


class ConvAlt(nn.Module):
    def __init__(self, input_channels, output_channels, kernel, stride, bias=True):
        super(ConvAlt, self).__init__()

        # self.spc = SPController()
        # self.spc.start(verbose=True)
        self.spc = None

        self.kernel = kernel
        self.stride = stride

        self.input_features = input_channels
        self.output_features = output_channels

        #technically masked weights
        self.weight = nn.Parameter(torch.Tensor(output_channels, input_channels * (kernel ** 2)))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(output_channels))
        else:
            self.register_parameter('bias', None)

        self.weight.data.uniform_(-0.1, 0.1)
        self.weight.data += super_mega_mask[0:self.weight.shape[0], 0:self.weight.shape[1]]
    
        if bias is not None:
            self.bias.data.uniform_(-0.1, 0.1)


    def forward(self, input):
        # See the autograd section for explanation of what happens here.
        # print(input.shape)
        patches = extract_image_patches(input, self.kernel, self.stride)
        
        # print(patches.size())
        return MyFunction2.apply(patches, self.weight, self.bias, self.spc)

def extract_image_patches(x, kernel, stride=1, dilation=1):
    # Do TF 'SAME' Padding
    b,h,w,c = x.shape
    h2 = np.ceil(h / stride).astype(int)
    w2 = np.ceil(w / stride).astype(int)
    pad_row = (h2 - 1) * stride + (kernel - 1) * dilation + 1 - h
    pad_col = (w2 - 1) * stride + (kernel - 1) * dilation + 1 - w
    xp = F.pad(x, (0,0,pad_row//2, pad_row - pad_row//2, pad_col//2, pad_col - pad_col//2), "constant", 1)

    # print(stride)
    # Extract patches
    patches = xp.unfold(1, kernel, stride).unfold(2, kernel, stride)
    # print(patches.shape)
    patches = patches.permute(0,1,2,5,3,4).contiguous()
    # print(patches.shape)
    
    return patches.reshape(b,h//stride, w//stride, -1)