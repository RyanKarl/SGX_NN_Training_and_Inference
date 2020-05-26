import torch


global_wb = 16
global_ab = 8
global_gb = 8
global_eb = 8
global_ub = 8
global_qb = 8
global_pb = 8
global_rb = 16
global_sb = 1
global_rfb = 8
global_sig = 8

global_w1update = 0
global_w2update = 0
global_w3update = 0

global_beta = 1.5
global_lr = 1 #8

weight_mult = 1

# checks for quant
valid_w_vals = None
count_w_vals = None


def step_d(bits): 
    return 2.0 ** (bits - 1)

def org_sigma(bits): 
    return 2.0 ** (1 - bits)

def shift(x):
    if x == 0:
        return 1
    return 2 ** torch.round(torch.log2(x))

def clip(x, bits):
    if bits == 1:
        delta = 0.
    else:
        delta = 1./step_d(bits)
    maxv = +1 - delta
    minv = -1 + delta
    return torch.clamp(x, float(minv), float(maxv))

def quant(x, bits):
    if bits == 1: # BNN
        return torch.sign(x)
    else:
        scale = step_d(bits)
        return torch.round(x * scale ) / scale

def quant_w(x, scale = 1):
    if x is None:
        return 0

    with torch.no_grad():
        y = quant(clip(x, global_wb) , global_wb)
        diff = (y - x)

    #if scale <= 1.8:
    #    return x + diff
    return (x + diff)/scale

def quant_s(x, scale = 1):
    if x is None:
        return 0

    with torch.no_grad():
        y = quant(clip(x, global_sb) , global_sb)
        diff = (y - x)

    #if scale <= 1.8:
    #    return x + diff
    return (x + diff)/scale

def quant01(x, bits = 8):
    scale = 2.0 ** (bits)
    return torch.round(x * scale ) / scale


def quant11(x, bits = 8):
    scale = 2.0 ** (bits-1) -1
    return torch.round(x*scale)/scale

def squant_ns(x, bits):
    x = torch.clamp(x, 0, 1)
    norm_int = torch.floor(x)
    norm_float = quant(x - norm_int, global_rb)
    rand_float = quant(torch.FloatTensor(x.shape).uniform_(0,1).to(x.device), global_rb)

    zero_prevention_step = torch.sign(norm_float - rand_float)
    zero_prevention_step[zero_prevention_step == 0] = 1
    norm = ( norm_int + 0.5 * (zero_prevention_step + 1) )
    scale = 2.0 ** (bits)
    return torch.round(x * scale ) / scale

def quant_act(x):
    save_x = x
    x = clip(x, global_ab)
    diff_map = (save_x == x)
    with torch.no_grad():
        y = quant(x, global_ab)
        diff = y - x
    return x + diff #, diff_map

def quant_sig(x):
    # inputs are outputs of sigmoid, so betwen 0 and 1
    scale = 2.0 ** global_sig
    return torch.round(x * scale ) / scale


def quant_generic(x, cb):
    save_x = x
    x = clip(x, cb)
    diff_map = (save_x == x)
    with torch.no_grad():
        y = quant(x, global_ab)
        diff = y - x
    return x + diff, diff_map


def quant_grad(x):
    # those gonna be ternary, or we can tweak the lr
    xmax = torch.max(torch.abs(x))
    norm = global_lr * x / shift(xmax)

    norm_sign = torch.sign(norm)
    norm_abs = torch.abs(norm)
    norm_int = torch.floor(norm_abs)
    norm_float = quant(norm_abs - norm_int, global_rb)
    rand_float = quant(torch.FloatTensor(x.shape).uniform_(0,1).to(x.device), global_rb)
    #norm = norm_sign.double() * ( norm_int.double() + 0.5 * (torch.sign(norm_float.double() - rand_float.double()) + 1) )
    zero_prevention_step = torch.sign(norm_float - rand_float)
    zero_prevention_step[zero_prevention_step == 0] = 1
    norm = norm_sign * ( norm_int + 0.5 * (zero_prevention_step + 1) )

    return norm / step_d(global_gb) * weight_mult

def quant_err(x):
    # if (x.abs() > 1).sum() != 0:
    #     import pdb; pdb.set_trace()
    alpha = shift(torch.max(torch.abs(x)))
    return quant(clip(x / alpha, global_eb), global_eb)

def init_layer_weights(weights_layer, shape, factor=1):
    fan_in = shape

    limit = torch.sqrt(torch.tensor([3*factor/fan_in]))
    Wm = global_beta/step_d(torch.tensor([float(global_wb)]))
    scale = 2 ** round(math.log(Wm / limit, 2.0))
    scale = scale if scale > 1 else 1.0
    limit = Wm if Wm > limit else limit

    torch.nn.init.uniform_(weights_layer, a = -float(limit), b = float(limit))
    weights_layer.data = quant_generic(weights_layer.data, global_gb)[0]
    return torch.tensor([float(scale)])

# sum of square errors
def SSE(y_true, y_pred):
    y_pred = to_cat(y_pred, 10, device='cuda:0')
    return 0.5 * torch.sum((y_true - y_pred)**2)

def to_cat(inp_tensor, num_class, device):
    out_tensor = torch.zeros([inp_tensor.shape[0], num_class], device=device)
    out_tensor[torch.arange(inp_tensor.shape[0]).to(device), torch.tensor(inp_tensor, dtype = int, device=device)] = 1
    return out_tensor
