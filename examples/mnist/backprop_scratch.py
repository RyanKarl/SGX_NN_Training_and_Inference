def one_min_sq(x, term):
  return 1 - torch.tanh(x + term)**2


def SGXB(grad_output, input, weight, output):
    rand_mask = super_mega_mask[0:input.shape[0], 0:input.shape[1]]
    weight_rand_mask = super_mega_mask[0:weight.shape[0], 0:weight.shape[1]]
    a = input 
    b = weight 
    c = grad_output.clone()
    grad_rand_mask = super_mega_mask[0:c.shape[0], 0:c.shape[1]]
    
    
    diff = rand_mask @ b.t()
    diff2 = a @ weight_rand_mask.t()
    diff3 = rand_mask @ weight_rand_mask.t()
    c = c * (1-torch.tanh(output  - diff - diff2 + diff3)**2)
    
    diff_tmp = diff3 - diff2 - diff
    grad_rand_mask_transformed = one_min_sq(grad_rand_mask, diff_tmp)
    
    d = c @ b
    diffa = grad_rand_mask_transformed @ b
    diffb = c @ weight_rand_mask
    diffc = grad_rand_mask_transformed @ weight_rand_mask
    
    d = d - diffa - diffb + diffc
    
    '''
    ct = grad_output.clone() - grad_rand_mask
    ct = ct * one_min_sq(output, diff_tmp)
    '''
    
    e = c.t() @ a
    diffe = c.t() @ rand_mask
    difff = grad_rand_mask_transformed.t() @ a
    diffg = grad_rand_mask_transformed.t() @ rand_mask
    
    e = (e - diffe) + (-difff + diffg)
    #Masking
    rand_mask = torch.ones(d.shape, device = "cuda:0")
    weight_rand_mask = super_mega_mask[0:e.shape[0], 0:e.shape[1]]
    return d + super_mega_mask[0:d.shape[0], 0:d.shape[1]], e + weight_rand_mask
