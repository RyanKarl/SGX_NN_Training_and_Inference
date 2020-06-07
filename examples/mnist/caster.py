#import torch
from SPController import SPController
import numpy as np
import sys

spc = SPController(debug=False)

spc.start(verbose=3)

f = open("caster.dat", "wb")


layers = 5

activations = [None] * layers
weights = [None] * layers
outputs = [None] * layers

for i in range(layers):
    print("GPU on layer " + str(i+1))
    if spc.good():
        a = spc.read_matrix_from_enclave()
        a = a.astype(np.float32)
        activations[i] = a
        print(a.shape)
    else:
        sys.exit(1)

    if spc.good():
        b = spc.read_matrix_from_enclave()
        b = b.astype(np.float32)
        weights[i] = b
        print(b.shape)
    else:
        sys.exit(1)

    if spc.good():
        c = (a @ b)
        spc.send_to_enclave(c)
        outputs[i] = c
        outdata = spc.validate_one_matrix(c)
        f.write(outdata[0])
        f.write(outdata[1])
        #spc.send_to_enclave((a @ b.t()))
    else:
        sys.exit(1)    
        
        
print("GPU starting backprop")        
        
for i in range(layers)[::-1]: 

  if spc.good():
    grad_output = spc.read_matrix_from_enclave()
    print(grad_output)


  d = grad_output @ weights[i].transpose()
  e = grad_output.transpose() @ actvations[i]

  if spc.good():
    spc.send_to_enclave(d)
    outdata = spc.validate_one_matrix(d)
    f.write(outdata[0])
    f.write(outdata[1])  

  if spc.good():
    spc.send_to_enclave(e)
    outdata = spc.validate_one_matrix(e)
    f.write(outdata[0])
    f.write(outdata[1])    
    
  if spc.good():
    spc.send_to_enclave(activations[i])
  if spc.good():
    spc.send_to_enclave(weights[i])

        
        
    

