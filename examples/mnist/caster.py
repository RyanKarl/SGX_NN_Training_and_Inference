#import torch
from SPController import SPController
import numpy as np
import sys

spc = SPController(debug=False)

spc.start(verbose=3)

f = open("caster.dat", "wb")

with open("Master_Arch.txt", 'r') as mf:
  arch = mf.readline()
  archs = arch.split(' ')
  EPOCHS = int(archs[-1])
  num_inputs = int(archs[0])
  batchsize = int(archs[1])
  num_batches = int(num_inputs / batchsize)
  if (num_inputs % batchsize) != 0:
    num_batches += 1


layers = 5

activations = [None] * layers
weights = [None] * layers
outputs = [None] * layers

def is_zero_mat(arg):
  for x in np.nditer(arg):
    if x != 0:
      return False
  return True    
    

for j in range(EPOCHS):
  print("GPU starting epoch " + str(j))
  for b_idx in range(num_batches):
    for i in range(layers):
    
          a = spc.read_matrix_from_enclave()
            #a = a.astype(np.float64)
          activations[i] = a
            #print("a received by GPU: " + str(a)
            #a_sum = np.sum(a)
            #print("Sum of a: " + str(a_sum))
              
            #print(a.shape)

          b = spc.read_matrix_from_enclave()
          weights[i] = b
       
          c = (a @ b)
          spc.send_to_enclave(c)
          outputs[i] = c
          
    for i in range(layers-1)[::-1]: 

      grad_output = spc.read_matrix_from_enclave()
      d = grad_output @ weights[i].transpose()
         
      e = grad_output.transpose() @ activations[i]

      spc.send_to_enclave(d)

      spc.send_to_enclave(e)


          
spc.close(force=False)        
