#import torch
from SPController import SPController
import numpy as np
import sys

FILE_OUT = True

spc = SPController(debug=False)

VERBOSITY = 1

spc.start(verbose=VERBOSITY)

if FILE_OUT:
  f = open("caster.dat", "wb")

with open("Master_Arch.txt", 'r') as mf:
  arch = mf.readline()
  archs = arch.split(' ')
  EPOCHS = int(archs[-1])
  num_inputs = int(archs[0])
  batchsize = int(archs[1])
  layers = int(archs[3])
  num_batches = int(num_inputs / batchsize)
  if (num_inputs % batchsize) != 0:
    num_batches += 1
    
if VERBOSITY >= 2:    
  print("Epochs: ", EPOCHS)
  print("Batches: ", num_batches)    


activations = [None] * layers
weights = [None] * layers
outputs = [None] * layers

def is_zero_mat(arg):
  for x in np.nditer(arg):
    if x != 0:
      return False
  return True    
    

for j in range(EPOCHS):
  if VERBOSITY >= 2: 
    print("GPU starting epoch " + str(j))
  for b_idx in range(num_batches):
    for i in range(layers):
          if VERBOSITY >= 2:
            print("GPU (forward) on batch ", b_idx, ", epoch ", j, ", layer ", i)
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
          if FILE_OUT:
            outdata = spc.validate_one_matrix(c)
            f.write(outdata[0])
            f.write(outdata[1])
          outputs[i] = c
          
    for i in range(layers-1)[::-1]: 
      if VERBOSITY >= 2:
        print("GPU (backwards) on batch ", b_idx, ", epoch ", j, ", layer ", i)
      grad_output = spc.read_matrix_from_enclave()
      d = grad_output @ weights[i].transpose()
         
      e = grad_output.transpose() @ activations[i]

      spc.send_to_enclave(d)

      spc.send_to_enclave(e)
      
      if FILE_OUT:
        d_out = spc.validate_one_matrix(d)
        f.write(d_out[0])
        f.write(d_out[1])
        e_out = spc.validate_one_matrix(e)
        f.write(e_out[0])
        f.write(e_out[1])

if VERBOSITY >= 2:
  print("GPU finished, waiting on enclave")          
spc.close(force=False)        
