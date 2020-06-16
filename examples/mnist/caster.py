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
        print("GPU on layer " + str(i))
        if spc.good():
            a = spc.read_matrix_from_enclave()
            #a = a.astype(np.float64)
            activations[i] = a
            #print("a received by GPU: " + str(a))
            if is_zero_mat(a):
              print("WARNING: a is ZERO on layer " + str(i))
            #a_sum = np.sum(a)
            #print("Sum of a: " + str(a_sum))
              
            #print(a.shape)
        else:
            sys.exit(1)

        if spc.good():
            b = spc.read_matrix_from_enclave()
            #b = b.astype(np.float64)
            weights[i] = b
            #print("b received by GPU: " + str(b))
            #print(b.shape)
        else:
            sys.exit(1)

        if spc.good():
            c = (a @ b)
            #print("c calculated by GPU: " + str(c))
            if is_zero_mat(c):
              print("WARNING: c is ZERO on layer " + str(i))
            spc.send_to_enclave(c)
            outputs[i] = c
            outdata = spc.validate_one_matrix(c)
            f.write(outdata[0])
            f.write(outdata[1])
            #spc.send_to_enclave((a @ b.t()))
        else:
            sys.exit(1)    
            
            
    print("GPU starting backprop")        
            
    for i in range(layers-1)[::-1]: 

      if spc.good():
        grad_output = spc.read_matrix_from_enclave()
        #print("grad_output at layer " + str(i) + ": " + str(grad_output))

      if grad_output is not None:
        print("Received grad_output at layer " + str(i))
      else:
        print("ERROR receiving grad_output at layer " + str(i))  
        
      print("gradoutput at layer ", i, " epoch ", j, str(grad_output))  

      d = grad_output @ weights[i].transpose()
         
      e = grad_output.transpose() @ activations[i]
      if is_zero_mat(activations[i]):
        print("WARNING: activations at layer " + str(i) + " is ZERO")
      #print("grad_output of GPU: " + str(grad_output.transpose()))
      #print("activations of GPU: " + str(activations[i]))

      
      #print("d from GPU: " + str(d))
      #print("e from GPU: " + str(e))

      if spc.good():
        spc.send_to_enclave(d)
        #print("d from GPU: " + str(d))
        outdata = spc.validate_one_matrix(d)
        f.write(outdata[0])
        f.write(outdata[1])  
        

      if spc.good():
        spc.send_to_enclave(e)
        outdata = spc.validate_one_matrix(e)
        f.write(outdata[0])
        f.write(outdata[1])    
        
      print("e_sent back by enclave at layer ", i, " epoch ", j, str(e))
        
      print("Sent d, e at layer " + str(i))   
        
      '''
      if spc.good():
        spc.send_to_enclave(activations[i])
        print("Sent activations at layer " + str(i)) 
      if spc.good():
        spc.send_to_enclave(weights[i])
      '''  

          
spc.close(force=False)        
