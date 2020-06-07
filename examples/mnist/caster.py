#import torch
from SPController import SPController
import numpy as np
import sys

spc = SPController(debug=False)

spc.start(verbose=3)

f = open("caster.dat", "wb")

layers = 5

for i in range(layers):
    print("GPU on layer " + str(i+1))
    if spc.good():
        a = spc.read_matrix_from_enclave()
        a = a.astype(np.float32)
        print(a.shape)
    else:
        sys.exit(1)

    if spc.good():
        b = spc.read_matrix_from_enclave()
        b = b.astype(np.float32)
        print(b.shape)
    else:
        sys.exit(1)

    if spc.good():
        spc.send_to_enclave((a @ b))
        outdata = spc.validate_one_matrix((a @ b))
        f.write(outdata[0])
        f.write(outdata[1])
        #spc.send_to_enclave((a @ b.t()))
    else:
        sys.exit(1)    
        
        
print("GPU starting backprop")        
        
for i in range(layers):     
  if spc.good():
    x = spc.read_matrix_from_enclave()
    assert(x.dtype == np.float32)
  if spc.good():
    y = spc.read_matrix_from_enclave()  
  if spc.good():
    z = spc.read_matrix_from_enclave()    
    
  xy = x @ y
  xz = x @ z

  if spc.good():
    spc.send_to_enclave(xy)
    outdata = spc.validate_one_matrix(xy)
    f.write(outdata[0])
    f.write(outdata[1])  
  if spc.good():
    spc.send_to_enclave(xz)
    outdata = spc.validate_one_matrix(xz)
    f.write(outdata[0])
    f.write(outdata[1])    
        
        
        
    

