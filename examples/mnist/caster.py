#import torch
from SPController import SPController
import numpy as np
import sys

spc = SPController(debug=False)

spc.start(verbose=2)

f = open("caster.dat", "wb")

for i in range(5):
    print("GPU on layer " + str(i+1))
    if spc.good():
        a = spc.read_matrix_from_enclave()
        print(a.shape)
    else:
        sys.exit(1)

    if spc.good():
        b = spc.read_matrix_from_enclave()
        print(b.shape)
    else:
        sys.exit(1)

    if spc.good() or True:
        #print(a @ b)
        print("Max. elt. of a: " + str(np.max(a)))
        print("Max. elt. of b: " + str(np.max(b)))
        spc.send_to_enclave((a @ b))
        outdata = spc.validate_one_matrix((a @ b))
        f.write(outdata[0])
        f.write(outdata[1])
        #spc.send_to_enclave((a @ b.t()))
    else:
        sys.exit(1)

