#import torch
from SPController import SPController
import numpy as np
import sys

spc = SPController(debug=False)

spc.start(verbose=3)

for i in range(5):
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

    if spc.good():
        spc.send_to_enclave((a @ b.t()))
    else:
        sys.exit(1)

