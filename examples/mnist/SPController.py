#!/usr/bin/python3
#SPController.py
#Jonathan S. Takeshita, Ryan Karl, Mark Horeni

import subprocess
import os
import sys
import struct
import numpy as np
from subprocess import PIPE

ENCLAVE_EXE_PATH = "./app"
TMPDIR = "./"
#Name is of the writer
FIFO_NAMES = {"gpu":TMPDIR+".gpu.pipe", "enclave":TMPDIR+".enclave.pipe"}
INT_BYTES = 4

NP_FLOATYPE = np.float32
FLOAT_BYTES = 4
# f for float, d for double
STRUCT_PACK_FMT = 'f'

# endianess
BYTEORDER = sys.byteorder
DATA_DIMENSIONS = 3

NUM_MATRICES = 3
MAT_DIM = 2
BUFFERING=-1

DO_BACKPROP = True

#Hardcoded names
INPUT_FILE = 'mnist_train.csv'
ARCH_FILE = 'Master_Arch.txt'
LAYER_FILE = 'weights16.txt'

class SPController:

  #Initialization does not actually start subprocess
  def __init__(self, enclave_executable=ENCLAVE_EXE_PATH, pipe_names=FIFO_NAMES, debug=False, use_std_io=False, do_backprop=DO_BACKPROP):
    self.pipe_names = pipe_names
    for producer, filepath in self.pipe_names.items():
      #Clear any previous file
      try:
        #os.mkfifo is system call to create pipe
        #print("FILEPATH: " + filepath)
        os.mkfifo(filepath)
      except FileExistsError as e:
        os.unlink(filepath)
        os.mkfifo(filepath)
    self.args = list()   
    self.debug = debug 
    self.args = [enclave_executable, "-i", self.pipe_names["gpu"], "-o", self.pipe_names["enclave"]]
    if do_backprop:
      self.args += ['-b']
    self.use_std_io = use_std_io
    if not use_std_io:
      self.args += ["-c", INPUT_FILE, "-s", ARCH_FILE]
    #Pipe that python program writes to
    self.gpu_pipe_w = None
    #Pipe that enclave reads from
    self.enclave_pipe_r = None
    #Subprocess object
    self.proc = None

  #Call to actually start the subprocess and open pipes
  def start(self, verbose=0):
    if verbose > 0:
      for i in range(verbose):
        self.args.append("-v")
    #No buffering to force immediate writes
    if not self.use_std_io:
      self.proc = subprocess.Popen(self.args)  
      self.gpu_pipe_w = open(self.pipe_names["gpu"], 'wb', buffering=BUFFERING)
      self.enclave_pipe_r = open(self.pipe_names["enclave"], 'rb', buffering=BUFFERING)
      if self.debug:
        print("-i " + self.gpu_pipe_w)
        print("-o " + self.enclave_pipe_r)
    else:
      self.proc = subprocess.Popen(self.args, stdout=PIPE, stdin=PIPE)  
      self.gpu_pipe_w = self.proc.stdin
      self.enclave_pipe_r = self.proc.stdout
      
  #Deprecated    
  @staticmethod  
  def validate_and_pack_matrices(input_matrices, raw_bytes=False):
    if len(input_matrices) != NUM_MATRICES:
      print("ERROR: Matrix list incorrect: " + len(input_matrices))
      return None
      
    matrix_dims = list()
    
    for im in input_matrices:
      data_shape = im.shape
      if len(data_shape) != MAT_DIM:
        print("ERROR: matrix non 2-dimensional")
        return None 
      for s in data_shape:
        matrix_dims.append(s)  

    if len(matrix_dims) != MAT_DIM*NUM_MATRICES:
      print("ERROR: incorrect number of dimensions: " + str(len(matrix_dims)))
      return None
    #b'' is python notation for byte string
    #Send 3 ints as shape - use nditer to go over array of floats
    header = b''
    for dim in matrix_dims:
      header += dim.to_bytes(INT_BYTES, byteorder=BYTEORDER)  
    #Packing input data from 3D numpy array to bytes
    input_data = b''
    if not raw_bytes:
      #.pack takes object and puts into bit field
      #.nditer iterates over multidimensional array in c style i.e. gets raw values
      for y in input_matrices:
        input_data += b''.join([struct.pack(STRUCT_PACK_FMT, x) for x in y.flat])
    else:
      input_data = np.nditer(input_data, order='C')  
      
    return (header, input_data)  

  @staticmethod 
  def validate_one_matrix(mat):
    data_shape = mat.shape
    if len(data_shape) != MAT_DIM:
      print("ERROR: matrix non 2-dimensional")
      return None 
    header = b''
    for dim in data_shape:
      header += dim.to_bytes(INT_BYTES, byteorder=BYTEORDER) 
    #packed_data = b''.join([struct.pack(STRUCT_PACK_FMT, x) for x in np.nditer(mat, order='C')])
    #WARNING: assumes row-major order
    packed_data = b''.join([struct.pack(STRUCT_PACK_FMT, x) for x in mat.flat])
    return (header, packed_data)


  def read_matrix_from_enclave(self):
    #Read response header
    #Check to see if process is still ok
    if self.proc.poll() is not None:
      print("ERROR: subprocess ended prematurely, return code is " + str(self.proc.poll()))
      return None
    if self.enclave_pipe_r.closed:
      print("ERROR: input pipe closed")
      return None 
    header_resp = self.enclave_pipe_r.read(MAT_DIM*INT_BYTES)
    if len(header_resp) != MAT_DIM*INT_BYTES:
      print("ERROR: incorrect number of header bytes read in: " + str(len(header_resp)))
      print("\t Header response was: " + str(header_resp))
      return None

    #Reads blocks of 4 bytes into ints
    response_sizes = [int.from_bytes(header_resp[i:i+INT_BYTES], byteorder=BYTEORDER, signed=True) for i in [INT_BYTES*y for y in range(MAT_DIM)]]
    #Equivalent to: response_sizes = [int.from_bytes(x, byteorder=BYTEORDER) for x in [header_resp[0:4], header_resp[4:8], header_resp[8:12]]]
    if len(response_sizes) != MAT_DIM:
      print("ERROR: shape incorrect")
      return None
    #If any dimension sizes are negative 1 this fails
    if -1 in response_sizes:
      #print("Frievald's Algorithm failed to verify!")
      return np.asarray([]) #Return an empty list 
        
    num_floats = 1
    for d in response_sizes:
      num_floats *= d  
    if self.enclave_pipe_r.closed:
      print("ERROR: input pipe closed")
      return None 
      
    #Check to see if process is still ok
    if self.proc.poll() is not None:
      print("ERROR: subprocess ended prematurely, return code is " + str(self.proc.poll()))
      return None  

    enclave_response = self.enclave_pipe_r.read(num_floats*FLOAT_BYTES)
    if len(enclave_response) != num_floats*FLOAT_BYTES:
      print("ERROR: incorrect number of data bytes read in: " + str(len(enclave_response)))
      return None
    
    #Unpacks bytes into array of floats
    float_resp = np.asarray([struct.unpack(str(num_floats) + STRUCT_PACK_FMT, enclave_response)])
    return np.reshape(float_resp, response_sizes).astype(NP_FLOATYPE)

  
  #TODO figure out return vals
  def send_to_enclave(self, mult_result):  
    output_data = SPController.validate_one_matrix(mult_result.astype(NP_FLOATYPE))
    if output_data is None:
      print("Bad input")
      return None
    #Write input header  
    #Check to see if process is still ok
    if self.proc.poll() is not None:
      print("ERROR: subprocess ended prematurely, return code is " + str(self.proc.poll()))
      return None  
    if self.gpu_pipe_w.closed:
      print("ERROR: output pipe closed")
      raise RuntimeError  
    self.gpu_pipe_w.write(output_data[0])
    self.gpu_pipe_w.flush()
    
    #Write input  
    #Check to see if process is still ok
    if self.proc.poll() is not None:
      print("ERROR: subprocess ended prematurely, return code is " + str(self.proc.poll()))
      return None  
    if self.gpu_pipe_w.closed:
      print("ERROR: output pipe closed")
      raise RuntimeError    
    self.gpu_pipe_w.write(output_data[1])
    self.gpu_pipe_w.flush()  


  #input_data is list of 3 float ndarrays (2-D) by default
  def query_enclave(self, input_matrices, raw_bytes=False):
    packed_input = SPController.validate_and_pack_matrices(input_matrices, raw_bytes)
    if packed_input is None:
      print("Bad input")
      return None
   
    #Write input header  
    #Check to see if process is still ok
    if self.proc.poll() is not None:
      print("ERROR: subprocess ended prematurely, return code is " + str(self.proc.poll()))
      return None  
    if self.gpu_pipe_w.closed:
      print("ERROR: output pipe closed")
      raise RuntimeError  
    self.gpu_pipe_w.write(packed_input[0])
    self.gpu_pipe_w.flush()
    
    #Write input  
    #Check to see if process is still ok
    if self.proc.poll() is not None:
      print("ERROR: subprocess ended prematurely, return code is " + str(self.proc.poll()))
      return None  
    if self.gpu_pipe_w.closed:
      print("ERROR: output pipe closed")
      raise RuntimeError    
    self.gpu_pipe_w.write(packed_input[1])
    self.gpu_pipe_w.flush()
    response_sizes = list()

    #Read response header
    #Check to see if process is still ok
    if self.proc.poll() is not None:
      print("ERROR: subprocess ended prematurely, return code is " + str(self.proc.poll()))
      return None
    if self.enclave_pipe_r.closed:
      print("ERROR: input pipe closed")
      return None 
    header_resp = self.enclave_pipe_r.read(MAT_DIM*INT_BYTES)
    if len(header_resp) != MAT_DIM*INT_BYTES:
      print("ERROR: incorrect number of header bytes read in: " + str(len(header_resp)))
      print("\t Header response was: " + str(header_resp))
      return None

    #Reads blocks of 4 bytes into ints
    response_sizes = [int.from_bytes(header_resp[i:i+INT_BYTES], byteorder=BYTEORDER, signed=True) for i in [INT_BYTES*y for y in range(MAT_DIM)]]
    #Equivalent to: response_sizes = [int.from_bytes(x, byteorder=BYTEORDER) for x in [header_resp[0:4], header_resp[4:8], header_resp[8:12]]]
    if len(response_sizes) != MAT_DIM:
      print("ERROR: shape incorrect")
      return None
    #If any dimension sizes are negative 1 this fails
    if -1 in response_sizes:
      #print("Frievald's Algorithm failed to verify!")
      return np.asarray([]) #Return an empty list 
        
    num_floats = 1
    for d in response_sizes:
      num_floats *= d  
    if self.enclave_pipe_r.closed:
      print("ERROR: input pipe closed")
      return None 
      
    #Check to see if process is still ok
    if self.proc.poll() is not None:
      print("ERROR: subprocess ended prematurely, return code is " + str(self.proc.poll()))
      return None  
    enclave_response = self.enclave_pipe_r.read(num_floats*FLOAT_BYTES)
    if len(enclave_response) != num_floats*FLOAT_BYTES:
      print("ERROR: incorrect number of data bytes read in: " + str(len(enclave_response)))
      return None
    
    #Unpacks bytes into array of floats
    float_resp = [struct.unpack(str(num_floats) + STRUCT_PACK_FMT, enclave_response)]
    return np.reshape(float_resp, response_sizes).astype(NP_FLOATYPE)
    
  def close(self, force=True, cleanup=True):
    if not force:
      self.proc.wait()
    self.proc.kill()
    self.gpu_pipe_w.close()
    self.enclave_pipe_r.close()
    if cleanup:
      self.cleanup()
    
  #WARNING: do not call this before close()  
  #CLEANUP gets rid of files
  def cleanup(self):
    for producer, pipepath in self.pipe_names.items():
      try:
        os.unlink(pipepath)
      except FileExistsError as e:
        continue
        
  def good(self):
    return True if self.proc.poll() is None else False     
    
#An example of how to use the SubProcess Controller
#TODO strip out using std. IO
def main():

  outfile_name = None
  fout = None
  #Create standalone file for input
  if len(sys.argv) >= 2:
    outfile_name = sys.argv[1]
    fout = open(outfile_name, 'wb', buffering=BUFFERING)
  
  #initializes SPController
  spc = SPController(debug=False)

  spc.start(verbose=3)
  
  for i in range(2):
  
    a = spc.read_matrix_from_enclave()
    if a is None:
      print("ERROR in reading input")
    else:
      print("GPU received input:")
      print(str(a))  
      print(a.shape)
      
    b = spc.read_matrix_from_enclave()
    if b is None:
      print("ERROR in reading weights")
    else:
      print("GPU received weights:")
      print(str(b))   
      print(b.shape)
      
    c = a @ b
    print("GPU's result:")
    print(str(c))
    spc.send_to_enclave(c)  
    print("GPU sent result")
    if outfile_name is not None: 
      packed_output = SPController.validate_one_matrix(c)
      fout.write(packed_output[0])
      fout.write(packed_output[1])

  
  spc.close(force=False)
  if outfile_name is not None:
    fout.close()
  return  
    
if __name__ == '__main__':
  main()
  
