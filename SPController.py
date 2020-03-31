#!/usr/bin/python3
#SPController.py
#Jonathan S. Takeshita, Ryan Karl, Mark Horeni

import subprocess
import os
import sys
import struct
import numpy as np

ENCLAVE_EXE_PATH = "./enclave_driver"
TMPDIR = "/tmp/"
#Name is of the writer
FIFO_NAMES = {"gpu":TMPDIR+"gpu.pipe", "enclave":TMPDIR+"enclave.pipe"}
INT_BYTES = 4
FLOAT_BYTES = 4
# endianess
BYTEORDER = sys.byteorder
DATA_DIMENSIONS = 3
# f for float
STRUCT_PACK_FMT = 'f'
NUM_MATRICES = 3
MAT_DIM = 2
BUFFERING=-1

class SPController:

  #Initialization does not actually start subprocess
  def __init__(self, enclave_executable=ENCLAVE_EXE_PATH, pipe_names=FIFO_NAMES):
    self.pipe_names = pipe_names
    for producer, filepath in self.pipe_names.items():
      #Clear any previous file
      try:
        #os.mkfifo is system call to create pipe
        os.mkfifo(filepath)
      except FileExistsError as e:
        os.unlink(filepath)
        os.mkfifo(filepath)
    self.args = [enclave_executable, "-i", self.pipe_names["gpu"], "-o", self.pipe_names["enclave"]]
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
    
    #Popen starts the process
    self.proc = subprocess.Popen(self.args)
    #No buffering to force immediate writes
    self.gpu_pipe_w = open(self.pipe_names["gpu"], 'wb', buffering=BUFFERING)
    self.enclave_pipe_r = open(self.pipe_names["enclave"], 'rb', buffering=BUFFERING)
    
  #input_data is list of 3 float ndarrays (2-D) by default
  def query_enclave(self, input_matrices, raw_bytes=False):
    #Send 3 ints as shape - use nditer to go over array of floats
    
    if len(input_matrices) != NUM_MATRICES:
      print("ERROR: Matrix list incorrect: " + str(data_shape))
      return None
      
    matrix_dims = list()
    
    for im in input_matrices:
      data_shape = im.shape
      if len(data_shape) != MAT_DIM:
        print("ERROR: matrix non 2-dimensional")
        raise RuntimeError 
      for s in data_shape:
        matrix_dims.append(s)  

    if len(matrix_dims) != MAT_DIM*NUM_MATRICES:
      print("ERROR: incorrect number of dimensions: " + str(len(matrix_dims)))
      raise RuntimeError
    
    #b'' is python notation for byte string
    header = b''
    for dim in matrix_dims:
      header += dim.to_bytes(INT_BYTES, byteorder=BYTEORDER)
    
    #Write input header  
    #Check to see if process is still ok
    if self.proc.poll() is not None:
      print("ERROR: subprocess ended prematurely")
      return None  
    if self.gpu_pipe_w.closed:
      print("ERROR: output pipe closed")
      raise RuntimeError  
    self.gpu_pipe_w.write(header)
    self.gpu_pipe_w.flush()
    
    #Packing input data from 3D numpy array to bytes
    input_data = b''
    if not raw_bytes:
      #.pack takes object and puts into bit field
      #.nditer iterates over multidimensional array in c style i.e. gets raw values
      for y in input_matrices:
        input_data += b''.join([struct.pack(STRUCT_PACK_FMT, x) for x in np.nditer(y, order='C')])
    else:
      input_data = np.nditer(input_data, order='C')  
      
    #Write input  
    #Check to see if process is still ok
    if self.proc.poll() is not None:
      print("ERROR: subprocess ended prematurely")
      return None  
    if self.gpu_pipe_w.closed:
      print("ERROR: output pipe closed")
      raise RuntimeError    
    self.gpu_pipe_w.write(input_data)
    self.gpu_pipe_w.flush()
    response_sizes = list()

    #Read response header
    #Check to see if process is still ok
    if self.proc.poll() is not None:
      print("ERROR: subprocess ended prematurely")
      return None
    if self.enclave_pipe_r.closed:
      print("ERROR: input pipe closed")
      return None 
    header_resp = self.enclave_pipe_r.read(MAT_DIM*INT_BYTES)
    if len(header_resp) != MAT_DIM*INT_BYTES:
      print("ERROR: incorrect number of header bytes read in: " + str(len(header_resp)))
      return None

    #Reads blocks of 4 bytes into ints
    response_sizes = [int.from_bytes(header_resp[i:i+INT_BYTES], byteorder=BYTEORDER, signed=True) for i in [INT_BYTES*y for y in range(MAT_DIM)]]
    #Equivalent to: response_sizes = [int.from_bytes(x, byteorder=BYTEORDER) for x in [header_resp[0:4], header_resp[4:8], header_resp[8:12]]]
    if len(response_sizes) != MAT_DIM:
      print("ERROR: shape incorrect")
      return None
    #If any dimension sizes are negative 1 this fails
    if -1 in response_sizes:
      print("Frievald's Algorithm failed to verify!")
      return np.asarray([]) #Return an empty list
    #DEBUG
    print(response_sizes)  
        
    num_floats = 1
    for d in response_sizes:
      num_floats *= d  
    if self.enclave_pipe_r.closed:
      print("ERROR: input pipe closed")
      return None 
      
    #Check to see if process is still ok
    if self.proc.poll() is not None:
      print("ERROR: subprocess ended prematurely")
      return None  
    enclave_response = self.enclave_pipe_r.read(num_floats*FLOAT_BYTES)
    if len(enclave_response) != num_floats*FLOAT_BYTES:
      print("ERROR: incorrect number of data bytes read in: " + str(len(enclave_response)))
      return None
    
    #Unpacks bytes into array of floats
    float_resp = [struct.unpack(str(num_floats) + STRUCT_PACK_FMT, enclave_response)]
    return np.reshape(float_resp, response_sizes)
    
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
        
    
#An example of how to use the SubProcess Controller
def main():
  a = [ [ 1.0, 1.0 ], [ 1.0, 1.0 ] ] 
  b = [ [ 1.0, 1.0 ], [ 1.0, 1.0 ] ] 
  c = [ [ 2.0, 2.0 ], [ 2.0, 2.0 ] ] 
  arrs = [np.asarray(x) for x in [a, b, c]]
  #initializes SPController
  spc = SPController()
  spc.start(verbose=2)
  for i in range(4):
    ret = spc.query_enclave(arrs)
    if ret is None:
      print("Error in subprocess, exiting")
      return
    elif len(ret) == 0:
      print("Frievald's Algorithm failed!")  
    else:  
      print("Response: " + str(ret) + '\n')
  spc.close()
  return  
    
if __name__ == '__main__':
  main()
  
