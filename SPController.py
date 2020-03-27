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
  def start(self, verbose=False):
    if verbose:
      self.args.append("-v")
    
    #Popen starts the process
    self.proc = subprocess.Popen(self.args)
    #No buffering to force immediate writes
    self.gpu_pipe_w = open(self.pipe_names["gpu"], 'wb', buffering=0)
    self.enclave_pipe_r = open(self.pipe_names["enclave"], 'rb', buffering=0)
    
  #input_data is float ndarray (3-D) by default
  def query_enclave(self, input_data, send_header=True, raw_bytes=False):
    #Send 3 ints as shape - use nditer to go over array of floats
    data_shape = input_data.shape
    if len(data_shape) != DATA_DIMENSIONS:
      print("Data shape incorrect: " + str(data_shape))
      sys.exit(0)
    if send_header:
      #b'' is python notation for byte string
      header = b''
      for dim in data_shape:
        header += dim.to_bytes(INT_BYTES, byteorder=BYTEORDER)
      self.gpu_pipe_w.write(header)
    #Packing input data from 3D numpy array to bytes
    if not raw_bytes:
      #.pack takes object and puts into bit field
      #.nditer iterates over multidimensional array in c style i.e. gets raw values
      input_data = b''.join([struct.pack(STRUCT_PACK_FMT, x) for x in np.nditer(input_data, order='C')])
    self.gpu_pipe_w.write(input_data)
    response_sizes = list()
    if send_header:
      header_resp = self.enclave_pipe_r.read(DATA_DIMENSIONS*INT_BYTES)
      #Reads blocks of 4 bytes into ints
      response_sizes = [int.from_bytes(header_resp[i:i+INT_BYTES], byteorder=BYTEORDER) for i in [INT_BYTES*y for y in range(DATA_DIMENSIONS)]]
      #response_sizes = [int.from_bytes(x, byteorder=BYTEORDER) for x in [header_resp[0:4], header_resp[4:8], header_resp[8:12]]]
    else:
      raise NotImplementedError #Need to implement this if we do file-based sizes
    if len(response_sizes) != DATA_DIMENSIONS:
      print("ERROR: shape incorrect")
      sys.exit(0)  
    #If any dimension sizes are negative 1 this fails
    if -1 in response_sizes:
      print("Frievald's Algorithm failed to verify!")
      return None
    num_floats = 1
    for d in response_sizes:
      num_floats *= d  
    enclave_response = self.enclave_pipe_r.read(num_floats*FLOAT_BYTES)
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
  floats = np.array([0.1*x for x in range(12)])
  floats = np.reshape(floats, (2, 3, 2))
  #initializes SPController
  spc = SPController()
  spc.start()
  ret = spc.query_enclave(floats)
  if ret is None:
    print("Verification failed!")
  else:  
    print("Response: " + str(ret))
  spc.close()
  return  
    
if __name__ == '__main__':
  main()
  
