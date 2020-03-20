#!/usr/bin/python3
#SPController.py
#Jonathan S. Takeshita, Ryan Karl, Mark Horeni, Kathryn Hund

import subprocess
import os
import struct
import sys
from struct import *

ENCLAVE_EXE_PATH = "./enclave_driver"
TMPDIR = "/tmp/"
#Name is of the writer
FIFO_NAMES = {"gpu":TMPDIR+"gpu.pipe", "enclave":TMPDIR+"enclave.pipe"}
INT_BYTES = 4
BYTEORDER = sys.byteorder

class SPController:

  #Initialization does not actually start subprocess
  def __init__(self, enclave_executable=ENCLAVE_EXE_PATH, pipe_names=FIFO_NAMES):
    self.pipe_names = pipe_names
    for producer, filepath in self.pipe_names.items():
      #Clear any previous file
      try:
        os.mkfifo(filepath)
      except FileExistsError as e:
        os.unlink(filepath)
        os.mkfifo(filepath)
    self.args = [enclave_executable, "-i", self.pipe_names["gpu"], "-o", self.pipe_names["enclave"]]
    self.gpu_pipe_w = None
    self.enclave_pipe_r = None
    self.proc = None

  #Call to actually start the subprocess and open pipes
  def start(self):
    self.proc = subprocess.Popen(self.args)
    self.gpu_pipe_w = open(self.pipe_names["gpu"], 'wb', buffering=0) #Convert FD to pipe object
    self.enclave_pipe_r = open(self.pipe_names["enclave"], 'rb', buffering=0)
    
  #input_data is str by default (changeable if desired)
  #Returns bytes
  def query_enclave(self, input_data, send_header=True, raw_bytes=False):
    if send_header:  
      header = len(input_data).to_bytes(INT_BYTES, byteorder=BYTEORDER)
      self.gpu_pipe_w.write(header)
    if not raw_bytes:
      input_data = input_data.encode()
    self.gpu_pipe_w.write(input_data)
    response_size = -1
    if send_header:
      header_resp = self.enclave_pipe_r.read(INT_BYTES)
      response_size = int.from_bytes(header_resp, byteorder=BYTEORDER)
    else:
      raise NotImplementedError #Need to implement this if we do file-based sizes
    if response_size == -1:
      return None  
    enclave_response = self.enclave_pipe_r.read(response_size)
    return enclave_response
    
  def close(self, force=True, cleanup=True):
    if not force:
      self.proc.wait()
    self.proc.kill()
    self.gpu_pipe_w.close()
    self.enclave_pipe_r.close()
    if cleanup:
      self.cleanup()
    
  #WARNING: do not call this before close()  
  def cleanup(self):
    for producer, pipepath in self.pipe_names.items():
      try:
        os.unlink(pipepath)
      except FileExistsError as e:
        continue
        
    
#An example of how to use the SubProcess Controller
def main():
  strings = ["Jon", "Ryan", "Taeho", "Changhao"]
  spc = SPController()
  spc.start()
  for s in strings:
    ret = spc.query_enclave(s)
    if ret is None:
      print("Verification failed!")
    else:  
      print("Response: " + str(ret))
  spc.close()
  return  
    
if __name__ == '__main__':
  main()
  
