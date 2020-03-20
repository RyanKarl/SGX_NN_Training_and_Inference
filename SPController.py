#!/usr/bin/python3
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

  def __init__(self, enclave_executable=ENCLAVE_EXE_PATH, pipe_names=FIFO_NAMES):
    self.pipe_names = pipe_names
    for producer, filepath in self.pipe_names.items():
      #Clear any previous file
      #if os.path.exists(filepath):
      try:
        os.mkfifo(filepath)
      except FileExistsError as e:
        os.unlink(filepath)
        os.mkfifo(filepath)
    #Close pipes used by C process?
    #os.close(gpu_pipe_r)
    #os.close(enclave_pipe_w)
    self.args = [enclave_executable, "-i", self.pipe_names["gpu"], "-o", self.pipe_names["enclave"]]
    self.gpu_pipe_w = None
    self.enclave_pipe_r = None
    self.proc = None

  def start(self):
    self.proc = subprocess.Popen(self.args)
    self.gpu_pipe_w = open(self.pipe_names["gpu"], 'wb', buffering=0) #Convert FD to pipe object
    self.enclave_pipe_r = open(self.pipe_names["enclave"], 'rb', buffering=0)
    
    
  #input_data should be str by default
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
      #DEBUG
      print(header_resp)
      response_size = int.from_bytes(header_resp, byteorder=BYTEORDER)
    else:
      raise NotImplementedError #Need to implement this if we do file-based sizes
    enclave_response = self.enclave_pipe_r.read(response_size)
    return enclave_response
    
  #Waits for C program to finish by default  
  def close(self, force=True):
    if not force:
      self.proc.wait()
    self.proc.kill()
    self.gpu_pipe_w.close()
    self.enclave_pipe_r.close()
    
    
    
def main():
  strings = ["Jon", "Ryan", "Taeho", "Changhao"]
  spc = SPController()
  spc.start()
  for s in strings:
    ret = spc.query_enclave(s)
    print("Response: " + str(ret))
  spc.close()
  return  
    
if __name__ == '__main__':
  main()
  
  
  
  
  
  
  
  
'''
to_send = "Hello world"
#Little-endian mode
header = len(to_send).to_bytes(4, byteorder='little')
#header = bytes([len(to_send)])
print("Sending header: " + str(header))
gpu_pipe_w.write(header)
#Experiment: try closing and reopening pipe to remove lock
#gpu_pipe_w.close()
#gpu_pipe_w = open(FIFO_NAMES["gpu"], 'wb', buffering=0)
print("Sending message: " + str(to_send))
gpu_pipe_w.write(to_send.encode())
#Close pipe so C can read
#gpu_pipe_w.close()

 #Convert FD to pipe object


result = enclave_pipe_r.read(len(to_send))

print("C response: " + str(result))

#C proc will close its pipes
#gpu_pipe_r.close()
gpu_pipe_w.close()
enclave_pipe_r.close()
#enclave_pipe_w.close()

proc.kill()
'''
