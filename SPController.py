import subprocess
import os
import struct
from struct import *

ENCLAVE_EXE_PATH = "./enclave_driver"
TMPDIR = "/tmp/"
#Name is of the writer
FIFO_NAMES = {"gpu":TMPDIR+"gpu.pipe", "enclave":TMPDIR+"enclave.pipe"}
#File descriptors from GPU and enclave
#gpu_pipe_r, gpu_pipe_w = os.pipe()
#enclave_pipe_r, enclave_pipe_w = os.pipe()
for producer, filepath in FIFO_NAMES.items():
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
args = [ENCLAVE_EXE_PATH, "-i", FIFO_NAMES["gpu"], "-o", FIFO_NAMES["enclave"]]
proc = subprocess.Popen(args)


gpu_pipe_w = open(FIFO_NAMES["gpu"], 'wb', buffering=0) #Convert FD to pipe object
enclave_pipe_r = open(FIFO_NAMES["enclave"], 'rb', buffering=0)

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
