import torch
import pickle

super_mega_mask = torch.rand(10000,10000)

pickle.dump(super_mega_mask, open("mask.p", 'wb'))