import numpy as np
import torch

n_f = 20000
lrate = 0.005
Nx, Nt= 256, 100

N0 = 100
Nb = 100

tf_iter1= 15000
newton_iter1= 60000

num_layer=8
width=30
layer_sizes=[2]
for i in range(num_layer):
    layer_sizes.append(width)
layer_sizes.append(1)

# 训练设备为GPU还是CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if device.type == 'cpu':
    print("wrong device")

