import torch

n_f = 20000
lrate = 0.005
N0 = 3000
Nb = 3000

tf_iter1= 10000
newton_iter1=15000

num_layer=7
width= 48
layer_sizes=[3]
for i in range(num_layer):
    layer_sizes.append(width)
layer_sizes.append(1)

doubpa = 0
if doubpa == 1:
    torch_dtype = torch.float64
else:
    torch_dtype = torch.float32

# 训练设备为GPU还是CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if device.type == 'cpu':
    print("wrong device")

