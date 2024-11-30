import tensorflow as tf

n_f = 40000
lrate = 0.005
N0 = 7000#8000
Nb = 2000#4000
tryerr = 5*10**(-4)

tf_iter1=10000
newton_iter1=60000

num_layer=3
width=70
layer_sizes=[3]
for i in range(num_layer):
    layer_sizes.append(width)
layer_sizes.append(1)

doubpa=0
if(doubpa ==1):
    tfdoubstr = tf.float64
else:
    tfdoubstr = tf.float32
    
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

