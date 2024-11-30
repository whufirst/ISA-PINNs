import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.io
import math
import matplotlib.gridspec as gridspec
import pickle
import os
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import layers, activations
from scipy.interpolate import griddata
from pyDOE import lhs
from globalf import *
from ISA_PINN_2D import ISA_PINN_2D

@tf.function
def f_model(model,x,y,t):
    x_diag = 2*np.pi
    y_diag = 2*np.pi

    ff = tf.exp(-2*( (x - x_diag)**2 + (y - y_diag)**2) )
    u = model.model(tf.concat([x,y,t], 1))
    #u=tf.cast(u,dtype=tfdoubstr)
    u_t=tf.gradients(u,t)[0]
    u_x=tf.gradients(u,x)[0]
    u_xx=tf.gradients(u_x,x)[0]
    u_y=tf.gradients(u,y)[0]
    u_yy=tf.gradients(u_y,y)[0]
    f=u_t- 0.5*(u_xx + u_yy) - ff

    return f
   
def loss(model,x_f,y_f, t_f,u0,u_x_lb,u_x_ub,u_y_lb,u_y_ub, XY, XT, YT,SA_weight):
    f_u_pred = model.f_model(model,x_f,y_f, t_f)
    u0_pred = model.model(tf.concat([XY[:,0:1], XY[:,1:2],model.t0],1))
    u_x_lb_pred= model.uv_model(model.x_lb, YT[:,0:1],YT[:,1:2])
    u_x_ub_pred= model.uv_model(model.x_ub, YT[:,0:1],YT[:,1:2])
    u_y_lb_pred= model.uv_model( XT[:,0:1],model.y_lb,XT[:,1:2])
    u_y_ub_pred= model.uv_model( XT[:,0:1],model.y_ub,XT[:,1:2])

    mse_0_u = tf.reduce_mean(tf.square(SA_weight["u_weights"]*(u0 - u0_pred)))

    mse_b_u = tf.reduce_mean(tf.square(SA_weight["ub_weights"]*(u_x_lb_pred - u_x_lb))) + \
            tf.reduce_mean(tf.square(SA_weight["ub_weights"]*(u_x_ub_pred - u_x_ub))) +\
            tf.reduce_mean(tf.square(SA_weight["ub_weights"]*(u_y_ub_pred - u_y_ub))) +\
            tf.reduce_mean(tf.square(SA_weight["ub_weights"]*(u_y_lb_pred - u_y_lb)))

    mse_f_u = tf.reduce_mean(tf.square(SA_weight["col_weights"]*f_u_pred))

    return  mse_0_u + mse_b_u + mse_f_u, mse_0_u, mse_f_u

model=ISA_PINN_2D("data.mat",layer_sizes,tf_iter1,newton_iter1,f_model=f_model,Loss=loss,N_f= n_f)
model.fit()

Exact_u=model.u_star
X,Y, T = np.meshgrid(model.x,model.y,model.t)

N_x= 103
N_y= 105
N_t= 401
Exact_u = Exact_u.reshape(N_y, N_x, N_t)

X= 4*np.pi
Y= 4*np.pi
T= 40

xx= np.linspace(0, X, N_x)
yy= np.linspace(0, Y, N_y)
tt= np.linspace(0, T, N_t)
Ntinter = 10

u_pred = np.zeros((N_y, N_x, Ntinter))
f_u_pred = np.zeros((N_y, N_x, Ntinter))

dN_t = round(N_t/Ntinter)

for i in range(dN_t):
   t = tt[Ntinter*i:Ntinter*(i+1)]
   if (i == dN_t):
      t= tt[-1]

   x = xx
   y = yy

   tmp_u_pred, tmp_f_u_pred = model.predict(x,y,t)
   if(i==0):
      tmp_u_pred = tmp_u_pred.reshape(N_y, N_x, Ntinter)
      tmp_f_u_pred = tmp_f_u_pred.reshape(N_y, N_x, Ntinter)
      u_pred = tmp_u_pred
      f_u_pred= tmp_f_u_pred

      Exact_u_i = Exact_u[:,:, Ntinter*i:Ntinter*(i+1)]
      perror_u = np.linalg.norm((Exact_u_i - tmp_u_pred).flatten(),2)
      perror_uEx = np.linalg.norm(Exact_u_i.flatten(),2)

   elif(i!=dN_t):
      tmp_u_pred = tmp_u_pred.reshape(N_y, N_x, Ntinter)
      tmp_f_u_pred = tmp_f_u_pred.reshape(N_y, N_x, Ntinter)

      u_pred = np.concatenate((u_pred, tmp_u_pred), axis=2)
      f_u_pred = np.concatenate((f_u_pred, tmp_f_u_pred), axis=2)

      Exact_u_i = Exact_u[:,:, Ntinter*i:Ntinter*(i+1)]
      perror_u += np.linalg.norm((Exact_u_i - tmp_u_pred).flatten(),2)
      perror_uEx += np.linalg.norm(Exact_u_i.flatten(),2)

   else:
      tmp_u_pred = tmp_u_pred.reshape(N_y, N_x, 1)
      tmp_f_u_pred = tmp_f_u_pred.reshape(N_y, N_x, 1)
      u_pred = np.concatenate((u_pred, tmp_u_pred), axis=2)
      f_u_pred = np.concatenate((f_u_pred, tmp_f_u_pred), axis=2)

      Exact_u_i = Exact_u[:,:, Ntinter*i:Ntinter*(i+1)]
      perror_u += np.linalg.norm((Exact_u_i - tmp_u_pred).flatten(),2)
      perror_uEx += np.linalg.norm(Exact_u_i.flatten(),2)


X,Y, T = np.meshgrid(model.x,model.y,model.t)
X_star = model.X_star
u_star = model.u_star

tb=model.t
x=model.x
t=model.t

error_u = perror_u/perror_uEx
print('Error u: %e' % (error_u))

ferru = np.mean(np.absolute(f_u_pred))
print('ferru = ', ferru)

