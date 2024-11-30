import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
import time
import scipy.io
from scipy.interpolate import griddata
from pyDOE import lhs
import pickle
import os
from globalf import *
from ISA_PINN_2D import ISA_PINN_2D

#torch.manual_seed(1234)
#np.random.seed(1234)

def u_x_model(model, x, y, t):
    x = x.requires_grad_(True).float().to(device)
    y = y.requires_grad_(True).float().to(device)
    t = t.requires_grad_(True).float().to(device)

    u = model(torch.cat([x, y, t], dim=1))
    return u

def ensure_grad(tensors):
    for tensor in tensors:
        if not tensor.requires_grad:
            tensor.requires_grad = True

def f_model(model, x, y, t):
    inputs = torch.cat([x, y, t], dim=1)
    u = model(inputs)

    # Calculate gradients
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    u_xxx = torch.autograd.grad(u_xx, x, grad_outputs=torch.ones_like(u_xx), create_graph=True)[0]
    u_xt = torch.autograd.grad(u_x, t, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    u_xy = torch.autograd.grad(u_x, y, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xxxy = torch.autograd.grad(u_xxx, y, grad_outputs=torch.ones_like(u_xxx), create_graph=True)[0]

    f = u_xt - 4 * u_xy * u_x - 2 * u_xx * u_y - u_xxxy

    return f

def loss(model, x_f, y_f, t_f, t0, u0, x_lb, x_ub, y_lb, y_ub, u_x_lb, u_x_ub, u_y_lb, u_y_ub, XY, XT, YT, col_weights, u_weights, ub_weights):
    x_f, y_f, t_f, t0, u0, x_lb, x_ub, y_lb, y_ub, u_x_lb, u_x_ub, u_y_lb, u_y_ub, XY, XT, YT = \
        [tensor.float().to(device) for tensor in [x_f, y_f, t_f, t0, u0, x_lb, x_ub, y_lb, y_ub, u_x_lb, u_x_ub, u_y_lb, u_y_ub, XY, XT, YT]]
    
    f_u_pred = f_model(model, x_f, y_f, t_f)

    u0_pred = u_x_model(model, XY[:, 0:1], XY[:, 1:2], t0)
    u_x_lb_pred = u_x_model(model, x_lb, YT[:, 0:1], YT[:, 1:2])
    u_x_ub_pred = u_x_model(model, x_ub, YT[:, 0:1], YT[:, 1:2])
    u_y_lb_pred = u_x_model(model, XT[:, 0:1], y_lb, XT[:, 1:2])
    u_y_ub_pred = u_x_model(model, XT[:, 0:1], y_ub, XT[:, 1:2])

    mse_0_u = torch.mean((u_weights * (u0 - u0_pred)) ** 2)
    mse_b_u = torch.mean((ub_weights * (u_x_lb_pred - u_x_lb)) ** 2) + \
              torch.mean((ub_weights * (u_x_ub_pred - u_x_ub)) ** 2) + \
              torch.mean((ub_weights * (u_y_ub_pred - u_y_ub)) ** 2) + \
              torch.mean((ub_weights * (u_y_lb_pred - u_y_lb)) ** 2)
    mse_f_u = torch.mean((col_weights * f_u_pred) ** 2)

    return mse_0_u + mse_b_u + mse_f_u, mse_0_u, mse_b_u, mse_f_u

for id_t in range(1):
    # Initialize and fit the model
    model = ISA_PINN_2D("data.mat", layer_sizes, tf_iter1, newton_iter1, f_model=f_model, Loss=loss, N_f=n_f)
    model.fit()
    model.fit_lbfgs()

    Exact_u = model.u_star
    X_star = model.X_star
    u_star = model.u_star

    Ntinter = 10
    N_t= model.t.shape[0]
    N_x= model.x.shape[0]
    N_y= model.y.shape[0]
    print('N_t= ', N_t)

    u_pred = np.zeros((N_y, N_x, Ntinter))
    f_u_pred = np.zeros((N_y, N_x, Ntinter))
    
    dN_t = round(N_t/Ntinter)
    
    for i in range(dN_t+1):
       if (i == dN_t):
         t= model.t[-1]
       else:
         t = model.t[Ntinter*i:Ntinter*(i+1)]
    
       tmp_u_pred, tmp_f_u_pred = model.predict(model.x, model.y, t)
       if(i==0):
          tmp_u_pred = tmp_u_pred.reshape(N_y, N_x, Ntinter)
          tmp_f_u_pred = tmp_f_u_pred.reshape(N_y, N_x, Ntinter)
          u_pred = tmp_u_pred
          f_u_pred= tmp_f_u_pred
    
       elif(i!=dN_t):
          tmp_u_pred = tmp_u_pred.reshape(N_y, N_x, Ntinter)
          tmp_f_u_pred = tmp_f_u_pred.reshape(N_y, N_x, Ntinter)
          u_pred = np.concatenate((u_pred, tmp_u_pred), axis=2)
          f_u_pred = np.concatenate((f_u_pred, tmp_f_u_pred), axis=2)
    
       else:
          tmp_u_pred = tmp_u_pred.reshape(N_y, N_x, 1)
          tmp_f_u_pred = tmp_f_u_pred.reshape(N_y, N_x, 1)
          u_pred = np.concatenate((u_pred, tmp_u_pred), axis=2)
          f_u_pred = np.concatenate((f_u_pred, tmp_f_u_pred), axis=2)

    Exact_u = Exact_u.reshape(N_y, N_x, N_t)

    U3_pred = u_pred.reshape((N_y, N_x, N_t))
    f_U3_pred = f_u_pred.reshape((N_y, N_x, N_t))

    u_pred_tmp= u_pred.flatten()[:,None]
    error_u = np.linalg.norm(u_star-u_pred_tmp,2)/np.linalg.norm(u_star,2)
    print('Error u: %e' % (error_u))

    ferru = np.mean(np.absolute(f_u_pred))
    print('ferru = ', ferru)

