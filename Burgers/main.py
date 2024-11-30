import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import numpy as np
import time
import scipy.io
from scipy.interpolate import griddata
from pyDOE import lhs
import pickle
import os
from globalf import *
from ISA_PINN import ISA_PINN

def u_x_model(u_model, x, t):
    x = x.requires_grad_(True)
    t = t.requires_grad_(True)
    
    u = u_model(torch.cat([x, t], dim=1))
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]

    return u, u_x

# Define the loss function
def loss(model, x_f_batch, t_f_batch, x0, t0, u0, u_lb, u_ub, x_lb, t_lb, x_ub, t_ub, col_weights, u_weights):
    u0_pred = model(torch.cat([x0, t0], 1))
    u_lb_pred, u_x_lb_pred = u_x_model(model, x_lb, t_lb)
    u_ub_pred, u_x_ub_pred = u_x_model(model, x_ub, t_ub)
    f_u_pred = f_model(model, x_f_batch, t_f_batch)
    
    mse_0_u = torch.mean((u_weights * (u0 - u0_pred))**2)
    mse_b_u = torch.mean((u_lb_pred - u_lb)**2) + torch.mean((u_ub_pred - u_ub)**2)
    mse_f_u = torch.mean((col_weights * f_u_pred)**2)
    
    return mse_0_u + mse_b_u + mse_f_u, mse_0_u, mse_b_u, mse_f_u

def ensure_grad(tensors):
    for tensor in tensors:
        if not tensor.requires_grad:
            tensor.requires_grad = True

def f_model(model, x, t):
    # Compute derivatives
    ensure_grad([x, t])
    u = model(torch.cat([x, t], 1))
    u_t = autograd.grad(u, t, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
    u_x = autograd.grad(u, x, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
    u_tx = autograd.grad(u_x, t, grad_outputs=torch.ones_like(u_x), retain_graph=True, create_graph=True)[0]
    u_xx = autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), retain_graph=True, create_graph=True)[0]
    
    # Constants
    nu = 0.01/np.pi
    
    # Compute f
    f = u_t + u * u_x - nu * u_xx
    
    return f

def main():
    model=ISA_PINN("burgers_shock.mat",layer_sizes,tf_iter1,newton_iter1,f_model=f_model,Loss=loss,N_f= n_f)

    model.fit()
    model.fit_lbfgs()
    
    u_pred, f_u_pred = model.predict()
    
    U3_pred = u_pred.reshape((Nt, Nx)).T
    f_U3_pred = f_u_pred.reshape((Nt, Nx)).T

    perror_u  = np.linalg.norm((model.Exact_u - U3_pred),2)
    perror_uEx = np.linalg.norm(model.Exact_u,2)
 
    error_u = perror_u/perror_uEx
    print('Error u: %e' % (error_u))
 
if __name__ == '__main__':
    main()
