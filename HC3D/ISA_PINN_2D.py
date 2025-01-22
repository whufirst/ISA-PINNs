import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.io
import math
import matplotlib.gridspec as gridspec
import pickle
import os
import datetime
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import layers, activations
from scipy.interpolate import griddata
from pyDOE import lhs
from globalf import *

class dummy(object):
  pass

class Struct(dummy):
  def __getattribute__(self, key):
    if key == '__dict__':
      return super(dummy, self).__getattribute__('__dict__')
    return self.__dict__.get(key, 0)
    
class ISA_PINN_2D:
    @tf.function
    def uv_model(self,x,y,t):
        uv=self.model(tf.concat([x,y,t],1))
        u=uv[:,0:1]
        return u

    def __init__(self, mat_filename, layers: [], tf_iter: int, newton_iter: int, f_model, ux_model=None, Loss=..., lbfgs_lr=0.8, N_f=10000, checkPointPath="./checkPoint"):
        self.N_f=N_f
        self.__Loadmat(mat_filename)
        self.layers=layers
        self.sizes_w=[]
        self.sizes_b=[]
        self.lbfgs_lr=lbfgs_lr

        for i, width in enumerate(layers):
            if i != 1:
                self.sizes_w.append(int(width * layers[1]))
                self.sizes_b.append(int(width if i != 0 else layers[1]))

        col_weights1 = tf.Variable(tf.reshape(tf.repeat(100.0, N_f),(N_f, -1)))
        u_weights1 = tf.Variable(tf.random.uniform([self.u0.shape[0], 1]))
        u_weights2 = tf.Variable(tf.random.uniform([self.u_x_lb.shape[0], 1]))
        col_weights = tf.cast(col_weights1, dtype=tfdoubstr)
        u_weights = tf.cast(u_weights1, dtype=tfdoubstr)
        ub_weights = tf.cast(u_weights2, dtype=tfdoubstr)
        self.col_weights=tf.Variable(col_weights)
        self.u_weights=tf.Variable(u_weights)
        self.ub_weights=tf.Variable(ub_weights)
        self.SA_weights={"u_weights":self.u_weights,"ub_weights":self.ub_weights,"col_weights":self.col_weights}
        self.model=self.__neural_net(self.layers)
        self.model.summary()
        self.Loss=Loss
        self.tf_iter=tf_iter
        self.newton_iter=newton_iter
        self.f_model=f_model
        self.ux_model=ux_model
        self.checkPointPath=checkPointPath
        if not os.path.exists(self.checkPointPath):
            os.makedirs(self.checkPointPath)

    def __set_weights(self,model, w, sizes_w, sizes_b):
        for i, layer in enumerate(model.layers[0:]):
            start_weights = sum(sizes_w[:i]) + sum(sizes_b[:i])
            end_weights = sum(sizes_w[:i+1]) + sum(sizes_b[:i])
            weights = w[start_weights:end_weights]
            w_div = int(sizes_w[i] / sizes_b[i])
            weights = tf.reshape(weights, [w_div, sizes_b[i]])
            biases = w[end_weights:end_weights + sizes_b[i]]
            weights_biases = [weights, biases]
            layer.set_weights(weights_biases)
    
    def get_weights(self,model):
        w = []
        for layer in model.layers[0:]:
            weights_biases = layer.get_weights()
            weights = weights_biases[0].flatten()
            biases = weights_biases[1]
            w.extend(weights)
            w.extend(biases)

        w = tf.convert_to_tensor(w)
        return w

    def __neural_net(self,layer_sizes):
        model = Sequential()
        model.add(layers.InputLayer(input_shape=(layer_sizes[0],)))
        for width in layer_sizes[1:-1]:
            model.add(layers.Dense(
                width, activation=tf.nn.tanh,
                kernel_initializer="glorot_normal"))
        model.add(layers.Dense(
            layer_sizes[-1], activation=None,
            kernel_initializer="glorot_normal"))
        return model

    def fit(self):
        batch_sz = self.N_f
        n_batches =  self.N_f // batch_sz
        start_time = time.time()
        tf_optimizer = tf.keras.optimizers.Adam(lr = 0.005, beta_1=.90)
        optimizers=[]
        for i in range(len(self.SA_weights)):
            optimizers.append(tf.keras.optimizers.Adam(lr = 0.005, beta_1=.90))

        print("Starting Adam training")

        for epoch in range(self.tf_iter):
            for i in range(n_batches):
                loss_value,mse_0, mse_f, grads, SA_grads = self.__grad(self.x_f,self.y_f,self.t_f,self.u0,self.u_x_lb,self.u_x_ub,self.u_y_lb,self.u_y_ub,self.XY,self.XT,self.YT,self.SA_weights)

                tf_optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
                a=0
                for key in self.SA_weights:
                    optimizers[a].apply_gradients(zip([-SA_grads[a]],[self.SA_weights[key]]))
                    a+=1
                a=0


            if (epoch+1) % 100 == 0:
                elapsed = time.time() - start_time
                print('It: %d, Time: %.2f, mse_0: %.4e, mse_f: %.4e, total loss: %.4e' % (epoch+1, elapsed, mse_0, mse_f, loss_value))
                start_time = time.time()
        
        print("Starting L-BFGS training")

        loss_and_flat_grad = self.__get_loss_and_flat_grad(self.x_f,self.y_f,self.t_f,self.u0,self.u_x_lb,self.u_x_ub,self.u_y_lb,self.u_y_ub,self.XY,self.XT,self.YT,self.SA_weights)
        self.lbfgs(self.checkPointPath,self.model,loss_and_flat_grad,
        self.get_weights(self.model),
        Struct(), maxIter=self.newton_iter, learningRate=self.lbfgs_lr)

    def __grad(self, x_f_batch,y_f_batch, t_f_batch, u0,u_x_lb,u_x_ub,u_y_lb,u_y_ub, XY, XT, YT, SA_weights):
        with tf.GradientTape(persistent=True) as tape:
            loss_value, mse_0, mse_f = self.Loss(self,x_f_batch,y_f_batch, t_f_batch, u0,u_x_lb,u_x_ub,u_y_lb,u_y_ub, XY, XT, YT, SA_weights)
            grads = tape.gradient(loss_value, self.model.trainable_variables)
            SA_grads=[]
            for key in SA_weights:
                SA_grads.append(tape.gradient(loss_value,SA_weights[key]))

        return loss_value, mse_0, mse_f, grads, SA_grads

    def __get_loss_and_flat_grad(self,x_f_batch,y_f_batch, t_f_batch, u0,u_x_lb,u_x_ub,u_y_lb,u_y_ub, XY, XT, YT, SA_weights):
        def loss_and_flat_grad(w):
            with tf.GradientTape() as tape:
                self.__set_weights(self.model, w, self.sizes_w, self.sizes_b)
                loss_value, _, _ = self.Loss(self,x_f_batch,y_f_batch, t_f_batch, u0,u_x_lb,u_x_ub,u_y_lb,u_y_ub, XY, XT, YT, SA_weights)
            grad = tape.gradient(loss_value, self.model.trainable_variables)
            grad_flat = []
            for g in grad:
                grad_flat.append(tf.reshape(g, [-1]))
            grad_flat = tf.concat(grad_flat, 0)
            return loss_value, grad_flat

        return loss_and_flat_grad

    def __Loadmat(self, fileName):
        data = scipy.io.loadmat(fileName)

        t = data['t'].flatten()[:,None]
        x = data['x'].flatten()[:,None]
        y = data['y'].flatten()[:,None]
        self.x=x
        self.y=y
        self.t=t
        ub=np.array([x.min(0),y.min(0),t.min(0)]).T
        lb=np.array([x.max(0),y.max(0),t.max(0)]).T
        X_f = lb + ((ub-lb)*lhs(3, self.N_f))

        XY_X,XY_Y=np.meshgrid(x,y)
        XT_X,XT_T=np.meshgrid(x,t)
        YT_Y,YT_T=np.meshgrid(y,t)
        XY=np.hstack((XY_X.flatten()[:, None], XY_Y.flatten()[:, None]))
        XT=np.hstack((XT_X.flatten()[:, None], XT_T.flatten()[:, None]))
        YT=np.hstack((YT_Y.flatten()[:, None], YT_T.flatten()[:, None]))

        selected_indices = np.random.choice(XY.shape[0], N0, replace=False)
        selected_indicesxt = np.random.choice(XT.shape[0], Nb, replace=False)
        selected_indicesyt = np.random.choice(YT.shape[0], Nb, replace=False)
        XY= XY[selected_indices, :]
        YT= YT[selected_indicesyt, :]
        XT= XT[selected_indicesxt, :]

        u0all=data['Exact_u0'].flatten()[:,None]
        u0= u0all[selected_indices]
        u_x_lball =data['Exact_u_x_lb'].flatten()[:,None]
        u_x_uball =data['Exact_u_x_ub'].flatten()[:,None]
        u_y_lball =data['Exact_u_y_lb'].flatten()[:,None]
        u_y_uball =data['Exact_u_y_ub'].flatten()[:,None]

        u_x_lb= u_x_lball[selected_indicesyt]
        u_x_ub= u_x_uball[selected_indicesyt]
        u_y_lb= u_y_lball[selected_indicesxt]
        u_y_ub= u_y_uball[selected_indicesxt]

        self.u0=tf.cast(u0,dtype=tfdoubstr)
        self.u_x_lb=tf.cast(u_x_lb,dtype=tfdoubstr)
        self.u_x_ub=tf.cast(u_x_ub,dtype=tfdoubstr)
        self.u_y_lb=tf.cast(u_y_lb,dtype=tfdoubstr)
        self.u_y_ub=tf.cast(u_y_ub,dtype=tfdoubstr)
        self.XY=tf.convert_to_tensor(XY,dtype=tfdoubstr)
        self.XT=tf.convert_to_tensor(XT,dtype=tfdoubstr)
        self.YT=tf.convert_to_tensor(YT,dtype=tfdoubstr)
        self.x_f=tf.convert_to_tensor(X_f[:,0:1],dtype=tfdoubstr)
        self.y_f=tf.convert_to_tensor(X_f[:,1:2],dtype=tfdoubstr)
        self.t_f=tf.convert_to_tensor(X_f[:,2:3],dtype=tfdoubstr)
        self.t0=tf.cast(tf.reshape(tf.repeat(t.min(0),XY[:,1:2].shape[0]),(XY[:,1:2].shape[0],-1)),dtype=tfdoubstr)
        self.x_lb=tf.cast(tf.reshape(tf.repeat(x.min(0),YT[:,1:2].shape[0]),(YT[:,1:2].shape[0],-1)),dtype=tfdoubstr)
        self.x_ub=tf.cast(tf.reshape(tf.repeat(x.max(0),YT[:,1:2].shape[0]),(YT[:,1:2].shape[0],-1)),dtype=tfdoubstr)
        self.y_lb=tf.cast(tf.reshape(tf.repeat(y.min(0),XT[:,1:2].shape[0]),(XT[:,1:2].shape[0],-1)),dtype=tfdoubstr)
        self.y_ub=tf.cast(tf.reshape(tf.repeat(y.max(0),XT[:,1:2].shape[0]),(XT[:,1:2].shape[0],-1)),dtype=tfdoubstr)


        X,Y,T=np.meshgrid(x,y,t)
        self.X_star=np.hstack((X.flatten()[:, None],Y.flatten()[:, None],T.flatten()[:, None]))
        self.u_star=data["Exact"].flatten()[:,None]

    def predict(self, x, y, t):
        X,Y,T=np.meshgrid(x,y,t)
        XX_star=np.hstack((X.flatten()[:, None],Y.flatten()[:, None],T.flatten()[:, None]))
        X_star = tf.convert_to_tensor(XX_star, dtype=tfdoubstr)
        u_star= self.uv_model(X_star[:,0:1],
                        X_star[:,1:2],X_star[:,2:3])

        f_u_star = self.f_model(self,X_star[:,0:1],
                    X_star[:,1:2],X_star[:,2:3])

        return u_star.numpy(), f_u_star.numpy()

    def dot(self, a, b):
      return tf.reduce_sum(a*b)
    
    def verbose_func(self, s):
      print(s)
    
    final_loss = None
    times = []
    def lbfgs(self, fileName,u_model,opfunc, x, state, maxIter = 100, learningRate = 1, do_verbose = True, id_t=0):
      """port of lbfgs.lua, using TensorFlow eager mode.
      """
    
      global final_loss, times
      
      maxEval = maxIter*1.25
      tolFun = 1e-7
      tolX = 1e-11
      nCorrection = 50
      isverbose = False
    
      # verbose function
      if isverbose:
        verbose = self.verbose_func
      else:
        verbose = lambda x: None
        
      f, g = opfunc(x)
    
      f_hist = [f]
      currentFuncEval = 1
      state.funcEval = state.funcEval + 1
      p = g.shape[0]
    
      # check optimality of initial point
      tmp1 = tf.abs(g)
      if tf.reduce_sum(tmp1) <= tolFun:
        verbose("optimality condition below tolFun")
        return x, f_hist
    
      # optimize for a max of maxIter iterations
      nIter = 0
      times = []
    
      loss_l = []
    
      lstart_time = time.time()
      while nIter < maxIter:
        
        # keep track of nb of iterations
        nIter = nIter + 1
        state.nIter = state.nIter + 1
    
        ############################################################
        ## compute gradient descent direction
        ############################################################
        if state.nIter == 1:
          d = -g
          old_dirs = []
          old_stps = []
          Hdiag = 1
        else:
          # do lbfgs update (update memory)
          y = g - g_old
          s = d*t
          ys = self.dot(y, s)
          
          if ys > 1e-10:
            # updating memory
            if len(old_dirs) == nCorrection:
              # shift history by one (limited-memory)
              del old_dirs[0]
              del old_stps[0]
    
            # store new direction/step
            old_dirs.append(s)
            old_stps.append(y)
    
            # update scale of initial Hessian approximation
            Hdiag = ys/self.dot(y, y)
    
          # compute the approximate (L-BFGS) inverse Hessian 
          # multiplied by the gradient
          k = len(old_dirs)
    
          # need to be accessed element-by-element, so don't re-type tensor:
          ro = [0]*nCorrection
          for i in range(k):
            ro[i] = 1/self.dot(old_stps[i], old_dirs[i])
            
    
          # iteration in L-BFGS loop collapsed to use just one buffer
          # need to be accessed element-by-element, so don't re-type tensor:
          al = [0]*nCorrection
    
          q = -g
          for i in range(k-1, -1, -1):
            al[i] = self.dot(old_dirs[i], q) * ro[i]
            q = q - al[i]*old_stps[i]
    
          # multiply by initial Hessian
          r = q*Hdiag
          for i in range(k):
            be_i = self.dot(old_stps[i], r) * ro[i]
            r += (al[i]-be_i)*old_dirs[i]
            
          d = r
          # final direction is in r/d (same object)
    
        g_old = g
        f_old = f
        
        ############################################################
        ## compute step length
        ############################################################
        # directional derivative
        gtd = self.dot(g, d)
    
        # check that progress can be made along that direction
        if gtd > -tolX:
          verbose("Can not make progress along direction.")
          break
    
        # reset initial guess for step size
        if state.nIter == 1:
          tmp1 = tf.abs(g)
          t = min(1, 1/tf.reduce_sum(tmp1))
        else:
          t = learningRate
    
        x += t*d
    
        if nIter != maxIter:
        # re-evaluate function only if not in last iteration
        # the reason we do this: in a stochastic setting,
        # no use to re-evaluate that function here
          f, g = opfunc(x)
    
        lsFuncEval = 1
        f_hist.append(f)
    
    
        # update func eval
        currentFuncEval = currentFuncEval + lsFuncEval
        state.funcEval = state.funcEval + lsFuncEval
    
        ############################################################
        ## check conditions
        ############################################################
        if nIter == maxIter:
          break
    
        if currentFuncEval >= maxEval:
          # max nb of function evals
          print('max nb of function evals')
          break
    
        tmp1 = tf.abs(g)
        if tf.reduce_sum(tmp1) <=tolFun:
          # check optimality
          print('optimality condition below tolFun')
          break
        
        tmp1 = tf.abs(d*t)
        if tf.reduce_sum(tmp1) <= tolX:
          # step size below tolX
          print('step size below tolX')
          break
        
        if tf.abs(f,f_old) < tolX:
          # function value changing less than tolX
          print('function value changing less than tolX'+str(tf.abs(f-f_old)))
          break
    
        if do_verbose:
          if (nIter+1) % 100 == 0:
            elapsed = time.time() - lstart_time
            print("Step: %3d Time: %.2f loss: %.4e "%(nIter+1, elapsed, f.numpy()))
            lstart_time = time.time()
    
        if nIter == maxIter - 1:
          final_loss = f.numpy()
    
      # save state
      state.old_dirs = old_dirs
      state.old_stps = old_stps
      state.Hdiag = Hdiag
      state.g_old = g_old
      state.f_old = f_old
      state.t = t
      state.d = d
    
      return x, f_hist, currentFuncEval
   
