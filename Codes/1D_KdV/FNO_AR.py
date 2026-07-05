#!/usr/bin/env python
# coding: utf-8

#Importing all the necessary libraries
import os
import sys
import pickle
from tqdm import tqdm

import jax
import jax.numpy as jnp
import numpy as np
from scipy.io import loadmat

import flax
from flax import linen as nn

import optax
import matplotlib.pyplot as plt
import matplotlib
from typing import Callable, List
import scipy

from models import FNO1d
from utils import save_model_params, load_model_params
from utils import dataloader

# seed = 42
seed = np.random.choice(np.arange(99999), size = 1, replace=True)[0]
print("Seed: ",seed)
np.random.seed(seed)
key = jax.random.PRNGKey(seed)

base_path = "/home/dnayak2/data_sgoswam4/Dibya/Datasets/1D_KdV"
data = jnp.load(base_path + "/data_kdv.npz")
u = data['u']  #Initial condition
xt = data['xt']  #grid
g_u = data['g_u']  #Output
ns = 1000
nx = 100
nt = 200
g_u = g_u.reshape(ns, nt, nx)
print(f"u: {u.shape}, xt: {xt.shape}, g_u: {g_u.shape}")

output = g_u
Ns, Nt, Nx = output.shape
print(f"Ns: {Ns}, Nt: {Nt}, Nx: {Nx}")

tt = int(Nt//2)

#Creating the input and output training data
init_timestep = 0
end_timestep = tt

input_data_NN = output[:,init_timestep,:]
output_data_NN = output[:,init_timestep+1,:]

for i in range(init_timestep+1, end_timestep):
    input_data_NN = jnp.vstack((input_data_NN, output[:,i,:]))
    output_data_NN = jnp.vstack((output_data_NN, output[:,i+1,:]))

print("After arrangement of data for t -> t+1 mapping, we have:")
print(f"Inputs: {input_data_NN.shape}, Outputs: {output_data_NN.shape}")

#Create the mesh
print("Creating the mesh (x-direction)")
mesh = jnp.linspace(0, 1, Nx)

print("Tiling it for all samples and timesteps")
mesh_tensor_repeated = jnp.tile(mesh[None, None, :], (input_data_NN.shape[0], 1, 1))
print(f"Mesh tensor repeated shape: {mesh_tensor_repeated.shape}")

#Concatenate to original input-output solution fields
input_data_NN_mod = jnp.concatenate([input_data_NN[:,None,:], mesh_tensor_repeated], axis=1)
output_data_NN_mod = output_data_NN[:,None,:]
print(f"Inputs to FNO: {input_data_NN_mod.shape}")
print(f"Outputs from FNO: {output_data_NN_mod.shape}")

#Free up some memory
print("Freeing up some memory")
del input_data_NN, output_data_NN, mesh_tensor_repeated

#Separate into train and test datasets
Ntrain = int(0.8*Ns)
perm = jax.random.permutation(jax.random.PRNGKey(0), Ns)

train_idx = perm[:Ntrain]
test_idx = perm[Ntrain:]

train_x = jnp.take(input_data_NN_mod, train_idx, axis=0)
test_x = jnp.take(input_data_NN_mod, test_idx, axis=0)

train_y = jnp.take(output_data_NN_mod, train_idx, axis=0)
test_y = jnp.take(output_data_NN_mod, test_idx, axis=0)

print(f"train_x shape: {train_x.shape}, train_y shape: {train_y.shape}")
print(f"test_x shape: {test_x.shape}, test_y shape: {test_y.shape}")

#Stop gradients for test_x and test_y
# test_x = jax.lax.stop_gradient(test_x)
# test_y = jax.lax.stop_gradient(test_y)

#Create the FNO-1D model object
fno = FNO1d(in_channels = 2,
            out_channels = 1,
            modes = 24,
            width = 64,
            activation = jax.nn.gelu,
            n_blocks = 6
)
print(f"FNO model: {fno}")
model_fn = jax.jit(fno.apply)

#Instantiate the model params
params = fno.init(jax.random.PRNGKey(seed), train_x[0:1])    #Earlier seed = 42

@jax.jit
def loss_fn(params, x, y):
    y_pred = model_fn(params, x)
    loss = jnp.mean((y_pred - y) ** 2)
    return loss

@jax.jit
def make_step(params, opt_state, x, y):
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss


lr = 1e-3
lr_scheduler = optax.schedules.exponential_decay(init_value=lr, transition_steps=2000, decay_rate=0.96)
optimizer = optax.adam(lr_scheduler)
opt_state = optimizer.init(params)

loss_history = []
val_loss_history = []
# batch_size = 128
batch_size = 256    #This is only set to access the compute speed of training
shuffle_key = jax.random.PRNGKey(seed)     #Earlier seed = 80
epochs = int(3e4)
min_val_loss = jnp.inf

result_dir = "./FNO_Autoregressive"
filename = f"best_model_params_seed{seed}.pkl"

print("Starting training...")
for epoch in range(epochs):
    shuffle_key, subkey = jax.random.split(shuffle_key)
    total_loss = 0
    nbatches = 0
    for batch_x, batch_y in tqdm(dataloader(subkey, train_x, train_y, batch_size),
                                    desc=f"Epoch: {epoch}"):
        params, opt_state, loss = make_step(params, opt_state, batch_x, batch_y)
        total_loss += loss
        nbatches += 1
    
    loss = total_loss/nbatches
    val_loss = loss_fn(params, test_x, test_y)
    
    #Save the best model
    if val_loss < min_val_loss:
        best_params = params
        min_val_loss = val_loss
        # save_model_params(best_params, result_dir, filename = filename)
    
    loss_history.append(loss)
    val_loss_history.append(val_loss)
    
    if epoch%500==0:
        print(f"Epoch: {epoch}, Train loss: {loss}, Val loss: {val_loss}, Min Val loss: {min_val_loss}") 
print("Training completed.")

plt.figure(dpi = 130)
plt.semilogy(np.arange(epoch+1), loss_history, label = "Train loss")
plt.semilogy(np.arange(epoch+1), val_loss_history, label = "Test loss")

plt.xlabel("Epochs")
plt.ylabel("Loss")

plt.tick_params(which = 'major', axis = 'both', direction = 'in', length = 6)
plt.tick_params(which = 'minor', axis = 'both', direction = 'in', length = 3.5)
plt.minorticks_on()

plt.grid(alpha = 0.3)
plt.legend(loc = 'best')
# plt.savefig(result_dir + "/loss_plot.jpeg", dpi = 800)
plt.show()

#Save the loss arrays
save = False
if save:
    np.save(result_dir + "/Train_loss.npy",loss_history)
    np.save(result_dir + "/Test_loss.npy",val_loss_history)

def create_input(x):
    mesh = jnp.linspace(0, 1, Nx)
    mesh_tensor_repeated = jnp.tile(mesh[None, None, :], (x.shape[0], 1, 1))
    x_with_mesh = jnp.concatenate([x[:,None,:], mesh_tensor_repeated], axis=1)
    return x_with_mesh    #(Ns, Nc=2, Nx)


def run_inference(initial_u, n_steps):
    u_states = np.zeros_like(output)  # List to store the states over time
    u_states[:,0,:] = initial_u
    
    # Initialize the previous state (this could be your u_0 and u_1, etc.)
    u_curr = initial_u  # Set the current state to the initial state
    
    for i in range(1, n_steps):
        
        u_curr_in_FNO = create_input(u_curr)    #(Ns, in_channels, Nx)
        u_next_out_FNO = model_fn(best_params, u_curr_in_FNO)   #(Ns, out_channels, Nx)
        u_next = u_next_out_FNO[:, 0, :]
        
        # Append the predicted state to the list
        u_states[:, i, :] = u_next
        
        # Update previous and current states for the next step
        u_curr = u_next
    
    return u_states

############################
#----PERFORM INFERENCE-----#
############################

print("Moving on to the inference stage!")
# Load the best model parameters
import time
best_params = load_model_params(result_dir, filename = filename)
print(f"Best model params loaded: {filename}")

print("Reading ground truth data")
base_path = "/home/dnayak2/data_sgoswam4/Dibya/Datasets/1D_KdV"
data = jnp.load(base_path + "/data_kdv.npz")
u = data['u']  #Initial condition
xt = data['xt']  #grid
g_u = data['g_u']  #Output
ns = 1000
nx = 100
nt = 200
g_u = g_u.reshape(ns, nt, nx)
print(f"u: {u.shape}, xt: {xt.shape}, g_u: {g_u.shape}")

output = g_u
Ns, Nt, Nx = output.shape
u_curr = output[:, 0, :]

start_time = time.time()
u_pred = run_inference(u_curr, n_steps=Nt)
end_time = time.time()
print(f"Total inferencing time for {u_pred.shape[0]} samples: {end_time-start_time}")
print(f"u_pred shape: {u_pred.shape}, output: {output.shape}")

#Compute overall relative L2 error
overall_rel_l2_err = jnp.linalg.norm(u_pred-output)/jnp.linalg.norm(output)
print(f"Overall relative L2 error: {overall_rel_l2_err}")

#Find the autoregressive errors
auto_reg_error = []

for i in range(Nt):
    err = np.linalg.norm(u_pred[:,i,:] - output[:,i,:])/np.linalg.norm(output[:,i,:])
    auto_reg_error.append(err)

print("----Extrapolation errors----")
#Compute statistics
t = [120, 140, 160, 200]

for t_idx in t:
    print(f"t: {t_idx}, L2 error: {auto_reg_error[t_idx-1]}")

#Save all the relevant arrays
save=False
if save:
    np.save(result_dir + "/u_pred.npy", u_pred)
    np.save(result_dir + "/auto_reg_err_FNO_AR.npy", auto_reg_error)
print("Program executed successfully!")