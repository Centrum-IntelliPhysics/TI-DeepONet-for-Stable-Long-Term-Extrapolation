#!/usr/bin/env python
# coding: utf-8

#importing all necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.io
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import argparse
import torch
import jax
import jax.numpy as jnp
from jax import random
from jax import jit, vmap, pmap, grad, value_and_grad
from utils_jax import *

from tqdm import tqdm

import flax
import flax.linen as nn
import optax

from typing import Callable, Tuple, List, Dict, Optional, Any, Sequence

from sklearn.model_selection import train_test_split

from functools import partial

import os
import sys
import pickle

#Read arguments from command line
parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', type=str, required=True)
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()

seed = args.seed
# seed = np.random.choice(np.arange(99999), size=1, replace=True)[0]
np.random.seed(seed)
key = jax.random.PRNGKey(seed)
print(f"Seed set with value = {seed}")

base_path = "/home/dnayak2/scr4_sgoswam4/Dibya/backup/Datasets/3D_Heat_Conduction/"

outputs = jnp.load(os.path.join(base_path, "processed/3d_heat_field_results.npz"))['heat_field']

Ns, Nt, Nx, Ny, Nz = outputs.shape
print(f"Ns: {Ns}, Nt: {Nt}, Nx: {Nx}, Ny: {Ny}, Nz: {Nz}")

inputs = outputs[:, 0, :, :]
print(f"Inputs shape: {inputs.shape}, Outputs shape: {outputs.shape}")

#Get the xspan and yspan
xspan = jnp.linspace(0, 1, Nx)
yspan = jnp.linspace(0, 1, Ny)
zspan = jnp.linspace(0, 1, Nz)
tspan = jnp.linspace(0, 1, Nt)

#Print the xspan and yspan
print(f"tspan: {tspan.shape}, xspan: {xspan.shape}, yspan: {yspan.shape}, zspan: {zspan.shape}")

tt = int(Nt//3)
print(f"End timestep of training: {tt}")

#Only consider one-third of the data upto timestep = 33 out of 101 timesteps
outputs = outputs[:,:tt,:,:,:]

#Take only one-third of the temporal domain
tspan = tspan[:tt]

#Create for trunk network
[t,x,y,z] = jnp.meshgrid(tspan, xspan, yspan, zspan, indexing = 'ij')
grid = jnp.transpose(jnp.array([t.flatten(), x.flatten(), y.flatten(), z.flatten()]))
print(grid.shape)

# Split the data into training (2000) and testing (500) samples
inputs_train, inputs_test, outputs_train, outputs_test = \
                    train_test_split(inputs, outputs, test_size=0.2, random_state=seed)

# Check the shapes of the subsets
print("Shape of inputs_train:", inputs_train.shape)
print("Shape of inputs_test:", inputs_test.shape)
print("Shape of outputs_train:", outputs_train.shape)
print("Shape of outputs_test:", outputs_test.shape)

#Network Inputs - train
branch_inputs_train = inputs_train    
trunk_inputs_train = grid             

#Inspecting the shapes
print("Shape of train branch inputs: ",branch_inputs_train.shape)
print("Shape of train trunk inputs: ",trunk_inputs_train.shape)
print("Shape of train output: ",outputs_train.shape)

#Network Inputs - test
branch_inputs_test = inputs_test      
trunk_inputs_test = grid              

#Inspecting the shapes
print("Shape of test branch inputs: ",branch_inputs_test.shape)
print("Shape of test trunk inputs: ",trunk_inputs_test.shape)
print("Shape of test output: ",outputs_test.shape)

class branch_net(nn.Module):

    layer_sizes: Sequence[int] 
    activation: Callable
    
    @nn.compact
    def __call__(self, x):
        init = nn.initializers.glorot_normal()
        
        # #x has shape (ns, nx, ny, nz) - so add channel dimension: (ns, nx, ny, nz, nc)
        x = x[..., jnp.newaxis]
        
        #Convolutional layers
        x = nn.Conv(features = 32, kernel_size = (3,3,3), strides = (2,2,1), padding = "SAME")(x)
        x = nn.activation.gelu(x)
        x = nn.max_pool(x, window_shape=(2,2,2), strides = (2,2,2), padding = "SAME")
        
        x = nn.Conv(features = 32, kernel_size = (2,2,2), strides = (2,2,1), padding = "SAME")(x)
        x = nn.activation.gelu(x)
        x = nn.avg_pool(x, window_shape = (2,2,2), strides = (2,2,2), padding = "SAME")
        
        x = x.flatten()   #flatten
        
        #MLP layers
        for i, layer in enumerate(self.layer_sizes[:-1]):
            x = nn.Dense(layer, kernel_init = init)(x)
            x = self.activation(x)
        x = nn.Dense(self.layer_sizes[-1], kernel_init = init)(x)
        return x

class trunk_net(nn.Module):
    trunk_layer_config: Sequence[int]
    activation: Callable
    
    @nn.compact
    def __call__(self, x):
        
        init = nn.initializers.glorot_normal()
        
        #Trunk network forward pass
        for i, layer_size in enumerate(self.trunk_layer_config):
            x = nn.Dense(layer_size, kernel_init = init)(x)
            x = self.activation(x)
        return x

def add_fourier_features(inputs, num_frequencies=10, max_freq=10):
    x = inputs[:, 0:1]
    y = inputs[:, 1:2]
    z = inputs[:, 2:]
    
    freqs = jnp.pi * jnp.linspace(1, max_freq, num_frequencies).reshape(1, -1)
    
    x_feat = jnp.concatenate([jnp.sin(freqs * x), jnp.cos(freqs * x)], axis=-1)
    y_feat = jnp.concatenate([jnp.sin(freqs * y), jnp.cos(freqs * y)], axis=-1)
    z_feat = jnp.concatenate([jnp.sin(freqs * z), jnp.cos(freqs * z)], axis=-1)
    
    return jnp.concatenate([inputs, x_feat, y_feat, z_feat], axis=-1)

class WaveletAct(nn.Module):
    init_w1: float = 1.0
    init_w2: float = 1.0
    init_omega: float = 1.0

    @nn.compact
    def __call__(self, x):
        w1 = self.param('w1', nn.initializers.constant(self.init_w1), (1,))
        w2 = self.param('w2', nn.initializers.constant(self.init_w2), (1,))
        omega = self.param('omega', nn.initializers.constant(self.init_omega), (1,))
        return w1 * jnp.sin(omega * x) + w2 * jnp.cos(omega * x)

class DeepONet(nn.Module):

    branch_net_config: Sequence[int]
    trunk_net_config: Sequence[int]
    use_Fourier_feat: bool = False

    def setup(self):
        self.branch_net = branch_net(self.branch_net_config, nn.activation.gelu)
        self.trunk_net = trunk_net(self.trunk_net_config, WaveletAct())

    @nn.compact
    def __call__(self, x_branch, x_trunk):

        if self.use_Fourier_feat:
            #Encode x_trunk into fourier features
            x_trunk = add_fourier_features(x_trunk)
        
        #Vectorize over multiple samples of input functions
        branch_outputs = jax.vmap(self.branch_net, in_axes = 0)(x_branch)
        
        #Vectorize over multiple query points
        trunk_outputs = jax.vmap(self.trunk_net, in_axes = 0)(x_trunk)
        bias = self.param('bias', nn.initializers.zeros, (1,))      
        
        inner_product = jnp.einsum('ik,jk->ij', branch_outputs, trunk_outputs)
        inner_product+=bias
        return inner_product

#DeepONet settings
num_sensor_locations = branch_inputs_train.shape[1]
num_query_locations = 4
latent_vector_size = 100

branch_network_layer_sizes = [256, 128] + [latent_vector_size]
trunk_network_layer_sizes = [128]*4 + [latent_vector_size]

model = DeepONet(branch_network_layer_sizes, trunk_network_layer_sizes)
model_fn = jax.jit(model.apply)
print(f"Model: {model}")

@jax.jit
def loss_fn(params, branch_inputs, trunk_inputs, gt_outputs):
    
    u_init = branch_inputs  # Current state input (e.g., u(t=0, x, y, z))
    u_true = gt_outputs     # Ground truth spatiotemporal state (e.g., u(t, x, y, z))

    # Predict the full spatiotemporal solution state
    u_pred = model_fn(params, u_init, trunk_inputs)  # Model's prediction for next state

    #u_pred is (Ns, Nt*Nx*Ny*Nz)
    u_pred = u_pred.reshape(u_true.shape[0], u_true.shape[1], u_true.shape[2], 
                            u_true.shape[3], u_true.shape[4])
    
    #----------------Compute the MSE loss and L2 error for the L-shaped domain--------------#
    loss1 = u_pred[:,:,:Nx//2,:Ny//2,:] - u_true[:,:,:Nx//2,:Ny//2,:] 
    loss2 = u_pred[:,:,:Nx//2,Ny//2:,:] - u_true[:,:,:Nx//2,Ny//2:,:]
    loss3 = u_pred[:,:,Nx//2:,:Ny//2,:] - u_true[:,:,Nx//2:,:Ny//2,:]
    loss = jnp.mean(loss1**2+loss2**2+loss3**2)
    
    l2_error1 = jnp.linalg.norm(u_pred[:,:,:Nx//2,:Ny//2,:] - u_true[:,:,:Nx//2,:Ny//2,:])/ \
                jnp.linalg.norm(u_true[:,:,:Nx//2,:Ny//2,:])
    l2_error2 = jnp.linalg.norm(u_pred[:,:,:Nx//2,Ny//2:,:] - u_true[:,:,:Nx//2,Ny//2:,:])/\
            jnp.linalg.norm(u_true[:,:,:Nx//2,Ny//2:,:])
    l2_error3 = jnp.linalg.norm(u_pred[:,:,Nx//2:,:Ny//2,:] - u_true[:,:,Nx//2:,:Ny//2,:])/\
        jnp.linalg.norm(u_true[:,:,Nx//2:,:Ny//2,:])
    #----------------------------------------------------------------------------------------#
    
    return loss,[l2_error1, l2_error2, l2_error3]

@jax.jit
def update(params, branch_inputs, trunk_inputs, gt_outputs, opt_state):
    (loss, _), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, branch_inputs, 
                                                                      trunk_inputs, gt_outputs)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

# Initialize model parameters
key = random.PRNGKey(seed)

params = model.init(key, branch_inputs_train[0:1], trunk_inputs_train[0:1])

# Optimizer setup
lr_scheduler = optax.schedules.exponential_decay(init_value = 1e-3, 
                                                 transition_steps = 2000, decay_rate = 0.96)
optimizer = optax.adam(learning_rate=lr_scheduler)
opt_state = optimizer.init(params)

training_loss_history = []
test_loss_history = []
num_epochs = int(1.5e5)
batch_size = 64

min_test_mse_loss = jnp.inf

filepath = Path(f'DeepONet_full_rollout/{args.exp_name}', parents=True, exist_ok=True)
filename = f"model_params_best_{seed}.pkl"

#Freeing memory by deleting inputs and outputs
del inputs, outputs

print("Starting training")
for epoch in tqdm(range(num_epochs), desc="Training Progress"):

    #Perform mini-batching
    shuffled_indices = jax.random.permutation(jax.random.PRNGKey(epoch), branch_inputs_train.shape[0])
    batch_indices = shuffled_indices[:batch_size]

    branch_inputs_train_batch = branch_inputs_train[batch_indices]
    outputs_train_batch = outputs_train[batch_indices]

    # Update the parameters and optimizer state
    params, opt_state, loss = update(
        params=params,
        branch_inputs=branch_inputs_train_batch,
        trunk_inputs=trunk_inputs_train,
        gt_outputs=outputs_train_batch,
        opt_state=opt_state
    )

    training_loss_history.append(loss)
    
    #Do predictions on the test data simultaneously
    test_mse_loss, [test_L2_err1, test_L2_err2, test_L2_err3] = loss_fn(params = params,
                            branch_inputs = branch_inputs_test, 
                            trunk_inputs = trunk_inputs_test, 
                            gt_outputs = outputs_test)
    test_loss_history.append(test_mse_loss)
    
    #Save the params of the best model encountered till now      
    if test_mse_loss < min_test_mse_loss:
        best_params = params
        # save_model_params(best_params, path = filepath, filename = filename)
        min_test_mse_loss = test_mse_loss
    
    #Print the train and test loss history every 1000 epochs
    if epoch % 500 == 0:
        print(f"Epoch: {epoch}, train_loss: {loss}, test_loss: {test_mse_loss}, " 
              f"test_L2_err1: {test_L2_err1}, test_L2_err2: {test_L2_err2}, "
              f"test_L2_err3: {test_L2_err3}, best_test_loss: {min_test_mse_loss}")

plt.figure(dpi = 130)
plt.semilogy(np.arange(epoch+1), training_loss_history, label = "Train loss")
plt.semilogy(np.arange(epoch+1), test_loss_history, label = "Test loss")

plt.xlabel("Epochs")
plt.ylabel("Loss")

plt.tick_params(which = 'major', axis = 'both', direction = 'in', length = 6)
plt.tick_params(which = 'minor', axis = 'both', direction = 'in', length = 3.5)
plt.minorticks_on()

plt.grid(alpha = 0.3)
plt.legend(loc = 'best')
plt.savefig(filepath / "loss_plot.pdf", dpi = 300)
plt.close()

#Save the loss arrays
save = False
if save:
    np.save(filepath / "Train_loss.npy",training_loss_history)
    np.save(filepath / "Test_loss.npy",test_loss_history)

print("Training completed succesfully!")

#Save the loss arrays
save = False
if save:
    np.save(filepath + "/train_loss.npy", training_loss_history)
    np.save(filepath + "/test_loss.npy", test_loss_history)
print("Loss curves saved.")

#-------INFERENCE-------##

print("At inference...")

base_path = "/home/dnayak2/scr4_sgoswam4/Dibya/backup/Datasets/3D_Heat_Conduction/"

outputs = jnp.load(os.path.join(base_path, "processed/3d_heat_field_results.npz"))['heat_field']

Ns, Nt, Nx, Ny, Nz = outputs.shape
print(f"Ns: {Ns}, Nt: {Nt}, Nx: {Nx}, Ny: {Ny}, Nz: {Nz}")

inputs = outputs[:, 0, :, :, :]
print(f"Inputs shape: {inputs.shape}, Outputs shape: {outputs.shape}")

#Get the xspan and yspan
xspan = jnp.linspace(0, 1, Nx)
yspan = jnp.linspace(0, 1, Ny)
zspan = jnp.linspace(0, 1, Nz)
tspan = jnp.linspace(0, 1, Nt)

#Print the xspan and yspan
print(f"tspan: {tspan.shape}, xspan: {xspan.shape}, yspan: {yspan.shape}, zspan: {zspan.shape}")

#Create for trunk network
[t,x,y,z] = jnp.meshgrid(tspan, xspan, yspan, zspan, indexing = 'ij')
grid = jnp.transpose(jnp.array([t.flatten(), x.flatten(), y.flatten(), z.flatten()]))
print(grid.shape)

#Creating grid for branch inputs new and trunk_inputs_new
branch_inputs_new = inputs
trunk_inputs_new = grid

# Predictions
import sklearn
from sklearn import metrics
import time

#Import the best model saved after full training
best_params = load_model_params(path = filepath, filename = filename)
print(f"Best params loaded: {filename}")

start_time = time.time()
predictions_outputs_new = model_fn(best_params, branch_inputs_new, trunk_inputs_new)
end_time = time.time()
print(f"Total time of inference for {predictions_outputs_new.shape[0]} samples: {end_time-start_time}")

print(predictions_outputs_new.shape, trunk_inputs_new.shape)

predictions_outputs_new = predictions_outputs_new.reshape(predictions_outputs_new.shape[0], Nt, Nx, Ny, Nz)

overall_rel_L2_err = jnp.linalg.norm(predictions_outputs_new - outputs)/jnp.linalg.norm(outputs)
print(f"Overall relative L2 error: {overall_rel_L2_err}")

auto_reg_error = []
num_time_steps = Nt

for i in range(num_time_steps):
    l2_error = jnp.linalg.norm(predictions_outputs_new[:,i,:,:,:] - outputs[:,i,:,:,:])/jnp.linalg.norm(outputs[:,i,:,:,:])
    auto_reg_error.append(l2_error)

#Compute statistics
# t = [10, 20, 30, 50, 50, 60, 70, 90, 100]
t = [48, 63, 93, 100]

print("------Extrapolation Errors------")
for t_idx in t:
    print(f"T_idx: {t_idx}, L2 error: {auto_reg_error[t_idx]}")

#Save the auto_reg_error array for comparing with NODE approach
save = False
if save:
    np.save(filepath + "/Auto_reg_error_full_rollout.npy", auto_reg_error)
    np.save(filepath + "/u_pred.npy", predictions_outputs_new)
    np.save(filepath + "/actual.npy", outputs)
print("Program executed successfully!")