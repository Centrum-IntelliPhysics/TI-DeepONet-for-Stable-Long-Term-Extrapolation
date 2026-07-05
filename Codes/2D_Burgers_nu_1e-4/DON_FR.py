#!/usr/bin/env python
# coding: utf-8

#importing all necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.io
from sklearn.preprocessing import StandardScaler

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

print("Program started....")
# seed = 42
seed = np.random.choice(np.arange(99999), size=1, replace=True)[0]
np.random.seed(seed)
key = random.PRNGKey(seed)
print(f"Seed set with value = {seed}")

#Read the 2D Burgers' dataset
base_path = "/home/dnayak2/data_sgoswam4/Dibya/Datasets/2D_Burgers/low_viscosity"
dataset = torch.load(os.path.join(base_path, "2D_scalar_Burgers_N1000_64x64.pth"))
print("2D Burgers dataset loaded!")

#For u-velocity field
outputs = (jnp.array(dataset["output_field"]))[:1000]
print("Outputs_u shape: ",outputs.shape)
Ns, Nt, Nx, Ny = outputs.shape
print(f"Ns: {Ns}, Nt: {Nt}, Nx: {Nx}, Ny: {Ny}")
inputs = outputs[:, 0, :, :]

del dataset

tspan = jnp.linspace(0, 1, Nt)
xspan = jnp.linspace(0, 1, Nx)
yspan = jnp.linspace(0, 1, Ny)

print(inputs.shape, outputs.shape)
tt=int(Nt//3)

#Only consider one-third of the data upto timestep = 33 out of 101 timesteps
outputs = outputs[:,:tt,:,:]

#Take only one-third of the temporal domain
tspan = tspan[:tt]

#Create for trunk network
[t,x,y] = jnp.meshgrid(tspan, xspan, yspan, indexing = 'ij')
grid = jnp.transpose(jnp.array([t.flatten(), x.flatten(), y.flatten()]))
print(grid.shape)
print(grid)


# Split the data into training (2000) and testing (500) samples
inputs_train, inputs_test, outputs_train, outputs_test = \
                    train_test_split(inputs, outputs, test_size=0.2, random_state=seed)
outputs_train = outputs_train.reshape(outputs_train.shape[0], tt*Nx*Ny)
outputs_test = outputs_test.reshape(outputs_test.shape[0], tt*Nx*Ny)

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
        
        # #x has shape (ns, nx, ny) - so add channel dimension: (ns, nx, ny, nc)
        x = x[..., jnp.newaxis]
        
        #Convolutional layers
        x = nn.Conv(features = 64, kernel_size = (3,3), strides = 1, padding = "SAME")(x)
        x = nn.gelu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides = (2, 2), padding = "SAME")

        x = nn.Conv(features = 64, kernel_size = (3,3), strides = 1, padding = "SAME")(x)
        x = nn.gelu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides = (2, 2), padding = "SAME")
        
        x = nn.Conv(features = 64, kernel_size = (2, 2), strides = 1, padding = "SAME")(x)
        x = nn.gelu(x)
        x = nn.avg_pool(x, window_shape = (2,2), strides = (2,2), padding = "SAME")
        
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
    
def add_fourier_features(inputs, num_frequencies=20, max_freq=20):
    t = inputs[:, 0:1]
    x = inputs[:, 1:2]
    y = inputs[:, 2:3]
    
    freqs = jnp.pi * jnp.linspace(1, max_freq, num_frequencies).reshape(1, -1)
    t_feat = jnp.concatenate([jnp.sin(freqs * t), jnp.cos(freqs * t)], axis=-1)
    x_feat = jnp.concatenate([jnp.sin(freqs * x), jnp.cos(freqs * x)], axis=-1)
    y_feat = jnp.concatenate([jnp.sin(freqs * y), jnp.cos(freqs * y)], axis=-1)
    
    return jnp.concatenate([inputs, t_feat, x_feat, y_feat], axis=-1)

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
    use_Fourier_feat: bool = True

    def setup(self):
        self.branch_net = branch_net(self.branch_net_config, nn.gelu)
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

latent_vector_size = 100
branch_network_layer_sizes = [256, 128] + [latent_vector_size]
trunk_network_layer_sizes = [128]*5 + [latent_vector_size]

model = DeepONet(branch_net_config = branch_network_layer_sizes, 
                                      trunk_net_config = trunk_network_layer_sizes)
model_fn = jax.jit(model.apply)
print(f"Model: {model}")

# Define the training process from here
@jax.jit
def loss_fn(params, branch_inputs, trunk_inputs, gt_outputs):
    predictions = model_fn(params, branch_inputs,trunk_inputs)
    mse_loss = jnp.mean(jnp.square(predictions - gt_outputs))   
    l2_error = jnp.linalg.norm(predictions - gt_outputs)/jnp.linalg.norm(gt_outputs)
    return mse_loss, l2_error

@jax.jit
def update(params, branch_inputs, trunk_inputs, gt_outputs, opt_state):
    (loss, l2_error), grads = \
            jax.value_and_grad(loss_fn, has_aux=True)(params, branch_inputs, trunk_inputs, gt_outputs)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss, l2_error

# Initialize model parameters
key = random.PRNGKey(seed)

params = model.init(key, branch_inputs_train[0:1], trunk_inputs_train[0:1])

# Optimizer setup
lr_scheduler = optax.schedules.exponential_decay(init_value = 1e-3, 
                                                 transition_steps = 5000, decay_rate = 0.95)
optimizer = optax.adam(learning_rate=lr_scheduler)
opt_state = optimizer.init(params)

training_loss_history = []
test_loss_history = []
num_epochs = int(1e5)
# batch_size = 128
batch_size = 32   #This is only set to assess the compute speed of training

min_test_l2_error = jnp.inf
min_test_mse_loss = jnp.inf

filepath = 'DeepONet_full_rollout'
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
    params, opt_state, loss, l2_error = update(
        params=params,
        branch_inputs=branch_inputs_train_batch,
        trunk_inputs=trunk_inputs_train,
        gt_outputs=outputs_train_batch,
        opt_state=opt_state
    )

    training_loss_history.append(loss)
    
    #Do predictions on the test data simultaneously
    test_mse_loss, test_l2_error = loss_fn(params = params, 
                            branch_inputs = branch_inputs_test, 
                            trunk_inputs = trunk_inputs_test, 
                            gt_outputs = outputs_test)
    test_loss_history.append(test_mse_loss)
    
    #Save the params of the best model encountered till now
    if test_l2_error < min_test_l2_error:
        min_test_l2_error = test_l2_error
        
    if test_mse_loss < min_test_mse_loss:
        best_params = params
        # save_model_params(best_params, path = filepath, filename = filename)
        min_test_mse_loss = test_mse_loss
    
    #Print the train and test loss history every 1000 epochs
    if epoch % 1000 == 0:
        print(f"Epoch: {epoch}, train_loss: {loss}, test_loss: {test_mse_loss}, "
              f"min_test_loss: {min_test_mse_loss}, min_test_l2_error: {min_test_l2_error}")
print("Training complete!")

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
# plt.savefig(filepath + "/loss_plot_v1.jpeg", dpi = 800)
plt.close()

#Save the loss arrays
save = False
if save:
    np.save(filepath + "/train_loss.npy", training_loss_history)
    np.save(filepath + "/test_loss.npy", test_loss_history)

##-------INFERENCE-------##

print("At inference...")
#Read the 2D Burgers' dataset
base_path = "/home/dnayak2/data_sgoswam4/Dibya/Datasets/2D_Burgers/low_viscosity"
dataset = torch.load(os.path.join(base_path, "2D_scalar_Burgers_N1000_64x64.pth"))
print("2D Burgers dataset loaded!")

#For u-velocity field
outputs = (jnp.array(dataset["output_field"]))[:1000]
print("Outputs_u shape: ",outputs.shape)
Ns, Nt, Nx, Ny = outputs.shape
print(f"Ns: {Ns}, Nt: {Nt}, Nx: {Nx}, Ny: {Ny}")

inputs = outputs[:, 0, :, :]

#Free memory by deleting dataset
del dataset

tspan = jnp.linspace(0, 1, Nt)
xspan = jnp.linspace(0, 1, Nx)
yspan = jnp.linspace(0, 1, Ny)

#Create for trunk network
[t,x,y] = jnp.meshgrid(tspan, xspan, yspan, indexing = 'ij')
grid = jnp.transpose(jnp.array([t.flatten(), x.flatten(), y.flatten()]))

#Creating grid for branch inputs new and trunk_inputs_new
branch_inputs_new = inputs
trunk_inputs_new = grid
trunk_inputs_new.shape, trunk_inputs_new

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

predictions_outputs_new = predictions_outputs_new.reshape(predictions_outputs_new.shape[0], Nt, Nx, Ny)

#Randomly selecting "size" number of samples out of the test dataset
random_samples = np.random.choice(np.arange(outputs.shape[0]), size = 3, replace = 'False')

t_query = [0, 25, 50, 75, -1]
for i in random_samples:
    
    for t in t_query:
        prediction_i = predictions_outputs_new[i, t, :, :]
        target_i = outputs[i, t, :, :]
        
        error_i = np.abs(prediction_i - target_i)

        plt.figure(figsize = (12,3))

        plt.subplot(1,3,1)
        contour1 = plt.contourf(xspan, yspan, prediction_i.T, levels = 20, cmap = 'viridis')
        cbar1 = plt.colorbar()
        cbar1.ax.tick_params(labelsize = 12)
        plt.xlabel("x", fontsize = 14)
        plt.ylabel("y", fontsize = 14)
        plt.xticks(fontsize = 12)
        plt.yticks(fontsize = 12)
        plt.title("Predicted", fontsize = 16)

        plt.subplot(1,3,2)
        contour2 = plt.contourf(xspan, yspan, target_i.T, levels = 20, cmap = 'viridis')
        cbar2 = plt.colorbar()
        cbar2.ax.tick_params(labelsize = 12)
        plt.xlabel("x", fontsize = 14)
        plt.ylabel("y", fontsize = 14)
        plt.xticks(fontsize = 12)
        plt.yticks(fontsize = 12)
        plt.title("Actual", fontsize = 16)

        plt.subplot(1,3,3)
        contour3 = plt.contourf(xspan, yspan, error_i.T, levels = 20, cmap = 'Wistia')
        cbar3 = plt.colorbar()
        cbar3.ax.tick_params(labelsize = 12)
        plt.xlabel("x", fontsize = 14)
        plt.ylabel("y", fontsize = 14)
        plt.xticks(fontsize = 12)
        plt.yticks(fontsize = 12)
        plt.title("Error", fontsize = 16)
        
        plt.suptitle(f"Idx: {i}, timestep: {t}")

        plt.tight_layout()
        # plt.savefig(filepath + f"/Contour_plots_{i}_timestep_{t}.jpeg", dpi = 800)
        plt.close()

overall_rel_L2_err = jnp.linalg.norm(predictions_outputs_new - outputs)/jnp.linalg.norm(outputs)
print(f"Overall relative L2 error: {overall_rel_L2_err}")

auto_reg_error = []
num_time_steps = Nt

for i in range(num_time_steps):
    l2_error = jnp.linalg.norm(predictions_outputs_new[:,i,:,:] - outputs[:,i,:,:])/jnp.linalg.norm(outputs[:,i,:,:])
    auto_reg_error.append(l2_error)

#Compute statistics
t = [60, 70, 90, 100]
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