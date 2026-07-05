#!/usr/bin/env python
# coding: utf-8

import os, sys, pickle
import jax, jaxlib
from pathlib import Path
import jax.numpy as jnp
import matplotlib.pyplot as plt
import scipy
from scipy.io import loadmat
import numpy as np
import h5py
import argparse

import torch
import flax
from flax import linen as nn
import optax
from sklearn.model_selection import train_test_split
from typing import Callable, Sequence

from tqdm import tqdm
from utils_jax import *

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

#Get the xspan and yspan
xspan = jnp.linspace(0, 1, Nx)
yspan = jnp.linspace(0, 1, Ny)
zspan = jnp.linspace(0, 1, Nz)
tspan = jnp.linspace(0, 1, Nt)

#Print the xspan and yspan
print(f"tspan: {tspan.shape}, xspan: {xspan.shape}, yspan: {yspan.shape}, zspan: {zspan.shape}")

#Compute dt
dt_val = (1-0)/(Nt-1)
print(f"Computed dt: {dt_val}")

tt = int(Nt//3)
print(f"End timestep of training: {tt}")

#Creating the input and output training data
init_timestep = 0
end_timestep = tt

input_data_NN = outputs[:, init_timestep, :, :, :]    
output_data_NN = outputs[:, init_timestep+1, :, :, :]

for i in range(init_timestep+1, end_timestep):
    input_data_NN = jnp.vstack((input_data_NN, outputs[:,i,:,:,:]))
    output_data_NN = jnp.vstack((output_data_NN, outputs[:,i+1,:,:,:]))
print("After data arrangement shapes of inputs and outputs:")
print(f"Inputs: {input_data_NN.shape}, Outputs: {output_data_NN.shape}")

#Reshaping the output_data_NN from (ns*nt//3, nx, ny, nz) to (ns*nt//3, nx*ny*nz)
#Input_data_NN remains as it is, i.e., (ns*nt//3, nx, ny, nz)
output_data_NN = output_data_NN.reshape(output_data_NN.shape[0], 
                                        output_data_NN.shape[1]*output_data_NN.shape[2]*output_data_NN.shape[3])
print(input_data_NN.shape, output_data_NN.shape)

input_data_NN_train, input_data_NN_test, output_data_NN_train, output_data_NN_test = \
                        train_test_split(input_data_NN, output_data_NN, test_size = 0.2, random_state = 42)
print(input_data_NN_train.shape, input_data_NN_test.shape, 
      output_data_NN_train.shape, output_data_NN_test.shape)

#Freeing memory by deleting input_data_NN and output_data_NN
del input_data_NN, output_data_NN

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
        x = nn.activation.relu(x)
        x = nn.max_pool(x, window_shape=(2,2,2), strides = (2,2,2), padding = "SAME")
        
        x = nn.Conv(features = 32, kernel_size = (2,2,2), strides = (2,2,1), padding = "SAME")(x)
        x = nn.activation.relu(x)
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
        self.branch_net = branch_net(self.branch_net_config, nn.activation.tanh)
        self.trunk_net = trunk_net(self.trunk_net_config, nn.activation.tanh)

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
    
#Form branch and trunk inputs train

#Create for trunk network
[x,y,z] = jnp.meshgrid(xspan, yspan, zspan, indexing = 'ij')
grid = jnp.transpose(jnp.array([x.flatten(), y.flatten(), z.flatten()]))
print(grid.shape)

#Creating the training data for branch and trunk inputs
branch_inputs_train = input_data_NN_train
trunk_inputs_train = grid
outputs_train = output_data_NN_train

print("Shape of branch inputs train: ",branch_inputs_train.shape)
print("Shape of trunk inputs train: ",trunk_inputs_train.shape)
print("Shape of outputs train: ",outputs_train.shape)
print("Shape of grid: ",grid.shape)


#For branch and trunk inputs test
branch_inputs_test = input_data_NN_test
trunk_inputs_test = grid
outputs_test = output_data_NN_test

print("Shape of branch inputs test: ",branch_inputs_test.shape)
print("Shape of trunk inputs test: ",trunk_inputs_test.shape)
print("Shape of outputs test: ",outputs_test.shape)
print("Shape of grid: ",grid.shape)

#DeepONet settings
num_sensor_locations = branch_inputs_train.shape[1]
num_query_locations = 3
latent_vector_size = 100

branch_network_layer_sizes = [256, 128] + [latent_vector_size]
trunk_network_layer_sizes = [128]*4 + [latent_vector_size]

model = DeepONet(branch_network_layer_sizes, trunk_network_layer_sizes)
model_fn = jax.jit(model.apply)
print(f"Model: {model}")

@jax.jit
def loss_fn(params, branch_inputs, trunk_inputs, gt_outputs, dt):
    
    u_curr = branch_inputs  # Current state input (e.g., u(t))
    u_next = gt_outputs     # Ground truth next state (e.g., u(t+1))

    # Predict the next state of the system (u_next) from the current state using the model
    u_pred_next = model_fn(params, u_curr, trunk_inputs)  # Model's prediction for next state

    #----------------Compute the MSE loss and L2 error for the L-shaped domain--------------#
    u_pred_next = u_pred_next.reshape(-1, Nx, Ny, Nz)
    u_next = u_next.reshape(-1, Nx, Ny, Nz)

    loss1 = u_pred_next[:,:Nx//2,:Ny//2,:] - u_next[:,:Nx//2,:Ny//2,:] 
    loss2 = u_pred_next[:,:Nx//2,Ny//2:,:] - u_next[:,:Nx//2,Ny//2:,:]
    loss3 = u_pred_next[:,Nx//2:,:Ny//2,:] - u_next[:,Nx//2:,:Ny//2,:]
    loss4 = u_pred_next[:,Nx//2:,Ny//2:,:]
    loss = jnp.mean(loss1**2+loss2**2+loss3**2)
    
    l2_error1 = jnp.linalg.norm(u_pred_next[:,:Nx//2,:Ny//2,:] - u_next[:,:Nx//2,:Ny//2,:])/ \
                jnp.linalg.norm(u_next[:,:Nx//2,:Ny//2,:])
    l2_error2 = jnp.linalg.norm(u_pred_next[:,:Nx//2,Ny//2:,:] - u_next[:,:Nx//2,Ny//2:,:])/\
            jnp.linalg.norm(u_next[:,:Nx//2,Ny//2:,:])
    l2_error3 = jnp.linalg.norm(u_pred_next[:,Nx//2:,:Ny//2,:] - u_next[:,Nx//2:,:Ny//2,:])/\
        jnp.linalg.norm(u_next[:,Nx//2:,:Ny//2,:])
    #----------------------------------------------------------------------------------------#
    
    return loss,[l2_error1, l2_error2, l2_error3]

@jax.jit
def update(params, branch_inputs, trunk_inputs, gt_outputs, opt_state, dt):
    (loss, _), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, branch_inputs, 
                                                                      trunk_inputs, gt_outputs, dt)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

#Freeing memory by deleting inputs and outputs
del outputs

# Initialize model parameters
key = jax.random.PRNGKey(seed)   #42
params = model.init(key, branch_inputs_train[0:1, ...], trunk_inputs_train[0:1, ...])

# # Optimizer setup
lr_scheduler = optax.schedules.exponential_decay(init_value=1e-3, transition_steps=2000, decay_rate=0.96)
optimizer = optax.adam(learning_rate=lr_scheduler)
opt_state = optimizer.init(params)

training_loss_history = []
test_loss_history = []
num_epochs = int(1.25e5)
# batch_size = 32
batch_size = 64   #Only to compute speed of training

min_test_loss = jnp.inf

filepath = Path(f'DeepONet_Autoregressive/{args.exp_name}', parents=True, exist_ok=True)
filename = f"model_params_best_{seed}.pkl"

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
        opt_state=opt_state,
        dt=dt_val
    )

    training_loss_history.append(loss)
    
    #Do predictions on the test data simultaneously
    test_mse_loss, [test_L2_err1, test_L2_err2, test_L2_err3] = loss_fn(params = params, 
                            branch_inputs = branch_inputs_test, 
                            trunk_inputs = trunk_inputs_test, 
                            gt_outputs = outputs_test,
                            dt=dt_val)
    test_loss_history.append(test_mse_loss)
    
    #Save the params of the best model encountered till now
    if test_mse_loss < min_test_loss:
        best_params = params
        save_model_params(best_params, path = filepath, filename = filename)
        min_test_loss = test_mse_loss
    
    #Print the train and test loss history every 1000 epochs
    if epoch % 500 == 0:
        print(f"Epoch: {epoch}, train_loss: {loss}, test_loss: {test_mse_loss}, " 
              f"test_L2_err1: {test_L2_err1}, test_L2_err2: {test_L2_err2}, "
              f"test_L2_err3: {test_L2_err3}, best_test_loss: {min_test_loss}")

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
# plt.savefig(filepath / "loss_plot.pdf", dpi = 300)
plt.close()

#Save the loss arrays
save = False
if save:
    np.save(filepath / "Train_loss.npy",training_loss_history)
    np.save(filepath / "Test_loss.npy",test_loss_history)

print("Training completed succesfully!")

def inference(u_curr, best_params, trunk_inputs_test):
    u_next = model_fn(best_params, u_curr, trunk_inputs_test)
    return u_next

def run_inference(initial_u, best_params, trunk_inputs_test, n_steps):
    u_states = np.zeros(shape = (initial_u.shape[0], n_steps, Nx, Ny, Nz))  # List to store the states over time
    u_states[:,0,:,:,:] = initial_u    #(ns, nt, nx, ny, nz)

    # Initialize the previous state (this could be your u_0 and u_1, etc.)
    u_curr = initial_u  # Set the current state to the initial state   -> (n, nx, ny, nz)
    
    for i in range(1, n_steps):
        # Perform one inference step using the multistep method
        u_next = inference(u_curr, best_params, trunk_inputs_test)
        
        #u_next is (n, nx*ny), so reshape it to (n, nx, ny)
        u_next = u_next.reshape(u_next.shape[0], Nx, Ny, Nz)
        
        #Set the first quadrant in the L-shaped to all zeros
        u_next = u_next.at[:, Nx//2:, Ny//2:, :].set(0.0)

        # Append the predicted state to the list
        u_states[:, i, :, :, :] = u_next
        
        # Update previous and current states for the next step
        u_curr = u_next
    
    return u_states

print("At inference...")

# Load the best model parameters
import time
best_params = load_model_params(path=filepath, filename=filename)
print(f"Best params loaded: {filename}")

#-------INFERENCE-------##
print("At inference...")

base_path = "/home/dnayak2/scr4_sgoswam4/Dibya/backup/Datasets/3D_Heat_Conduction/"

outputs = jnp.load(os.path.join(base_path, "processed/3d_heat_field_results.npz"))['heat_field']

Ns, Nt, Nx, Ny, Nz = outputs.shape
print(f"Ns: {Ns}, Nt: {Nt}, Nx: {Nx}, Ny: {Ny}, Nz: {Nz}")

u_curr = outputs[:, 0, :, :, :]

NSTEPS = Nt
start_time = time.time()
u_pred = run_inference(u_curr, best_params, trunk_inputs_test, n_steps=NSTEPS)
end_time = time.time()

print(f"Total time of inference for {u_pred.shape[0]} samples: ",end_time-start_time)
print(f"Predictions: {u_pred.shape}, Ground Truth: {outputs.shape}")

L2_err = jnp.linalg.norm(u_pred - outputs)/jnp.linalg.norm(outputs)
print(f"Overall rel L2 error: {L2_err}")

#Plotting the relative L2 error obtained at every timestep to show accummulation of autoregressive error
auto_reg_error = []
num_time_steps = NSTEPS

for i in range(num_time_steps):
    l2_error = jnp.linalg.norm(u_pred[:,i,:,:,:] - outputs[:,i,:,:,:])/jnp.linalg.norm(outputs[:,i,:,:,:])
    auto_reg_error.append(l2_error)

#Compute statistics
# t = [10, 20, 30, 50, 50, 60, 70, 90, 100]
t = [48, 63, 93, 100]
print("------Extrapolation Errors------")
for t_idx in t:
    print(f"T_idx: {t_idx}, L2 error: {auto_reg_error[t_idx]}")

plt.figure(figsize =(5,3.5))
plt.plot(auto_reg_error, color = "blue", marker='o', markersize=1, linestyle='-')
plt.xlabel("Time (t)", fontsize = 14)
plt.ylabel("Relative L2 error", fontsize = 14)

plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)

plt.tick_params(axis = "both", which = "major", length = 6, direction = "in")
plt.tick_params(axis = "both", which = "minor", length = 3.5, direction = "in")
plt.minorticks_on()

plt.grid(which="major", axis="both", alpha=0.6, linestyle='-')
plt.grid(which="minor", axis="both", alpha=0.3, linestyle='--')

plt.tight_layout()
# plt.savefig(filepath / "Error_acc_v2.pdf", dpi=200)
plt.close()

#Save the auto_reg_error array for comparing with NODE approach
#Saving the u_pred and ground truth output arrays for separate postprocessing

save = False

if save:
    np.savez_compressed(filepath / f"results_{seed}.npz",
                        u_pred=u_pred)
print("Program executed successfully!")