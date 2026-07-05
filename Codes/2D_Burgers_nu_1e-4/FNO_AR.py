#!/usr/bin/env python
# coding: utf-8

#Importing all the necessary libraries
import os
import sys
import pickle
import time
from tqdm import tqdm
import torch

import torch
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

from models_fno import FNO2d
from utils_jax import *

print("Program started...")
# seed = 42
seed = np.random.choice(np.arange(99999), size=1, replace=True)[0]
np.random.seed(seed)
key = jax.random.PRNGKey(seed)
print(f"Seed set with value = {seed}")

#Read the 2D Burgers' dataset
base_path = "/home/dnayak2/data_sgoswam4/Dibya/Datasets/2D_Burgers/low_viscosity"
dataset = torch.load(os.path.join(base_path, "2D_scalar_Burgers_N1000_64x64.pth"))
print("2D Burgers dataset loaded!")

#For u-velocity field
outputs_u = (jnp.array(dataset["output_field"]))[:1000]
print("Outputs_u shape: ",outputs_u.shape)

#Free up memory
del dataset

Ns, Nt, Nx, Ny = outputs_u.shape
print(f"Ns: {Ns}, Nt: {Nt}, Nx: {Nx}, Ny: {Ny}")

#Define the training temporal to be one-third
tt = int(Nt//3)

#Creating the input and output training data
init_timestep = 0
end_timestep = tt

input_data_NN = outputs_u[:, init_timestep, :, :]    
output_data_NN = outputs_u[:, init_timestep+1, :, :]

for i in range(init_timestep+1, end_timestep):
    input_data_NN = jnp.vstack((input_data_NN, outputs_u[:,i,:,:]))
    output_data_NN = jnp.vstack((output_data_NN, outputs_u[:,i+1,:,:]))
print("After arranging the data for performing t->t+1 mapping, we have: ")
print(f"Input states: {input_data_NN.shape}")
print(f"Output states: {output_data_NN.shape}")


#Add channel dimension to the input-output data pairs
print("Creating meshgrid of only xy-locations")

#Create the meshes along X and Y axes
xspan = jnp.linspace(0, 1, Nx)
yspan = jnp.linspace(0, 1, Ny)

[X, Y] = jnp.meshgrid(xspan, yspan, indexing = "ij") 

#X and Y have shapes: (Nx, Ny). Next we tile it across all samples: (Ns*(Nt-1))
X_tiled = jnp.tile(X[None, :, :], (input_data_NN.shape[0], 1, 1))
Y_tiled = jnp.tile(Y[None, :, :], (input_data_NN.shape[0], 1, 1))
print(f"After tiling, X_tiled: {X_tiled.shape}, Y_tiled: {Y_tiled.shape}")

#Concatenate mesh to the input_data_NN_tensor
inputs_to_FNO = jnp.concatenate([input_data_NN[:, :, :, None], X_tiled[:, :, :, None], Y_tiled[:, :, :, None]], 
                                axis=-1)
output_FNO = output_data_NN[:, :, :, None]

print("After merging meshes")
print(f"Inputs: {inputs_to_FNO.shape}, Outputs: {output_FNO.shape}")

#Free up some memory
del input_data_NN, output_data_NN, X_tiled, Y_tiled

#Split into training and testing datasets
Ntrain = int(0.8*inputs_to_FNO.shape[0])
perm = jax.random.permutation(jax.random.PRNGKey(0), inputs_to_FNO.shape[0])

train_idx = perm[:Ntrain]
test_idx = perm[Ntrain:]

train_x = jnp.take(inputs_to_FNO, train_idx, axis=0)
test_x = jnp.take(inputs_to_FNO, test_idx, axis=0)

train_y = jnp.take(output_FNO, train_idx, axis=0)
test_y = jnp.take(output_FNO, test_idx, axis=0)

print(f"train_x shape: {train_x.shape}, train_y shape: {train_y.shape}")
print(f"test_x shape: {test_x.shape}, test_y shape: {test_y.shape}")

#Stop gradients for test_x and test_y
test_x = jax.lax.stop_gradient(test_x)
test_y = jax.lax.stop_gradient(test_y)

#Create the FNO-2D model object
modes1 = 12
modes2 = 12

#Create the FNO-2D model object
fno = FNO2d(in_channels = train_x.shape[-1],
            out_channels = train_y.shape[-1],
            modes1 = modes1,
            modes2 = modes2,
            width = 64,
            n_blocks = 6,
            activation = nn.activation.gelu,  
)
print(f"FNO2D model for AR learning: {fno}")
model_fn = jax.jit(fno.apply)

#Instantiate the model params
params = fno.init(key, train_x[0:1])

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
batch_size = 64
shuffle_key = jax.random.PRNGKey(80)
epochs = int(1e3)
min_val_loss = jnp.inf

result_dir = f"./FNO_AR"
file_name = f"best_model_params_{modes1}_seed{seed}.pkl"

print("Starting training...")
for epoch in range(epochs):
    shuffle_key, subkey = jax.random.split(shuffle_key)
    total_loss = 0
    nbatches = 0
    for batch_x, batch_y in tqdm(dataloader(subkey, train_x, train_y, batch_size),
                                desc=f"Epoch {epoch}"):
        params, opt_state, loss = make_step(params, opt_state, batch_x, batch_y)
        total_loss += loss
        nbatches += 1
    
    loss = total_loss/nbatches
    val_loss = loss_fn(params, test_x, test_y)
    
    #Save the best model
    if val_loss < min_val_loss:
        best_params = params
        min_val_loss = val_loss
        # save_model_params(best_params, result_dir, filename = file_name)
    
    loss_history.append(total_loss/nbatches)
    val_loss_history.append(val_loss)
    
    if epoch%50==0:
        print(f"Epoch: {epoch}, Train loss: {loss}, Val loss: {val_loss}") 
print("Training completed successfully! Moving on to plotting loss curves and inference.")

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

save = False
if save:
    plt.savefig(result_dir + f"/loss_plot_{modes1}_v2.jpeg", dpi = 300)
plt.show()


#Save the loss arrays
save = False
if save:
    np.save(result_dir + "/Train_loss.npy",loss_history)
    np.save(result_dir + "/Test_loss.npy",val_loss_history)


def create_input(x):    
    Ns_, Nx, Ny = x.shape
    
    xspan = jnp.linspace(0, 1, Nx)
    yspan = jnp.linspace(0, 1, Ny)
    
    [X, Y] = jnp.meshgrid(xspan, yspan, indexing = "ij")
    X_tiled = jnp.tile(X[None, :, :], (Ns_, 1, 1))
    Y_tiled = jnp.tile(Y[None, :, :], (Ns_, 1, 1))
    
    x_with_mesh = jnp.concatenate([x[:,:,:,None], X_tiled[:,:,:,None], Y_tiled[:,:,:,None]], axis=-1)
    return x_with_mesh


def run_inference(initial_u, n_steps):
    u_states = np.zeros_like(outputs_u)  # List to store the states over time
    u_states[:,0,:,:] = initial_u
    
    # Initialize the previous state (this could be your u_0 and u_1, etc.)
    u_curr = initial_u  # Set the current state to the initial state
    
    for i in range(1, n_steps):
        u_curr_in_FNO = create_input(u_curr)    #(Ns, Nx, Ny, in_channels)
        u_next_out_FNO = model_fn(best_params, u_curr_in_FNO)   #(Ns, Nx, Ny, out_channels)
        u_next = u_next_out_FNO[:, :, :, 0]
        
        # Append the predicted state to the list
        u_states[:, i, :, :] = u_next
        
        # Update previous and current states for the next step
        u_curr = u_next
    
    return u_states

print("At inference...")
# Load the best model parameters
best_params = load_model_params(result_dir, filename = file_name)
print(f"Best model params loaded with filename: {file_name}")

#Read the 2D Burgers' dataset
base_path = "/home/dnayak2/data_sgoswam4/Dibya/Datasets/2D_Burgers/low_viscosity"
dataset = torch.load(os.path.join(base_path, "2D_scalar_Burgers_N1000_64x64.pth"))
print("2D Burgers dataset loaded!")

#For u-velocity field
outputs_u = (jnp.array(dataset["output_field"]))[:1000]
print("Outputs_u shape: ",outputs_u.shape)

Ns, Nt, Nx, Ny = outputs_u.shape
print(f"Ns: {Ns}, Nt: {Nt}, Nx: {Nx}, Ny: {Ny}")

print("Running inference...")
u_start = outputs_u[:, 0, :, :]
start_time = time.time()
u_pred = run_inference(u_start, n_steps=Nt)
end_time = time.time()
print(f"Inference complete for {u_pred.shape[0]} samples in {end_time-start_time} secs")
print("Post inference, shapes...")
print(f"Predicted: {u_pred.shape}, Outputs: {outputs_u.shape}")

#Compute relative L2 error between predicted and outputs
rel_l2_err = np.linalg.norm(u_pred - outputs_u)/np.linalg.norm(outputs_u)
print(f"Overall relative L2 error: {rel_l2_err}")

#Plotting the relative L2 error obtained at every timestep to show accummulation of autoregressive error
auto_reg_error = []
num_time_steps = Nt

for i in range(num_time_steps):
    l2_error = jnp.linalg.norm(u_pred[:,i,:,:] - outputs_u[:,i,:,:])/jnp.linalg.norm(outputs_u[:,i,:,:])
    auto_reg_error.append(l2_error)

#Compute statistics
t = [60, 70, 90, 100]

for t_idx in t:
    print(f"t: {t_idx}, L2 error: {auto_reg_error[t_idx]}")

#Save all the relevant arrays
save=False
if save:
    np.save(result_dir + "/u_pred_v2.npy", u_pred)
    np.save(result_dir + "/auto_reg_error.npy", auto_reg_error)
print("Program executed successfully!")