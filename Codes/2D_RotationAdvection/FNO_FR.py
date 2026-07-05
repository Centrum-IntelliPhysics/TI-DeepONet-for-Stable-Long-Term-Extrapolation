#!/usr/bin/env python
# coding: utf-8

#Importing all the necessary libraries
import os
import sys
import pickle
import time
import torch
import h5py
import argparse
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.io import loadmat

import flax
from flax import linen as nn

import optax
import matplotlib.pyplot as plt
import matplotlib
from typing import Callable, List, Sequence
import scipy
from tqdm import tqdm

from models_fno import FNO3d
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

#Combining all .mat files into one
base_path = "/home/dnayak2/data_sgoswam4/Dibya/Datasets/2D_rotation_advection/Case3_quarter_rotation_coarsen2/samples/"
print(base_path)

dataset_lst = []

for i in range(len([f for f in os.listdir(base_path) if os.path.isfile(os.path.join(base_path, f))])):
    data = loadmat(os.path.join(base_path, f"sample_{i+1:04d}.mat"))
    dataset_lst.append(data['snapshots'])
    
    if i==0:
        X = jnp.array(data['x'])
        Y = jnp.array(data['y'])
        tspan = jnp.array(data['t_vec'])
        dt_sim = jnp.array(data['dt'])
        dt_saved = jnp.array(data['dt_saved'])
        
    del data

outputs = jnp.array(dataset_lst)
#Only consider first 101 (0:100) timesteps
outputs = outputs[:,:101,:,:]

Ns, Nt, Nx, Ny = outputs.shape
print(f"Ns: {Ns}, Nt: {Nt}, Nx: {Nx}, Ny: {Ny}")

#Free up memory
del dataset_lst

#Get the xspan and yspan
xspan = jnp.unique(X)
yspan = jnp.unique(Y)

#Print the xspan and yspan
print(f"xspan: {xspan}, {xspan.shape}")
print("\n")
print(f"yspan: {yspan}, {yspan.shape}")
print("\n")

#Print time vector
tspan = (tspan[:,:Nt]).T
tspan = tspan.flatten()
print(f"tspan: {tspan}, {tspan.shape}")
print("\n")
print(f"Actual dt: {dt_sim}, Saved dt: {dt_saved}")

#Use dt_saved as dt_val
dt_val = dt_saved

#Consider training only for the first one-third of the temporal domain
# tt = int(Nt//3)
tt = 40

inputs = outputs[:, 0, :, :, None]     #(Ns, Nx, Ny, Nv=1)
outputs = outputs[:, :tt, :, :, None]    #(Ns, Nt, Nx, Ny, Nv=1)
print(f"For training in FR mode - inputs: {inputs.shape}, outputs: {outputs.shape}")

# Create coordinate grids
print("Creating the meshgrid")
# Only consider one-third of the time domain coordinates
tspan = tspan[:tt]

# Meshgrid to create 2D coordinate arrays
[T, X, Y] = jnp.meshgrid(tspan, xspan, yspan, indexing='ij')

#T,X,Y all have (Nt, Nx, Ny)
T_tiled = jnp.tile(T[None,:,:,:], (Ns,1,1,1))
X_tiled = jnp.tile(X[None,:,:,:], (Ns,1,1,1))
Y_tiled = jnp.tile(Y[None,:,:,:], (Ns,1,1,1))

# Printing shapes after tiling across all samples
print(f"T_tiled shape: {T_tiled.shape}")
print(f"X_tiled shape: {X_tiled.shape}")
print(f"X_tiled shape: {X_tiled.shape}")

# tile inputs
inputs_tiled = jnp.tile(inputs[:,None,:,:,:], (1, tt, 1, 1, 1))    #Tile only for the seen temporal steps (tt)
print(f"Shape of tiled inputs: {inputs_tiled.shape}")

#Stack all
inputs_to_FNO = jnp.concatenate([inputs_tiled, T_tiled[...,None], X_tiled[...,None], Y_tiled[...,None]], axis=-1)
output_FNO = outputs
print("After dataset preparation for FNO full rollout, shapes...")
print(f"Inputs_to_FNO: {inputs_to_FNO.shape}, Outputs_from_FNO: {output_FNO.shape}")

#Free up some memory
del inputs_tiled, T_tiled, X_tiled, Y_tiled

#Separate into train and test datasets
Ntrain = int(0.8*Ns)
perm = jax.random.permutation(jax.random.PRNGKey(0), Ns)

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

#Free up some memory
del inputs_to_FNO, output_FNO
modes1 = 12
modes2 = 12
modes3 = 12

#Create the FNO-2D model object
fno = FNO3d(in_channels = train_x.shape[-1],
            out_channels = train_y.shape[-1],
            modes1 = modes1,
            modes2 = modes2,
            modes3 = modes3,
            width = 64,
            n_blocks = 6,
            activation = nn.activation.gelu,  
)
print(f"FNO3D model for full rollout mode: {fno}")
model_fn = jax.jit(fno.apply)

#Instantiate the model params
params = fno.init(jax.random.PRNGKey(seed), train_x[0:1])    #Earlier this seed was value 42  

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
batch_size = 32
shuffle_key = jax.random.PRNGKey(seed)   #Seed value was 80
epochs = int(5e3)
min_val_loss = jnp.inf

result_dir = Path(f'FNO_full_rollout/{args.exp_name}', parents=True, exist_ok=True)
# result_dir = Path(f'FNO_full_rollout/FNO_full_rollout_16657905', parents=True, exist_ok=True)
filename = f"best_model_params_FNO_{modes1}_seed{seed}.pkl"

print("Starting training...")
for epoch in range(epochs):
    shuffle_key, subkey = jax.random.split(shuffle_key)
    total_loss = 0
    nbatches = 0

    # for batch_x, batch_y in tqdm(dataloader(subkey, train_x, train_y, batch_size),
    #                             desc="Training Progress"):
    for batch_x, batch_y in tqdm(dataloader(subkey, train_x, train_y, batch_size),
                            desc=f"Epoch {epoch}",
                            unit="it"):
        params, opt_state, loss = make_step(params, opt_state, batch_x, batch_y)
        total_loss += loss
        nbatches += 1

    loss = total_loss / nbatches
    val_loss = loss_fn(params, test_x, test_y)

    if val_loss < min_val_loss:
        best_params = params
        min_val_loss = val_loss
        # save_model_params(best_params, result_dir, filename=filename)

    loss_history.append(loss)
    val_loss_history.append(val_loss)

    if epoch % 100 == 0:
        print(f"Epoch: {epoch}, Train loss: {loss}, Val loss: {val_loss}")
print("Training complete.")

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
    plt.savefig(result_dir / f"loss_plot_{modes1}_v2.pdf", dpi = 200)
plt.show()

#Save the loss arrays
save = False
if save:
    np.save(result_dir + f"/Train_loss.npy",loss_history)
    np.save(result_dir + f"/Test_loss.npy",val_loss_history)

print("At inference...")
#Perform Inference
# Load the best model parameters
best_params = load_model_params(result_dir, filename = filename)
print(f"Best params loaded with filename: {filename}")

#Combining all .mat files into one
dataset_lst = []

for i in range(len([f for f in os.listdir(base_path) if os.path.isfile(os.path.join(base_path, f))])):
    data = loadmat(os.path.join(base_path, f"sample_{i+1:04d}.mat"))
    dataset_lst.append(data['snapshots'])
    
    if i==0:
        X = jnp.array(data['x'])
        Y = jnp.array(data['y'])
        tspan = jnp.array(data['t_vec'])
        dt_sim = jnp.array(data['dt'])
        dt_saved = jnp.array(data['dt_saved'])
        
    del data

outputs = jnp.array(dataset_lst)
#Only consider first 101 (0:100) timesteps
outputs = outputs[:,:101,:,:]

Ns, Nt, Nx, Ny = outputs.shape
print(f"Ns: {Ns}, Nt: {Nt}, Nx: {Nx}, Ny: {Ny}")

#Free up memory
del dataset_lst

#Get the xspan and yspan
xspan = jnp.unique(X)
yspan = jnp.unique(Y)

#Print the xspan and yspan
print(f"xspan: {xspan}, {xspan.shape}")
print("\n")
print(f"yspan: {yspan}, {yspan.shape}")
print("\n")

#Print time vector
tspan = (tspan[:,:Nt]).T
tspan = tspan.flatten()
print(f"tspan: {tspan}, {tspan.shape}")
print("\n")
print(f"Actual dt: {dt_sim}, Saved dt: {dt_saved}")

#-----Peform batched inference-------#
u0 = outputs[:, 0, :, :, None]     #(Ns, Nx, Ny, Nv=1)

# Shared coordinate grid (Nt, Nx, Ny, 3)
# xspan = np.linspace(0, 1, Nx)  # spatial domain - x
# yspan = np.linspace(0, 1, Ny)  # spatial domain - y
# tspan = np.linspace(0, 1, Nt)  # temporal domain

T, X, Y = jnp.meshgrid(tspan, xspan, yspan, indexing="ij")
coords = jnp.stack([T, X, Y], axis=-1)   # (Nt, Nx, Ny, 3)

def fno_forward(params, u0, coords):
    """
    u0: (B, Nx, Ny, 1)    initial condition
    coords: (Nt, Nx, Ny, 3)  shared (t,x,y) grid
    returns: (B, Nt, Nx, Ny, 1)  predicted solution
    """
    B, Nx, Ny, _ = u0.shape
    Nt = coords.shape[0]

    # Broadcast u0 along time
    u0_b = jnp.broadcast_to(u0[:, None, ...], (B, Nt, Nx, Ny, 1))  # (B, Nt, Nx, Ny, 1)

    # Add batch axis to coords
    coords_b = jnp.broadcast_to(coords[None, ...], (B, Nt, Nx, Ny, 3))  # (B, Nt, Nx, Ny, 3)

    # Concatenate along channel axis
    inputs = jnp.concatenate([u0_b, coords_b], axis=-1)  # (B, Nt, Nx, Ny, 4)

    return model_fn(params, inputs)

start_time = time.time()
print("Starting batched FR inference")
BATCH = 8
preds = []
for i in range(0, Ns, BATCH):
    u0_batch = u0[i:i+BATCH]   # (B, Nx, Ny, 1)
    pred = fno_forward(best_params, u0_batch, coords)
    preds.append(pred)
pred_FNO = jnp.concatenate(preds, axis=0)  # (Ns, Nt, Nx, Ny, 1)
print(f"FNO output prediction shape: {pred_FNO.shape}")
print("FR inference complete..")
end_time = time.time()
print(f"Inference time for {pred_FNO.shape[0]} samples: {end_time-start_time} secs")
#-----------------------------------------------#

#Compute relative L2 error
rel_l2_err_u = np.linalg.norm(pred_FNO[...,0] - outputs)/np.linalg.norm(outputs)
print(f"Overall relative L2 error with {modes1}: {rel_l2_err_u}")

#Compute autoreg error
auto_reg_error = []
for t in range(outputs.shape[1]):
    err_val = np.linalg.norm(pred_FNO[:,t,...,0] - outputs[:,t,...])/np.linalg.norm(outputs[:,t,...])
    auto_reg_error.append(err_val)

#Compute statistics
# t = [10, 20, 30, 40, 50, 60, 70, 90, 100]
t = [52, 64, 88, 100]

for t_idx in t:
    print(f"t: {t_idx}, L2 error: {auto_reg_error[t_idx]}")

save = False
if save:
    np.save(result_dir / "u_pred.npy", pred_FNO[...,0])
    
print("Program executed succesfully!")