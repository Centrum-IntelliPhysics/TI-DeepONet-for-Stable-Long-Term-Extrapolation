#!/usr/bin/env python
# coding: utf-8

#Importing all the necessary libraries
import os
import sys
import pickle

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
from tqdm import tqdm

from models import FNO1d, FNO2d
from utils import save_model_params, load_model_params
from utils import dataloader

# seed = 42
seed = np.random.choice(np.arange(99999), size = 1)[0]
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

print(f"u: {u.shape}, xt: {xt.shape}, g_u: {g_u.shape}")

#Include u in the output and remake the grid
g_u = g_u.reshape(ns, nt, nx)   
u_ = u[:, jnp.newaxis, :]
g_u_new = jnp.concatenate([u_, g_u], axis = 1)

#Inputs = u, outputs = g_u_new
inputs = u
outputs = g_u_new
Ns, Nt, Nx = outputs.shape
print(f"Ns: {Ns}, Nt: {Nt}, Nx: {Nx}")

tt = int(Nt//2)
inputs = outputs[:,0,:]

print("For the full rollout mapping")
print(f"Inputs: {inputs.shape}, Outputs: {outputs.shape}")

# Create coordinate grids
print("Creating spatiotemporal meshgrid for fourier transform")
xspan = jnp.linspace(0, 1, Nx)  # spatial domain
tspan = jnp.linspace(0, 5, Nt)  # temporal domain

tspan = tspan[:tt]

# Meshgrid to create 2D coordinate arrays
[T, X] = jnp.meshgrid(tspan, xspan, indexing='ij')

#T and X have (Nt, Nx)
#Broadcast them to (Ns, Nt, Nx), i.e., they are common across all Ns samples
T_tiled = jnp.tile(T[None,:,:], (Ns,1,1))
X_tiled = jnp.tile(X[None,:,:], (Ns,1,1))

print("After broadcasting for Ns samples")
print(f"T_tiled shape: {T_tiled.shape}")
print(f"X_tiled shape: {X_tiled.shape}")

# tile inputs
print("Tiling inputs across temporal dimension to repeat IC")
inputs_tiled = jnp.tile(inputs[:,None,:], (1, tt, 1))
print(f"Inputs tiled shape: {inputs_tiled.shape}")

#Stack all
print("Creating FNO input-output pair..")
inputs_to_FNO = jnp.stack([inputs_tiled, T_tiled, X_tiled], axis=-1)
output_FNO = outputs[:,:tt,:,None]
print(f"Inputs to FNO shape: {inputs_to_FNO.shape}, Outputs FNO shape: {output_FNO.shape}")

#Free up some memory
print("Freeing up memory by deleting inputs_tiled, T_tiled, and X_tiled.")
del inputs_tiled, T_tiled, X_tiled

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
modes1 = 32
modes2 = 32

#Create the FNO-2D model object
fno = FNO2d(in_channels = train_x.shape[-1],
            out_channels = train_y.shape[-1],
            modes1 = modes1,
            modes2 = modes2,
            width = 64,
            n_blocks = 6,
            activation = nn.activation.gelu,  
)
print(f"FNO model: {fno}")
model_fn = jax.jit(fno.apply)

#Instantiate the model params
params = fno.init(jax.random.PRNGKey(seed), train_x[0:1])    #Earlier seed was 42

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
lr_scheduler = optax.schedules.exponential_decay(init_value=lr, transition_steps=5000, decay_rate=0.95)
optimizer = optax.adam(lr_scheduler)
opt_state = optimizer.init(params)

loss_history = []
val_loss_history = []
# batch_size = 64
batch_size = 256    #This is only set to access the compute speed of training
shuffle_key = jax.random.PRNGKey(seed)   #Earlier seed was 80
epochs = int(1e4)
min_val_loss = jnp.inf

result_dir = "./FNO_full_rollout"
filename = f"best_model_params_FNO_{modes1}modes_{seed}.pkl"

for epoch in range(epochs):
    shuffle_key, subkey = jax.random.split(shuffle_key)
    total_loss = 0
    nbatches = 0

    for batch_x, batch_y in tqdm(dataloader(subkey, train_x, train_y, batch_size),
                                 desc=f"Epoch: {epoch}"):
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

    if epoch % 10 == 0:
        print(f"Epoch: {epoch}, Train loss: {loss}, Val loss: {val_loss}, Min Val loss: {min_val_loss}")


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
    plt.savefig(result_dir + f"/loss_plot_{modes1}.jpeg", dpi = 800)
plt.show()

#Save the loss arrays
save_loss = False
if save_loss:
    np.save(result_dir + f"/Train_loss_{modes1}.npy",loss_history)
    np.save(result_dir + f"/Test_loss_{modes2}.npy",val_loss_history)

print("Program executed successfully!")