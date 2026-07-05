#This notebook implements a DeepONet from scratch using FLAX

#importing all necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import h5py
import time

import jax
import jax.numpy as jnp
from jax import random
from jax import jit, vmap, pmap, grad, value_and_grad

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

from models import branch_net, trunk_net, SinBranchNet, SinTrunkNet

print("Program started...")
# seed = 42
seed = np.random.choice(np.arange(99999), size = 1, replace=True)[0]
print("Seed: ",seed)
np.random.seed(seed)
key = jax.random.PRNGKey(seed)

base_path = "/home/dnayak2/data_sgoswam4/Dibya/Datasets/1D_KS/"
outputs = (scipy.io.loadmat(os.path.join(base_path, "KS_simple.mat"))["u_out"])[:3000]

Ns, Nt, Nx = outputs.shape
print(f"Ns: {Ns}, Nt: {Nt}, Nx: {Nx}")

inputs = outputs[:,0,:]

def visualize_data(idx, inputs, outputs):
    plt.figure(figsize = (7,3))
    plt.subplot(1, 2, 1)
    plt.plot(np.linspace(0, 1, Nx), inputs[idx])
    plt.xlabel("t")
    plt.ylabel("f(t)")
    plt.title("Initial Condition")

    plt.subplot(1, 2, 2)
    plt.contourf(np.linspace(0, 1, Nt), np.linspace(0, 1, Nx), outputs[idx].T, cmap="viridis")
    plt.colorbar()
    plt.xlabel("t")
    plt.ylabel("x")
    plt.title("Output field")
    plt.show()

# idx=np.random.choice(Ns, replace=True)
# visualize_data(idx, inputs, outputs)

#Only consider half of the temporal domain
tt=Nt//2
outputs = outputs[:,:tt,:]

def normalize_inputs(X, eps=1e-8):
    Ns, Nt = X.shape
    
    X_mean = np.mean(X, axis=0, keepdims=True)
    X_std = np.std(X, axis=0, keepdims=True)
    
    X_scaled = (X - X_mean)/(X_std + eps)
    
    return X_scaled, X_mean, X_std

def normalize_outputs(X, eps=1e-8):
    Ns, Nt, Nx = X.shape
    
    X_reshaped = X.reshape(Ns, Nt*Nx)
    X_mean = np.mean(X_reshaped, axis=0, keepdims=True)
    X_std = np.std(X_reshaped, axis=0, keepdims=True)
    
    X_reshaped_scaled = (X_reshaped - X_mean)/(X_std + eps)
    X_scaled = X_reshaped_scaled.reshape(Ns, Nt, Nx)
    
    return X_scaled, X_mean, X_std

normalize = False
if normalize:
    inputs, inputs_mean, inputs_std = normalize_inputs(inputs)
    outputs, outputs_mean, outputs_std = normalize_outputs(outputs)

#Create the grid with coordinate pairs (t,x) to feed into trunk network
tspan = jnp.linspace(0, 1, Nt)
xspan = jnp.linspace(0, 1, Nx)

#Take only half of the temporal domain
tspan = tspan[:tt]

[t,x] = jnp.meshgrid(tspan, xspan, indexing = 'ij')
grid = jnp.transpose(jnp.array([t.flatten(), x.flatten()]))
print(grid.shape)
print(grid)


# Split the data into training and testing samples
inputs_train, inputs_test, outputs_train, outputs_test = \
            train_test_split(inputs, outputs, test_size=0.2, random_state=seed)

outputs_train = outputs_train.reshape(outputs_train.shape[0], tt*Nx)
outputs_test = outputs_test.reshape(outputs_test.shape[0], tt*Nx)

# Check the shapes of the subsets
print("Shape of inputs_train:", inputs_train.shape)
print("Shape of inputs_test:", inputs_test.shape)
print("Shape of outputs_train:", outputs_train.shape)
print("Shape of outputs_test:", outputs_test.shape)


#Network Inputs
branch_inputs_train = inputs_train    ##--- (2000,101)
trunk_inputs_train = grid             ##--- (10201,2)

#Inspecting the shapes
print("Shape of train branch inputs: ",branch_inputs_train.shape)
print("Shape of train trunk inputs: ",trunk_inputs_train.shape)
print("Shape of train output: ",outputs_train.shape)


branch_inputs_test = inputs_test      ##--- (500, 101)
trunk_inputs_test = grid              ##--- (10201,2)

#Inspecting the shapes
print("Shape of test branch inputs: ",branch_inputs_test.shape)
print("Shape of test trunk inputs: ",trunk_inputs_test.shape)
print("Shape of test output: ",outputs_test.shape)


def add_fourier_features(inputs, num_frequencies=10, max_freq=10):
    t, x = inputs[:, 0:1], inputs[:, 1:2]
    freqs = jnp.pi * jnp.linspace(1, max_freq, num_frequencies).reshape(1, -1)
    
    t_feat = jnp.concatenate([jnp.sin(freqs * t), jnp.cos(freqs * t)], axis=-1)
    x_feat = jnp.concatenate([jnp.sin(freqs * x), jnp.cos(freqs * x)], axis=-1)
    
    return jnp.concatenate([inputs, t_feat, x_feat], axis=-1)


class DeepONet(nn.Module):

    branch_net_config: Sequence[int]
    trunk_net_config: Sequence[int]
    use_Fourier_feat: bool = True

    def setup(self):
        self.branch_net = branch_net(self.branch_net_config, activation=nn.activation.silu)
        self.trunk_net = trunk_net(self.trunk_net_config, activation=jnp.sin)

    def __call__(self, x_branch, x_trunk):
        
        if self.use_Fourier_feat:
            #Encode x_trunk into fourier features
            x_trunk = add_fourier_features(x_trunk)
        
        #Vectorize over multiple samples of input functions
        branch_outputs = self.branch_net(x_branch)
        
        #Vectorize over multiple query points
        trunk_outputs = self.trunk_net(x_trunk)
        
        inner_product = jnp.einsum('ik,jk->ij', branch_outputs, trunk_outputs)

        return inner_product


num_sensor_locations = branch_inputs_train.shape[1]
num_query_locations = 2
latent_vector_size = 100

branch_network_layer_sizes = [128]*7 + [latent_vector_size]
trunk_network_layer_sizes = [128]*7 + [latent_vector_size]

model = DeepONet(branch_net_config = branch_network_layer_sizes, 
                                      trunk_net_config = trunk_network_layer_sizes)
print(f"Model: {model}")

model_fn = jax.jit(model.apply)

# Check if it's a valid Flax model
print(isinstance(model, nn.Module))  # Should return True if it's a valid Flax model

def save_model_params(params, path, filename):
    
    #Create output directory for saving model params
    if not os.path.exists(path):
        os.makedirs(path)
    
    save_path = os.path.join(path, filename)
    with open(save_path, 'wb') as f:
        pickle.dump(params, f)

def load_model_params(path, filename):
    load_path = os.path.join(path, filename)
    with open(load_path, 'rb') as f:
        params = pickle.load(f)
    return params


# Define the training process from here
@jax.jit
def loss_fn(params, branch_inputs, trunk_inputs, gt_outputs):
    predictions = model_fn(params, branch_inputs,trunk_inputs)
    mse_loss = jnp.mean(jnp.square(predictions - gt_outputs))
    return mse_loss

@jax.jit
def update(params, branch_inputs, trunk_inputs, gt_outputs, opt_state):
    loss, grads = jax.value_and_grad(loss_fn)(params, branch_inputs, trunk_inputs, gt_outputs)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

# Initialize model parameters
params = model.init(key, branch_inputs_train[0:1], trunk_inputs_train[0:1])

# # Optimizer setup
lr_scheduler = optax.schedules.exponential_decay(init_value=1e-3, transition_steps=5000, decay_rate=0.95)
optimizer = optax.adamw(learning_rate=lr_scheduler, weight_decay=1e-4)
opt_state = optimizer.init(params)

training_loss_history = []
test_loss_history = []
num_epochs = int(1.5e5)
batch_size = 128

min_test_loss = jnp.inf

filepath = 'DeepONet_full_rollout'
filename = f'model_params_best_{seed}.pkl'

for epoch in tqdm(range(num_epochs), desc='Training Progress'):

    #Perform mini-batching
    shuffled_indices = jax.random.permutation(random.PRNGKey(epoch), branch_inputs_train.shape[0])
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
    test_mse_loss = loss_fn(params = params, 
                            branch_inputs = branch_inputs_test, 
                            trunk_inputs = trunk_inputs_test, 
                            gt_outputs = outputs_test)
    test_loss_history.append(test_mse_loss)
    
    #Save the params of the best model encountered till now
    if test_mse_loss < min_test_loss:
        best_params = params
        # save_model_params(best_params, path = filepath, filename = filename)
        min_test_loss = test_mse_loss
    
    #Print the train and test loss history every 1000 epochs
    if epoch % 500 == 0:
        print(f"Epoch: {epoch}, train_loss: {loss}, test_loss: {test_mse_loss}, best_test_loss: {min_test_loss}")


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

save = False
if save:
    plt.savefig(filepath + "/loss.jpeg", dpi = 500)
plt.show()


#Save loss arrays
save = False
if save:
    np.save(filepath + "/train_loss.npy", training_loss_history)
    np.save(filepath + "/test_loss.npy", test_loss_history)


# Predictions
print("At inference...")
#Import the best model saved after full training
outputs = (scipy.io.loadmat(os.path.join(base_path, "KS_simple.mat"))["u_out"])[:3000]
best_params = load_model_params(path = filepath, filename = filename)
print(f"Best params loaded: {filename}")

branch_input_new = outputs[:, 0, :]
tspan = jnp.linspace(0, 1, Nt)
xspan = jnp.linspace(0, 1, Nx)

[t_new, x_new] = jnp.meshgrid(tspan, xspan, indexing = 'ij')
grid = jnp.transpose(jnp.array([t_new.flatten(), x_new.flatten()]))
print(grid.shape)
print(grid)

start_time = time.time()
predictions_outputs_new = model_fn(best_params, branch_input_new, grid)
end_time = time.time()
print(f"Total inferencing time for {predictions_outputs_new.shape[0]} samples: {end_time-start_time}")
predictions_outputs_new = predictions_outputs_new.reshape(predictions_outputs_new.shape[0], Nt, Nx)

#Randomly selecting "size" number of samples out of the test dataset
# random_samples = np.random.choice(np.arange(outputs.shape[0]), size = 3, replace = 'True')

random_samples = [70, 140, 320]
for i in random_samples:
    
    prediction_i = predictions_outputs_new[i, :, :]
    target_i = outputs[i, :, :]
    error_i = np.abs(prediction_i - target_i)
    
    plt.figure(figsize = (12,3))
    
    plt.subplot(1,3,1)
    contour1 = plt.contourf(tspan, xspan, prediction_i.T, levels = 20, cmap = 'jet')
    cbar1 = plt.colorbar()
    cbar1.ax.tick_params(labelsize = 12)
    plt.xlabel("t", fontsize = 14)
    plt.ylabel("x", fontsize = 14)
    plt.xticks(fontsize = 12)
    plt.yticks(fontsize = 12)
    plt.title("Predicted", fontsize = 16)
    
    plt.subplot(1,3,2)
    contour2 = plt.contourf(tspan, xspan, target_i.T, levels = 20, cmap = 'jet')
    cbar2 = plt.colorbar()
    cbar2.ax.tick_params(labelsize = 12)
    plt.xlabel("t", fontsize = 14)
    plt.ylabel("x", fontsize = 14)
    plt.xticks(fontsize = 12)
    plt.yticks(fontsize = 12)
    plt.title("Actual", fontsize = 16)
  
    
    plt.subplot(1,3,3)
    contour3 = plt.contourf(tspan, xspan, error_i.T, levels = 20, cmap = 'Wistia')
    cbar3 = plt.colorbar()
    cbar3.ax.tick_params(labelsize = 12)
    plt.xlabel("t", fontsize = 14)
    plt.ylabel("x", fontsize = 14)
    plt.xticks(fontsize = 12)
    plt.yticks(fontsize = 12)
    plt.title("Error", fontsize = 16)
    
    plt.tight_layout()
    # plt.savefig(filepath + f"/Contour_plots_sidx{i}.jpeg", dpi = 800)
    plt.show()

#Compute autoregressive errors
overall_rel_L2_err = np.linalg.norm(predictions_outputs_new - outputs)/np.linalg.norm(outputs)
print(f"Overall relative L2 error: {overall_rel_L2_err}")

auto_reg_error = []
num_time_steps = Nt

for i in range(num_time_steps):
    l2_error = np.linalg.norm(predictions_outputs_new[:,i,:] - outputs[:,i,:])/np.linalg.norm(outputs[:,i,:])
    auto_reg_error.append(l2_error)

print(f"Length of auto_reg_error: {len(auto_reg_error)}")

print("----Extrapolation errors----")
# Compute statistics
t = [180, 210, 270, 300]

for t_idx in t:
    print(f"t: {t_idx}, L2 error: {auto_reg_error[t_idx-1]}")

#Save the auto_reg_error array, u_pred, outputs, for comparing with NODE approach
save = False
if save:
    np.save(filepath + "/Auto_reg_error_full_rollout.npy", auto_reg_error)
    np.save(filepath + "/u_actual.npy", outputs)
    np.save(filepath + "/u_pred.npy", predictions_outputs_new)
print("Program executed successfully!")