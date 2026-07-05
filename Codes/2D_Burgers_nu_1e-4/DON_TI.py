#!/usr/bin/env python
# coding: utf-8

import os, sys, pickle
import jax, jaxlib
import jax.numpy as jnp
import matplotlib.pyplot as plt
import scipy
from scipy.io import loadmat
import numpy as np

import torch
import flax
from flax import linen as nn
import optax
from sklearn.model_selection import train_test_split
from typing import Callable, Sequence

from tqdm import tqdm
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
outputs = (jnp.array(dataset["output_field"]))[:1000]
print("Outputs_u shape: ",outputs.shape)

#Free up memory
del dataset

Ns, Nt, Nx, Ny = outputs.shape
print(f"Ns: {Ns}, Nt: {Nt}, Nx: {Nx}, Ny: {Ny}")

#Compute dt value
dt = (1-0)/(Nt-1)
print(f"Computed dt value: {dt}")

tt = int(Nt//3)
print(f"End timestep of training: {tt}")

#Creating the input and output training data
init_timestep = 0
end_timestep = tt

input_data_NN = outputs[:, init_timestep, :, :]    
output_data_NN = outputs[:, init_timestep+1, :, :]

for i in range(init_timestep+1, end_timestep):
    input_data_NN = jnp.vstack((input_data_NN, outputs[:,i,:,:]))
    output_data_NN = jnp.vstack((output_data_NN, outputs[:,i+1,:,:]))
print(input_data_NN.shape, output_data_NN.shape)

#Reshaping the output_data_NN from (ns*nt//2, nx, ny) to (ns*nt//2, nx*ny)
#Input_data_NN remains as it is, i.e., (ns*nt//2, nx, ny)
output_data_NN = output_data_NN.reshape(output_data_NN.shape[0], 
                                        output_data_NN.shape[1]*output_data_NN.shape[2])
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
        
        # #x has shape (ns, nx, ny) - so add channel dimension: (ns, nx, ny, nc)
        x = x[..., jnp.newaxis]
        
        #Convolutional layers
        x = nn.Conv(features = 64, kernel_size = (3,3), strides = 1, padding = "SAME")(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides = (2, 2), padding = "SAME")

        x = nn.Conv(features = 64, kernel_size = (3,3), strides = 1, padding = "SAME")(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides = (2, 2), padding = "SAME")
        
        x = nn.Conv(features = 64, kernel_size = (2, 2), strides = 1, padding = "SAME")(x)
        x = nn.relu(x)
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
    x = inputs[:, 0:1]
    y = inputs[:, 1:2]
    
    freqs = jnp.pi * jnp.linspace(1, max_freq, num_frequencies).reshape(1, -1)
    
    x_feat = jnp.concatenate([jnp.sin(freqs * x), jnp.cos(freqs * x)], axis=-1)
    y_feat = jnp.concatenate([jnp.sin(freqs * y), jnp.cos(freqs * y)], axis=-1)
    
    return jnp.concatenate([inputs, x_feat, y_feat], axis=-1)

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

#Form branch and trunk inputs train
xspan = jnp.linspace(0, 1, Nx)
yspan = jnp.linspace(0, 1, Ny)

#Create for trunk network
[x,y] = jnp.meshgrid(xspan, yspan, indexing = 'ij')
grid = jnp.transpose(jnp.array([x.flatten(), y.flatten()]))
print(grid.shape)
print(grid)


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
num_query_locations = 2
latent_vector_size = 100

branch_network_layer_sizes = [256, 128] + [latent_vector_size]
trunk_network_layer_sizes = [128]*6 + [latent_vector_size]

model = DeepONet(branch_network_layer_sizes, trunk_network_layer_sizes)
model_fn = jax.jit(model.apply)
print(f"Model: {model}")

@jax.jit
def loss_fn(params, branch_inputs, trunk_inputs, gt_outputs, dt):
    
    u_curr = branch_inputs  # Current state input (e.g., u(t))
    u_next = gt_outputs     # Ground truth next state (e.g., u(t+1))

    # Predict the system dynamics (u_dot) at the current state using the model
    u_dot = model_fn(params, u_curr, trunk_inputs)  # Model's predicted rate of change

    # Implementing the 4th-order Runge-Kutta (RK4) time-stepping method
    k1 = u_dot   #(ns*nt, nx*ny)
    k1 = k1.reshape(k1.shape[0], Nx, Ny)   #(ns*nt, nx, ny)
    
    k2 = model_fn(params, u_curr + 0.5 * dt * k1, trunk_inputs)   #(ns*nt, nx*ny)
    k2 = k2.reshape(k2.shape[0], Nx, Ny)    #(ns*nt, nx, ny)
    
    k3 = model_fn(params, u_curr + 0.5 * dt * k2, trunk_inputs)     #(ns*nt, nx*ny)
    k3 = k3.reshape(k3.shape[0], Nx, Ny)    #(ns*nt, nx, ny)
    
    k4 = model_fn(params, u_curr + dt * k3, trunk_inputs)    #(ns*nt, nx*ny)
    k4 = k4.reshape(k4.shape[0], Nx, Ny)    #(ns*nt, nx, ny)
    
    # Calculate the next state using RK4
    u_pred_next = u_curr + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)  #(ns*nt, nx, ny)
    
    #Reshape u_pred_next to match compatibility of u_next
    u_pred_next = u_pred_next.reshape(u_pred_next.shape[0], Nx*Ny)

    # Compute the Mean Squared Error loss between the predicted and ground truth next states
    mse_loss = jnp.mean(jnp.square(u_pred_next - u_next))

    # Compute L2 err
    L2_err = jnp.linalg.norm(u_pred_next - u_next)/jnp.linalg.norm(u_next)
    
    return mse_loss, L2_err


@jax.jit
def update(params, branch_inputs, trunk_inputs, gt_outputs, opt_state, dt):
    (loss, L2_err), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, branch_inputs, 
                                                                      trunk_inputs, gt_outputs, dt)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss, L2_err


#Freeing memory by deleting inputs and outputs
del outputs

# Initialize model parameters
key = jax.random.PRNGKey(seed)   #42
params = model.init(key, branch_inputs_train[0:1, ...], trunk_inputs_train[0:1, ...])

# # Optimizer setup
lr_scheduler = optax.schedules.exponential_decay(init_value=1e-3, transition_steps=5000, decay_rate=0.95)
optimizer = optax.adam(learning_rate=lr_scheduler)
opt_state = optimizer.init(params)

training_loss_history = []
test_loss_history = []
num_epochs = int(2e5)
# batch_size = 64
batch_size = 32    #This value is only set to assess the compute speed of training

min_test_loss = jnp.inf

filepath = 'DeepONet_NODE'
filename = f"model_params_best_v1_{seed}.pkl"

for epoch in tqdm(range(num_epochs), desc="Training Progress"):

    #Perform mini-batching
    shuffled_indices = jax.random.permutation(jax.random.PRNGKey(epoch), branch_inputs_train.shape[0])
    batch_indices = shuffled_indices[:batch_size]

    branch_inputs_train_batch = branch_inputs_train[batch_indices]
    outputs_train_batch = outputs_train[batch_indices]

    # Update the parameters and optimizer state
    params, opt_state, loss, _ = update(
        params=params,
        branch_inputs=branch_inputs_train_batch,
        trunk_inputs=trunk_inputs_train,
        gt_outputs=outputs_train_batch,
        opt_state=opt_state,
        dt=dt
    )

    training_loss_history.append(loss)
    
    #Do predictions on the test data simultaneously
    test_mse_loss, test_L2_err = loss_fn(params = params, 
                            branch_inputs = branch_inputs_test, 
                            trunk_inputs = trunk_inputs_test, 
                            gt_outputs = outputs_test,
                            dt=dt)
    test_loss_history.append(test_mse_loss)
    
    #Save the params of the best model encountered till now
    if test_mse_loss < min_test_loss:
        best_params = params
        # save_model_params(best_params, path = filepath, filename = filename)
        min_test_loss = test_mse_loss
    
    #Print the train and test loss history every 1000 epochs
    if epoch % 1000 == 0:
        print(f"Epoch: {epoch}, train_loss: {loss}, test_loss: {test_mse_loss}, " 
              f"test_L2_err: {test_L2_err}, best_test_loss: {min_test_loss}")


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
# plt.savefig(filepath + "/loss_plot_v5_RK.jpeg", dpi = 800)
plt.close()

#Save the loss arrays
save = False
if save:
    np.save(filepath + "/Train_loss.npy",training_loss_history)
    np.save(filepath + "/Test_loss.npy",test_loss_history)


# @jax.jit
def inference_ab(u_curr, u_prev, trunk_inputs_test, dt):
    
    # Step 1: Apply the predictor (Adams-Bashforth) using u_curr and u_prev
    u_dot_curr = model_fn(best_params, u_curr, trunk_inputs_test)  # Predict the rate of change at u_curr
    u_dot_prev = model_fn(best_params, u_prev, trunk_inputs_test)  # Predict the rate of change at u_prev
    
    #Reshaping u_dot_curr and u_dot_prev to broadcast compatible with u_curr
    u_dot_curr = u_dot_curr.reshape(u_dot_curr.shape[0], Nx, Ny)
    u_dot_prev = u_dot_prev.reshape(u_dot_prev.shape[0], Nx, Ny)
    
    # Adams-Bashforth predictor (using previous two points)
    u_pred = u_curr + dt * (1.5 * u_dot_curr - 0.5 * u_dot_prev)
    
    # Step 2: Apply the corrector (Adams-Moulton) using the predicted u_pred
    u_dot_pred = model_fn(best_params, u_pred, trunk_inputs_test)  # Predict the rate of change at u_pred
    
    #Reshaping u_dot_pred to broadcast compatible with u_curr, u_dot_curr, u_dot_prev
    u_dot_pred = u_dot_pred.reshape(u_dot_pred.shape[0], Nx, Ny)
    
    # Adams-Moulton corrector (refine the prediction using u_pred)
    u_next = u_curr + dt * (5/12 * u_dot_pred + 8/12 * u_dot_curr - 1/12 * u_dot_prev)
    
    return u_next


# @jax.jit
def inference_rk(u_curr, trunk_inputs_test, dt):
    
    # Predict the system dynamics (u_dot) at the current state using the model
    u_dot = model_fn(params, u_curr, trunk_inputs_test)  # Model's predicted rate of change

    # Implementing the 4th-order Runge-Kutta (RK4) time-stepping method
    k1 = u_dot   #(ns*nt, nx*ny)
    k1 = k1.reshape(k1.shape[0], Nx, Ny)   #(ns*nt, nx, ny)
    
    k2 = model_fn(params, u_curr + 0.5 * dt * k1, trunk_inputs_test)   #(ns*nt, nx*ny)
    k2 = k2.reshape(k2.shape[0], Nx, Ny)    #(ns*nt, nx, ny)
    
    k3 = model_fn(params, u_curr + 0.5 * dt * k2, trunk_inputs_test)     #(ns*nt, nx*ny)
    k3 = k3.reshape(k3.shape[0], Nx, Ny)    #(ns*nt, nx, ny)
    
    k4 = model_fn(params, u_curr + dt * k3, trunk_inputs_test)    #(ns*nt, nx*ny)
    k4 = k4.reshape(k4.shape[0], Nx, Ny)    #(ns*nt, nx, ny)
    
    # Calculate the next state using RK4
    u_pred_next = u_curr + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)  #(ns*nt, nx, ny)
    
    return u_pred_next


def run_inference(initial_u, trunk_inputs_test, n_steps, method, dt):
    u_states = np.zeros(shape = (initial_u.shape[0], Nt, Nx, Ny))  # List to store the states over time
    u_states[:,0,:,:] = initial_u
    
    # Initialize the previous state (this could be your u_0 and u_1, etc.)
    u_prev = initial_u  # Set the previous state to the initial state
    u_curr = initial_u  # Set the current state to the initial state
    
    for i in range(1, n_steps):
        
        if method == "AB":
            # Perform one inference step using the multistep method
            u_next = inference_ab(u_curr, u_prev, trunk_inputs_test, dt)

            # Append the predicted state to the list
            u_states[:, i, :, :] = u_next

            # Update previous and current states for the next step
            u_prev = u_curr
            u_curr = u_next
        elif method == "RK":
            #Perform one inference step using the rk-4 method
            u_next = inference_rk(u_curr, trunk_inputs_test, dt)
            
            # Append the predicted state to the list
            u_states[:, i, :, :] = u_next
            
            #Update the current state for the next step
            u_curr = u_next
    
    return u_states

print("At inference...")
# Load the best model parameters
import time
best_params = load_model_params(path=filepath, filename=filename)
print(f"Best params loaded: {filename}")

#Read the 2D Burgers' dataset
base_path = "/home/dnayak2/data_sgoswam4/Dibya/Datasets/2D_Burgers/low_viscosity"
dataset = torch.load(os.path.join(base_path, "2D_scalar_Burgers_N1000_64x64.pth"))
print("2D Burgers dataset loaded!")

#For u-velocity field
outputs = (jnp.array(dataset["output_field"]))[:1000]
print("Outputs_u shape: ",outputs.shape)

Ns, Nt, Nx, Ny = outputs.shape
print(f"Ns: {Ns}, Nt: {Nt}, Nx: {Nx}, Ny: {Ny}")

del dataset

method = "AB"
print(f"Inference scheme being used: {method}")

print("Compute the no. of steps for inference!")
NSTEPS = int((1/dt) + 1)
print(f"Inference dt: {dt}, nsteps: {NSTEPS}")

u_curr = outputs[:, 0, :, :]
start_time = time.time()
u_pred = run_inference(u_curr, trunk_inputs_test, n_steps=NSTEPS, method = method, dt=dt)
end_time = time.time()

print(f"Total time of inference for {u_pred.shape[0]} samples: ",end_time-start_time)
print(f"Prediction shape: {u_pred.shape}, Outputs shape: {outputs.shape}")

overall_rel_l2_err = jnp.linalg.norm(u_pred - outputs)/jnp.linalg.norm(outputs)
print(f"Overall relative L2 error: {overall_rel_l2_err}")

#Randomly selecting "size" number of samples out of the test dataset
indices = np.random.choice(np.arange(u_pred.shape[0]), size = 3, replace = 'False')

x_test = jnp.linspace(0, 1, Nx)
y_test = jnp.linspace(0, 1, Ny)
t_test = jnp.linspace(0, 1, Nt)

t_query = [25, 50, 75, -1]

for idx in indices:
    n_cols = len(t_query)
    n_rows = 3  # Predicted, Actual, Error
    plt.figure(figsize=(4 * n_cols, 3 * n_rows))

    for j, t in enumerate(t_query):
        # --- Predicted --- #
        plt.subplot(n_rows, n_cols, 1 + j)
        contour1 = plt.contourf(x_test, y_test, u_pred[idx, t, :, :].T, levels=20, cmap='jet')
        plt.xlabel("x", fontsize=12)
        plt.ylabel("y", fontsize=12)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        cbar1 = plt.colorbar()
        cbar1.ax.tick_params(labelsize=10)
        if j == 0:
            plt.ylabel("Predicted", fontsize=14)
        plt.title(f"Timestep {t}", fontsize=14)

        # --- Actual --- #
        plt.subplot(n_rows, n_cols, 1 + n_cols + j)
        contour2 = plt.contourf(x_test, y_test, outputs[idx, t, :, :].T, levels=20, cmap='jet')
        plt.xlabel("x", fontsize=12)
        plt.ylabel("y", fontsize=12)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        cbar2 = plt.colorbar()
        cbar2.ax.tick_params(labelsize=10)
        if j == 0:
            plt.ylabel("Actual", fontsize=14)

        # --- Error --- #
        plt.subplot(n_rows, n_cols, 1 + 2 * n_cols + j)
        contour3 = plt.contourf(x_test, y_test, 
                                jnp.abs(u_pred[idx, t, :, :].T - outputs[idx, t, :, :].T), 
                                cmap='Purples')
        plt.xlabel("x", fontsize=12)
        plt.ylabel("y", fontsize=12)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        cbar3 = plt.colorbar()
        cbar3.ax.tick_params(labelsize=10)
        if j == 0:
            plt.ylabel("Error", fontsize=14)

    plt.suptitle(f"Sample Idx: {idx}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    # plt.savefig(filepath + f"/Contour_plots_{idx}_{method}.jpeg", dpi=800)
    plt.show()


#Plotting the relative L2 error obtained at every timestep to show accummulation of autoregressive error
auto_reg_error = []
num_time_steps = Nt

for i in range(num_time_steps):
    l2_error = jnp.linalg.norm(u_pred[:,i,:,:] - outputs[:,i,:,:])/jnp.linalg.norm(outputs[:,i,:,:])
    auto_reg_error.append(l2_error)

#Compute statistics
t = [60, 70, 90, 100]

print("----Extrapolation errors----")
for t_idx in t:
    print(f"t: {t_idx}, L2 error: {auto_reg_error[t_idx]}")

#Save the auto_reg_error array for comparing with NODE approach
save = False
if save:
    np.save(filepath + f"/Auto_reg_error_with_NODE_{method}_new.npy", auto_reg_error)
    np.save(filepath + f"/u_pred_{method}_new.npy", u_pred)
print("Program executed successfully!")