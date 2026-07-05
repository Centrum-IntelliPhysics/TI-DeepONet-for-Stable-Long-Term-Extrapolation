#!/usr/bin/env python
# coding: utf-8

import os, sys, pickle
import jax, jaxlib
import jax.numpy as jnp
import matplotlib.pyplot as plt
import scipy
from scipy.io import loadmat
import numpy as np

import flax
from flax import linen as nn
import optax
from sklearn.model_selection import train_test_split
from typing import Callable, Sequence

from tqdm import tqdm

import time
from models import branch_net, trunk_net

print("Program started...")
# seed = 42
seed = np.random.choice(np.arange(99999), size=1, replace=True)[0]
print("Seed: ",seed)
np.random.seed(seed)
key = jax.random.PRNGKey(seed)

base_path = "/home/dnayak2/data_sgoswam4/Dibya/Datasets/1D_KS/"
outputs = (loadmat(os.path.join(base_path, "KS_simple.mat"))["u_out"])[:3000]

Ns, Nt, Nx = outputs.shape
print(f"Ns: {Ns}, Nt: {Nt}, Nx: {Nx}")

#Creating the input and output training data
tt = int(Nt//2)
init_timestep = 0
end_timestep = tt
dt = (1-0)/(Nt-1)
print(f"Computed dt: {dt}")

input_data_NN = outputs[:,init_timestep,:]
output_data_NN = outputs[:,init_timestep+1,:]

for i in range(init_timestep+1, end_timestep):
    input_data_NN = jnp.vstack((input_data_NN, outputs[:,i,:]))
    output_data_NN = jnp.vstack((output_data_NN, outputs[:,i+1,:]))

print(input_data_NN.shape, output_data_NN.shape)


input_data_NN_train, input_data_NN_test, output_data_NN_train, output_data_NN_test = \
                        train_test_split(input_data_NN, output_data_NN, test_size = 0.2, random_state = 42)
print(input_data_NN_train.shape, input_data_NN_test.shape, output_data_NN_train.shape, output_data_NN_test.shape)

del input_data_NN, output_data_NN

def add_fourier_features(inputs, num_frequencies=10, max_freq=10):
    x = inputs[:, 0:1]
    freqs = jnp.pi * jnp.linspace(1, max_freq, num_frequencies).reshape(1, -1)
    x_feat = jnp.concatenate([jnp.sin(freqs * x), jnp.cos(freqs * x)], axis=-1)
    
    return jnp.concatenate([inputs, x_feat], axis=-1)

class DeepONet(nn.Module):

    branch_net_config: Sequence[int]
    trunk_net_config: Sequence[int]
    use_Fourier_feat: bool = True

    def setup(self):
        self.branch_net = branch_net(self.branch_net_config, activation=nn.activation.tanh)
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


class LearnableRK4(nn.Module):
    hidden_dim: int = 64
    
    @nn.compact
    def __call__(self, u_curr):
        x = u_curr
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.activation.tanh(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.activation.tanh(x)
        x = nn.Dense(4)(x)
        x = nn.activation.softmax(x)
        return x


def dynamic_rk4_step(u_curr, model_fn, params, model_rk_fn, rk_params, trunk_inputs, dt):
    
    alpha = jax.vmap(model_rk_fn, in_axes = (None, 0))(rk_params, u_curr)         #(Shape: (batch_size,4)
    
    #Extract the coefficients  - each with shape (batch_size, 1)
    alpha1 = alpha[:,0:1]
    alpha2 = alpha[:,1:2]
    alpha3 = alpha[:,2:3]
    alpha4 = alpha[:,3:]
    
    k1 = model_fn(params, u_curr, trunk_inputs)
    k2 = model_fn(params, u_curr + 0.5 * dt * k1, trunk_inputs)
    k3 = model_fn(params, u_curr + 0.5 * dt * k2, trunk_inputs)
    k4 = model_fn(params, u_curr + dt * k3, trunk_inputs)

    u_next = u_curr + dt * (alpha1 * k1 + alpha2 * k2 + alpha3 * k3 + alpha4 * k4)
    return u_next, alpha


@jax.jit
def loss_fn(params, rk_params, branch_inputs, trunk_inputs, gt_outputs, dt):
    
    u_curr = branch_inputs  # Current state input (e.g., u(t))
    u_next = gt_outputs     # Ground truth next state (e.g., u(t+1))
    
    u_pred_next, alpha = dynamic_rk4_step(u_curr, model_fn, params, model_rk_fn, rk_params, trunk_inputs, dt)

    # Compute the Mean Squared Error loss between the predicted and ground truth next states
    mse_loss = jnp.mean(jnp.square(u_pred_next - u_next))
    
    return (mse_loss, alpha)


@jax.jit
def update(params, rk_params, branch_inputs, trunk_inputs, gt_outputs, opt_state, opt_state_rk, dt):
    
    #Update for DeepONet params
    (loss, _), grads = \
            jax.value_and_grad(loss_fn, argnums = 0, has_aux=True)(params, rk_params, branch_inputs, 
                                                                   trunk_inputs, gt_outputs, dt)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    
    #Update for RK params
    (_, alpha), rk_grads = \
        jax.value_and_grad(loss_fn, argnums = 1, has_aux=True)(params, rk_params, branch_inputs, 
                                                                trunk_inputs, gt_outputs, dt)
    updates_rk, opt_state_rk = optimizer_rk.update(rk_grads, opt_state_rk)
    rk_params = optax.apply_updates(rk_params, updates_rk)
    
    return params, rk_params, opt_state, opt_state_rk, loss, alpha


#Form branch and trunk inputs train
grid = jnp.linspace(0, 1, Nx)[:,jnp.newaxis]
branch_inputs_train = input_data_NN_train
trunk_inputs_train = grid
outputs_train = output_data_NN_train
print("TRAINING DATA")
print(branch_inputs_train.shape, trunk_inputs_train.shape, outputs_train.shape)


#For branch and trunk inputs test
branch_inputs_test = input_data_NN_test
trunk_inputs_test = grid
outputs_test = output_data_NN_test
print("TESTING DATA")
print(branch_inputs_test.shape, trunk_inputs_test.shape, outputs_test.shape)


#DeepONet settings
num_sensor_locations = branch_inputs_train.shape[1]
num_query_locations = 1
latent_vector_size = 100

branch_network_layer_sizes = [128]*7 + [latent_vector_size]
trunk_network_layer_sizes = [128]*7 + [latent_vector_size]

model = DeepONet(branch_network_layer_sizes, trunk_network_layer_sizes)
model_fn = jax.jit(model.apply)

model_rk = LearnableRK4()
model_rk_fn = jax.jit(model_rk.apply)


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


# Initialize model parameters
key, subkey = jax.random.split(key)

params = model.init(key, branch_inputs_train, trunk_inputs_train)
rk_params = model_rk.init(subkey, branch_inputs_train)

# Optimizer setup

#Initialize optimizer for DeepONet
lr_scheduler = optax.schedules.exponential_decay(init_value=1e-3, transition_steps=5000, decay_rate=0.95)
optimizer = optax.adamw(learning_rate=lr_scheduler, weight_decay=1e-4)
opt_state = optimizer.init(params)

#Initialize optimizer for RK4
lr_scheduler = optax.schedules.exponential_decay(init_value=2e-3, transition_steps=5000, decay_rate=0.95)
optimizer_rk = optax.adam(learning_rate=lr_scheduler)
opt_state_rk = optimizer_rk.init(rk_params)

training_loss_history = []
test_loss_history = []
num_epochs = int(1.75e5)
batch_size = 128

min_test_loss = jnp.inf

filepath = 'DeepONet_NODE_learnableRK'
filename = f"model_params_best_{seed}.pkl"

alpha_lst = []

for epoch in tqdm(range(num_epochs), desc="Training Progress"):

    #Perform mini-batching
    shuffled_indices = jax.random.permutation(jax.random.PRNGKey(epoch), branch_inputs_train.shape[0])
    batch_indices = shuffled_indices[:batch_size]

    branch_inputs_train_batch = branch_inputs_train[batch_indices]
    outputs_train_batch = outputs_train[batch_indices]

    # Update the parameters and optimizer state
    params, rk_params, opt_state, opt_state_rk, loss, alpha = update(
        params=params,
        rk_params=rk_params,
        branch_inputs=branch_inputs_train_batch,
        trunk_inputs=trunk_inputs_train,
        gt_outputs=outputs_train_batch,
        opt_state=opt_state,
        opt_state_rk=opt_state_rk,
        dt=dt
    )

    training_loss_history.append(loss)
    alpha_lst.append(alpha)
    
    #Do predictions on the test data simultaneously
    test_mse_loss, _ = loss_fn(params = params, 
                            rk_params = rk_params,
                            branch_inputs = branch_inputs_test, 
                            trunk_inputs = trunk_inputs_test, 
                            gt_outputs = outputs_test,
                            dt=dt)
    test_loss_history.append(test_mse_loss)
    
    #Save the params of the best model encountered till now
    if test_mse_loss < min_test_loss:
        best_params = {"deeponet_params": params, "rk_params": rk_params}
        # save_model_params(best_params, path = filepath, filename = filename)
        min_test_loss = test_mse_loss
    
    #Print the train and test loss history every 1000 epochs
    if epoch % 1000 == 0:
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
    plt.savefig(filepath + "/loss_plot.jpeg", dpi = 800)
plt.show()

#Save the loss arrays
if False:
    np.save(filepath + "/Train_loss.npy",training_loss_history)
    np.save(filepath + "/Test_loss.npy",test_loss_history)


##Need to modify inferencing code as now we can use the learnt RK4 coefficients and do RK4 in prediction
#Instead of AB-AM predictor-corrector
@jax.jit
def inference(u_curr, trunk_inputs_test, dt):
    u_next = dynamic_rk4_step(u_curr, model_fn, best_params, model_rk_fn, 
                              best_rk_params, trunk_inputs_test, dt)
    return u_next


def run_inference(initial_u, trunk_inputs_test, n_steps, dt):
    u_states = np.zeros_like(outputs)  # List to store the states over time
    u_states[:,0,:] = initial_u
    
    # Initialize the current state (this could be your u_0 and u_1, etc.)
    u_curr = initial_u  # Set the current state to the initial state
    
    for i in range(1, n_steps):
        # Perform one inference step using the multistep method
        u_next = inference(u_curr, trunk_inputs_test, dt)
        
        # Append the predicted state to the list
        u_states[:, i, :] = u_next[0]
        
        # Update current states for the next step
        u_curr = u_next[0]
    return u_states

print("At inference...")
# Load the best model parameters
import time

outputs = (loadmat(os.path.join(base_path, "KS_simple.mat"))["u_out"])[:3000]
best_full_params = load_model_params(path=filepath, filename=filename)
print(f"Best params loaded: {filename}")
best_params = best_full_params['deeponet_params']
best_rk_params = best_full_params['rk_params']

Ns, Nt, Nx = outputs.shape
u_curr = outputs[:, 0, :]
print(u_curr.shape)

start_time = time.time()
u_pred = run_inference(u_curr, trunk_inputs_test, n_steps=Nt, dt=dt)
end_time = time.time()
print(f"TI(L)-DeepONet inference time for {u_pred.shape[0]} samples: {end_time-start_time}")

print(u_pred.shape, outputs.shape)
overall_rel_L2_err = np.linalg.norm(u_pred - outputs)/np.linalg.norm(outputs)
print(f"Overall relative L2 error: {overall_rel_L2_err}")

indices = [70, 140, 320]
x_test = jnp.linspace(0,1,Nx)
t_test = jnp.linspace(0,1,Nt)

for idx in indices:
    plt.figure(figsize = (12,3))
    plt.subplot(1, 3, 1)
    contour1 = plt.contourf(t_test, x_test, u_pred[idx, :, :].T, levels = 20, cmap = 'jet')
    plt.xlabel("t", fontsize = 14)
    plt.ylabel("x", fontsize = 14)
    plt.yticks(fontsize = 12)
    plt.xticks(fontsize = 12)
    cbar1 = plt.colorbar()
    cbar1.ax.tick_params(labelsize=12)
    plt.title("Predicted", fontsize = 16)

    plt.subplot(1, 3, 2)
    contour2 = plt.contourf(t_test, x_test, outputs[idx, :, :].T, levels = 20, cmap = 'jet')
    plt.xlabel("t", fontsize = 14)
    plt.ylabel("x", fontsize = 14)
    plt.yticks(fontsize = 12)
    plt.xticks(fontsize = 12)
    cbar2 = plt.colorbar()
    cbar2.ax.tick_params(labelsize=12)
    plt.title("Actual", fontsize=16)

    plt.subplot(1,3,3)
    contour3 = plt.contourf(t_test, x_test, np.abs(u_pred[idx, :, :].T - outputs[idx, :, :].T), cmap = 'Wistia')
    plt.xlabel("t", fontsize = 14)
    plt.ylabel("x", fontsize = 14)
    plt.yticks(fontsize = 12)
    plt.xticks(fontsize = 12)
    cbar3 = plt.colorbar()
    cbar3.ax.tick_params(labelsize=12)
    plt.title("Error", fontsize = 16)

    plt.tight_layout()
    
    # plt.savefig(filepath + f"/Contour_plots_{idx}.jpeg", dpi = 800)
    plt.show()

#Plotting the relative L2 error obtained at every timestep to show accummulation of autoregressive error
auto_reg_error = []
num_time_steps = Nt

for i in range(num_time_steps):
    l2_error = jnp.linalg.norm(u_pred[:,i,:] - outputs[:,i,:])/jnp.linalg.norm(outputs[:,i,:])
    auto_reg_error.append(l2_error)

print("----Extrapolation errors----")
# Compute statistics
t = [180, 210, 270, 300]

for t_idx in t:
    print(f"t: {t_idx}, L2 error: {auto_reg_error[t_idx-1]}")

#Save the auto_reg_error array for comparing with NODE approach
if False:
    np.save(filepath + f"/Auto_reg_error_with_NODE_learnableRK4.npy", auto_reg_error)
    np.save(filepath + f"/u_pred.npy", u_pred)
    np.save(filepath + f"/u_actual.npy", outputs)

print("Program executed successfully!")