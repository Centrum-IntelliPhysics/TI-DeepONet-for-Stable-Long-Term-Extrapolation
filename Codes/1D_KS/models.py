#This notebook implements a DeepONet from scratch using FLAX

#importing all necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import h5py

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

class branch_net(nn.Module):
    nn_layers: Sequence[int]
    activation: Callable = nn.activation.silu
    
    @nn.compact
    def __call__(self, x):
        init = nn.initializers.glorot_normal()
     
        for i, fs in enumerate(self.nn_layers[:-1]):
            x = nn.Dense(fs, kernel_init=init)(x)
            x = self.activation(x)
        x = nn.Dense(self.nn_layers[-1], kernel_init=init)(x)
        return x

class trunk_net(nn.Module):
    nn_layers: Sequence[int]
    activation: Callable = nn.activation.silu
    
    @nn.compact
    def __call__(self, x):
        init = nn.initializers.glorot_normal()
     
        for i, fs in enumerate(self.nn_layers):
            x = nn.Dense(fs, kernel_init=init)(x)
            x = self.activation(x)
        return x


class SinBranchNet(nn.Module):
    nn_layers: Sequence[int]  # Example: [128, 128, 128, 100]
    w0: float = 30.0          # Frequency scale for first layer

    @nn.compact
    def __call__(self, x):
        in_features = x.shape[-1]
        
        # First layer: scaled init for SIREN
        init_range = jnp.sqrt(6 / in_features) / self.w0
        W_init = nn.initializers.uniform(scale=init_range)
        x = nn.Dense(self.nn_layers[0], kernel_init=W_init)(x)
        x = jnp.sin(self.w0 * x)

        # Hidden layers with sin activation
        for width in self.nn_layers[1:-1]:
            x = nn.Dense(width, kernel_init=nn.initializers.kaiming_uniform())(x)
            x = jnp.sin(x)

        # Output layer
        x = nn.Dense(self.nn_layers[-1], kernel_init=nn.initializers.kaiming_uniform())(x)
        return x


class SinTrunkNet(nn.Module):
    nn_layers: Sequence[int]  # e.g., [64, 64, 64, 100]
    w0: float = 30.0          # frequency scale

    @nn.compact
    def __call__(self, x):
        in_features = x.shape[-1]

        # First layer: scaled init
        init_range = jnp.sqrt(6 / in_features) / self.w0
        dense_init = nn.initializers.uniform(scale=init_range)
        x = nn.Dense(self.nn_layers[0], kernel_init=dense_init)(x)
        x = jnp.sin(self.w0 * x)

        # Hidden layers
        for i, width in enumerate(self.nn_layers[1:]):
            x = nn.Dense(width, kernel_init=nn.initializers.kaiming_uniform())(x)
            x = jnp.sin(x)
        return x
