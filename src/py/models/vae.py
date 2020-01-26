import functools
import operator

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.py.models.utils.utils import reduce_shape


class Encoder(nn.Module):

    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self._input_dim = input_dim
        self._reduced_input_dim = reduce_shape(input_dim)
        self._latent_dim = latent_dim

        dim_of_nested_layer1 = latent_dim
        self._l1 = nn.Linear(self._reduced_input_dim, dim_of_nested_layer1)
        self._rl1 = nn.ReLU()
        self._drop1 = nn.Dropout(p=0.3)
        self._bn1 = nn.BatchNorm1d(dim_of_nested_layer1)

        dim_of_nested_layer2 = latent_dim
        self._l4 = nn.Linear(dim_of_nested_layer1, dim_of_nested_layer2)
        self._rl2 = nn.ReLU()
        self._drop2 = nn.Dropout(p=0.3)
        self._bn2 = nn.BatchNorm1d(dim_of_nested_layer2)

        self._l2_mean = nn.Linear(dim_of_nested_layer2, latent_dim)
        self._l3_logvar = nn.Linear(dim_of_nested_layer2, latent_dim)

    def forward(self, x):
        x = x.view(-1, reduce_shape(self._input_dim))

        x = self._l1(x)
        x = self._rl1(x)
        x = self._drop1(x)
        x = self._bn1(x)

        x = self._l4(x)
        x = self._rl2(x)
        x = self._drop2(x)
        x = self._bn2(x)

        mean = self._l2_mean(x)
        logvar = self._l3_logvar(x)
        return mean, logvar


class Decoder(nn.Module):

    def __init__(self, latent_dim, output_dim):
        super().__init__()
        self._latent_dim = latent_dim
        self._output_dim = output_dim
        self._reduced_output_dim = reduce_shape(output_dim)

        dim_of_nested_layer1 = latent_dim
        self._l1 = nn.Linear(latent_dim, dim_of_nested_layer1)
        self._rl1 = nn.ReLU()
        self._drop1 = nn.Dropout(p=0.3)
        self._bn1 = nn.BatchNorm1d(dim_of_nested_layer1)

        self._l2 = nn.Linear(dim_of_nested_layer1, self._reduced_output_dim)

    def forward(self, x):
        x = self._l1(x)
        x = self._rl1(x)
        x = self._drop1(x)
        x = self._bn1(x)

        x = self._l2(x)
        x = x.view(-1, *self._output_dim)
        return torch.sigmoid(x)


class VAE:

    # MOCK: default variable in the case has bad smell
    def __init__(self, tensor_board_writer, lr=0.001, input_dim=(1, 28, 28), latent_dim=500):
        self._learn_step_counter = 0
        self._tensor_board_writer = tensor_board_writer
        self._encoder = Encoder(input_dim, latent_dim)
        self._decoder = Decoder(latent_dim, input_dim)
        self._optim = torch.optim.Adam([*self._encoder.parameters(), *self._decoder.parameters()], lr=lr)
        # self._out_loss = F.mse_loss
        self._out_loss = F.binary_cross_entropy
        self._distribution_loss = lambda mean, logvar: -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())

    def _sampling(self, mean, logvar):
        assert mean.shape == logvar.shape
        samples = torch.as_tensor(np.random.randn(*mean.shape), dtype=torch.float)
        return mean + samples * logvar.exp()

    def learn(self, X):
        self._learn_step_counter += 1

        mean, logvar = self._encoder(X)
        x1 = self._sampling(mean, logvar)
        out_X = self._decoder(x1)

        loss = self._out_loss(out_X, X) + self._distribution_loss(mean, logvar)

        self._tensor_board_writer.add_scalar("loss", loss.item(), self._learn_step_counter)

        self._optim.zero_grad()
        loss.backward()
        self._optim.step()

    def get_models(self):
        return (self._encoder, self._decoder)

    def auto_encoder(self, X):
        with torch.no_grad():
            mean, logvar = self._encoder(X)
            X = self._sampling(mean, logvar)
            X = self._decoder(X)
            return X

    def encoder(self, X):
        with torch.no_grad():
            return self._auto_encoder._encoder(X)

    def decoder(self, z):
        with torch.no_grad():
            return self._auto_encoder._decoder(z)
