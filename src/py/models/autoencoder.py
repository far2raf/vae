import functools
import operator

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.py.models.utils.utils import reduce_shape


class NestedAutoencoder(nn.Module):

    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self._input_dim = input_dim
        self._latent_dim = latent_dim
        self._encoder = Encoder(input_dim, latent_dim)
        self._decoder = Decoder(latent_dim, input_dim)

    def forward(self, x):
        x = self._encoder(x)
        x = self._decoder(x)
        return x



class Encoder(nn.Module):

    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self._input_dim = input_dim
        self._latent_dim = latent_dim
        self._l1 = nn.Linear(reduce_shape(input_dim), latent_dim)

    def forward(self, x):
        x = x.view(-1, reduce_shape(self._input_dim))
        x = self._l1(x)
        # return x
        return torch.sigmoid(x)


class Decoder(nn.Module):

    def __init__(self, latent_dim, output_dim):
        super().__init__()
        self._latent_dim = latent_dim
        self._output_dim = output_dim
        self._l1 = nn.Linear(latent_dim, functools.reduce(operator.mul, output_dim))

    def forward(self, x):
        x = self._l1(x)
        x = x.view(-1, *self._output_dim)
        # return x
        return torch.sigmoid(x)


class Autoencoder:

    # MOCK: default variable in the case has bad smell
    def __init__(self, tensor_board_writer, lr=0.001, input_dim=(1, 28, 28), latent_dim=500):
        self._learn_step_counter = 0
        self._tensor_board_writer = tensor_board_writer
        self._auto_encoder = NestedAutoencoder(input_dim, latent_dim)
        self._optim = torch.optim.Adam(self._auto_encoder.parameters(), lr=lr)
        # self._loss = F.mse_loss
        self._loss = F.binary_cross_entropy

    def learn(self, X):
        self._learn_step_counter += 1
        out_X = self._auto_encoder(X)
        loss = self._loss(out_X, X)

        self._tensor_board_writer.add_scalar("loss", loss.item(), self._learn_step_counter)

        self._optim.zero_grad()
        loss.backward()
        self._optim.step()

    def get_models(self):
        return (self._auto_encoder, self._auto_encoder._encoder, self._auto_encoder._decoder)

    def auto_encoder(self, X):
        with torch.no_grad():
            return self._auto_encoder(X)

    def encoder(self, X):
        with torch.no_grad():
            return self._auto_encoder._encoder(X)

    def decoder(self, z):
        with torch.no_grad():
            return self._auto_encoder._decoder(z)

