"""Variational autoencoder for 43-D acoustic feature vectors."""

import torch
import torch.nn as nn


class VariationalAutoencoder(nn.Module):
    """A VAE whose mean latent vector is used for deterministic matching."""

    def __init__(self, input_dim=43, latent_dim=16):
        super().__init__()
        self.encoder_backbone = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
        )
        self.mu_layer = nn.Linear(32, latent_dim)
        self.logvar_layer = nn.Linear(32, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32), nn.ReLU(),
            nn.Linear(32, 64), nn.ReLU(),
            nn.Linear(64, input_dim),
        )

    def encode_distribution(self, x):
        hidden = self.encoder_backbone(x)
        return self.mu_layer(hidden), self.logvar_layer(hidden)

    def encode(self, x):
        """Return the mean latent vector; deterministic for authentication."""
        mu, _ = self.encode_distribution(x)
        return mu

    def reparameterize(self, mu, logvar):
        if not self.training:
            return mu
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, latent):
        return self.decoder(latent)

    def forward(self, x):
        mu, logvar = self.encode_distribution(x)
        latent = self.reparameterize(mu, logvar)
        return self.decode(latent), mu, logvar
