"""
autoencoder.py

Autoencoder architecture for compressing 43-D acoustic feature
vectors into a 16-D latent representation.

Encoder path:  43 → 64 → 32 → 16
Decoder path:  16 → 32 → 64 → 43

The encoder learns which combinations of acoustic features best
describe user behaviour. The resulting 16-D latent vector is then
passed to concept_mapper.py to produce named semantic concepts.

Architecture rationale:
  43  = input feature dimension (20 MFCC means + 20 MFCC stds
                                 + ZCR + RMS + Spectral Centroid)
  64  = slight expansion to learn richer feature combinations
  32  = gradual compression while retaining important structure
  16  = bottleneck latent space (R^43 → R^16)
"""

import torch.nn as nn


class Autoencoder(nn.Module):

    def __init__(self, input_dim=43, latent_dim=16):

        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )

    def forward(self, x):
        return self.decode(self.encode(x))

    def encode(self, x):
        """Return the deterministic latent representation used for matching."""
        return self.encoder(x)

    def decode(self, latent):
        return self.decoder(latent)
