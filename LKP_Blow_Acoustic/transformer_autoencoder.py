"""Transformer autoencoder for tabular 43-D acoustic feature vectors."""

import torch
import torch.nn as nn


class TransformerAutoencoder(nn.Module):
    """Treat each scalar acoustic feature as a token and encode it with attention."""

    def __init__(self, input_dim=43, latent_dim=16, embedding_dim=32,
                 num_heads=4, num_layers=2):
        super().__init__()
        self.feature_embedding = nn.Linear(1, embedding_dim)
        self.position_embedding = nn.Parameter(
            torch.randn(1, input_dim, embedding_dim) * 0.02
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, nhead=num_heads, dim_feedforward=64,
            dropout=0.0, batch_first=True, activation="gelu"
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.to_latent = nn.Linear(embedding_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.GELU(), nn.Linear(64, input_dim)
        )

    def encode(self, x):
        tokens = self.feature_embedding(x.unsqueeze(-1))
        tokens = tokens + self.position_embedding
        encoded = self.transformer_encoder(tokens)
        return self.to_latent(encoded.mean(dim=1))

    def decode(self, latent):
        return self.decoder(latent)

    def forward(self, x):
        return self.decode(self.encode(x))
