"""Shared model selection and checkpoint naming for LKP."""

from autoencoder import Autoencoder
from variational_autoencoder import VariationalAutoencoder
from transformer_autoencoder import TransformerAutoencoder


MODEL_TYPES = ("AUTOENCODER", "VAE", "TRANSFORMER_AUTOENCODER")


def create_model(model_type, input_dim=43, latent_dim=16):
    model_type = model_type.upper()
    if model_type == "AUTOENCODER":
        return Autoencoder(input_dim=input_dim, latent_dim=latent_dim)
    if model_type == "VAE":
        return VariationalAutoencoder(input_dim=input_dim, latent_dim=latent_dim)
    if model_type == "TRANSFORMER_AUTOENCODER":
        return TransformerAutoencoder(input_dim=input_dim, latent_dim=latent_dim)
    raise ValueError(f"Unknown MODEL_TYPE: {model_type}. Choose from {MODEL_TYPES}.")


def default_model_path(model_type):
    return {
        "AUTOENCODER": "autoencoder.pth",
        "VAE": "vae.pth",
        "TRANSFORMER_AUTOENCODER": "transformer_autoencoder.pth",
    }[model_type.upper()]
