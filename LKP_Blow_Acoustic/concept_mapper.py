"""
concept_mapper.py

Maps a 16-D latent vector produced by the autoencoder encoder
to a dictionary of named semantic concepts.

Normalisation strategy — z-score then sigmoid:
  Z-score centres each latent dimension across the dataset (mean=0,
  std=1), breaking saturation caused by large raw values. Sigmoid
  then maps the spread into (0, 1) for the Gaussian matcher.

    z             = (latent[i] - mean[i]) / (std[i] + 1e-8)
    concept_value = sigmoid(z)

  latent_mean and latent_std are computed from all latent vectors
  in main.py and passed here.

Concept layout:
  Dims 0–4  : named semantic concepts
                BlowStrength, BlowDuration, RhythmStability,
                SpectralConsistency, NoiseLevel
  Dims 5–15 : unnamed latent dimensions (Latent_05 … Latent_15)

Using all 16 dimensions ensures no learned representation is
discarded. knowledge_matcher averages similarity across all shared
concept keys automatically, so the extra dimensions are included
without any changes to the matching or rule engine logic.
"""

import math


def _sigmoid(x):
    return 1.0 / (1.0 + math.exp(-float(x)))


# Named concepts for latent dimensions 0–4
_NAMED_CONCEPTS = {
    0: "BlowStrength",
    1: "BlowDuration",
    2: "RhythmStability",
    3: "SpectralConsistency",
    4: "NoiseLevel"
}


def map_concepts(latent, latent_mean=None, latent_std=None):
    """
    Parameters
    ----------
    latent : array-like (16,)
        Raw latent vector from the autoencoder encoder.

    latent_mean : array-like (16,), optional
        Per-dimension mean computed across all dataset samples.

    latent_std : array-like (16,), optional
        Per-dimension std computed across all dataset samples.

    Returns
    -------
    dict : concept name → normalised score in (0, 1)
    """

    def normalise(i):
        raw = float(latent[i])
        if latent_mean is not None and latent_std is not None:
            z = (raw - float(latent_mean[i])) / (
                float(latent_std[i]) + 1e-8
            )
        else:
            z = raw   # fallback — may saturate
        return _sigmoid(z)

    concepts = {}

    for i in range(len(latent)):
        name = _NAMED_CONCEPTS.get(i, f"Latent_{i:02d}")
        concepts[name] = normalise(i)

    print("\n=== CONCEPTS ===")
    for k, v in concepts.items():
        print(f"  {k} = {round(v, 4)}")

    return concepts
