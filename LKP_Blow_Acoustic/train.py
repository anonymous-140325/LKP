"""
train_autoencoder.py

Trains the autoencoder with a combined reconstruction + triplet loss.

Loss function:
  total_loss = MSELoss(recon, x)  +  lambda_t * TripletLoss(latent, labels)

MSELoss (reconstruction):
  Teaches the encoder to compress 43-D features into a 16-D latent
  vector that retains enough information for reconstruction.

TripletLoss (discriminative)  [ACTIVE]:
  For each anchor sample, finds its hardest positive (same user,
  furthest away) and hardest negative (different user, closest).
  Enforces a margin so that every genuine pair is at least `margin`
  closer in latent space than any impostor pair.

  Directly pushes d' above 2.0 by simultaneously:
    → pulling same-user latent vectors together
    → pushing different-user latent vectors apart

  Centre loss ceiling: d' plateaus at ~1.30 regardless of LAMBDA_C
  because it only pulls same-user samples together without explicitly
  pushing different users apart.

CentreLoss (discriminative)  [COMMENTED OUT — preserved for reference]:
  Maintains one learnable centre vector per user in latent space.
  Results: LAMBDA_C=1.0 → d'=1.30, Accuracy=91.4%, EER=0.043 (best)
  Ceiling reached — triplet loss used instead.
  To reactivate: uncomment CentreLoss class and the marked sections
  in train_autoencoder(), comment out the TripletLoss sections.

References:
  Schroff et al. (2015) FaceNet — triplet loss for face verification.
  Wen et al. (2016) Centre loss for face recognition.
"""

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from time import perf_counter

from model_factory import create_model, default_model_path
from variational_autoencoder import VariationalAutoencoder


# -------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------

EPOCHS   = 200    # epochs needed for triplet loss to converge

# --- Triplet loss ---
LAMBDA_T = 2.0    # weight of triplet loss vs reconstruction loss
                  # 1.0  → balanced; increase if d' still low
MARGIN   = 1.0 #0.5    # minimum required gap between d(anchor, positive)
                  # and d(anchor, negative) in latent space
                  # larger → stricter separation, harder to satisfy

# --- Optimiser ---
LR       = 0.001  # Adam learning rate

MODEL_TYPE = "AUTOENCODER"
# Use the same model type in main.py when evaluating this checkpoint.
# Options:
#   "AUTOENCODER"             — feed-forward deterministic autoencoder
#   "VAE"                     — variational autoencoder (mean latent vector)
#   "TRANSFORMER_AUTOENCODER" — self-attention encoder over acoustic features
VAE_KL_WEIGHT = 0.001

# --- Centre loss (inactive) ---
# LAMBDA_C   = 1.0   # best result: d'=1.30, Accuracy=91.4%, EER=0.043
# LR_CENTRES = 0.5   # SGD learning rate for centre parameters


# -------------------------------------------------------
# TRIPLET LOSS
# -------------------------------------------------------

class TripletLoss(nn.Module):
    """
    Hard-negative triplet loss.

    For each anchor i:
      - hardest positive: same-user sample with maximum distance
      - hardest negative: different-user sample with minimum distance

    Loss = mean over all anchors of:
      max(0,  d(anchor, hardest_positive)
            - d(anchor, hardest_negative)
            + margin)

    Parameters
    ----------
    margin : float
        Minimum required separation between positive and negative
        distances. Typical values: 0.2 – 1.0.
    """

    def __init__(self, margin=MARGIN):
        super().__init__()
        self.margin = margin

    def forward(self, latent, labels):
        """
        Parameters
        ----------
        latent : (N, latent_dim) tensor
        labels : (N,) long tensor
        """
        n    = latent.shape[0]
        loss = torch.zeros(1, requires_grad=False)
        loss = loss.clone()
        count = 0

        for i in range(n):

            anchor = latent[i]
            label  = labels[i]

            # squared distances from anchor to all other samples
            dists = ((latent - anchor) ** 2).sum(dim=1)

            # hardest positive: same user, furthest away
            pos_mask    = (labels == label)
            pos_mask[i] = False
            if pos_mask.sum() == 0:
                continue
            d_pos = dists[pos_mask].max()

            # hardest negative: different user, closest
            neg_mask = (labels != label)
            if neg_mask.sum() == 0:
                continue
            d_neg = dists[neg_mask].min()

            triplet = torch.clamp(d_pos - d_neg + self.margin, min=0.0)
            loss    = loss + triplet
            count  += 1

        return loss / max(count, 1)


# -------------------------------------------------------
# CENTRE LOSS  (commented out — preserved for reference)
# -------------------------------------------------------

# class CentreLoss(nn.Module):
#     """
#     Pulls each user's latent vectors toward their own learnable centre.
#     Best result: LAMBDA_C=1.0 → d'=1.30, Accuracy=91.4%, EER=0.043
#     Ceiling: d' plateaus ~1.30 — triplet loss used instead.
#
#     To reactivate:
#       1. Uncomment this class.
#       2. Uncomment the CentreLoss sections in train_autoencoder().
#       3. Comment out the TripletLoss sections in train_autoencoder().
#     """
#
#     def __init__(self, num_users, latent_dim):
#         super().__init__()
#         self.centres = nn.Parameter(
#             torch.randn(num_users, latent_dim)
#         )
#
#     def forward(self, latent, labels):
#         centres_batch = self.centres[labels]
#         return ((latent - centres_batch) ** 2).mean()


# -------------------------------------------------------
# TRAINING
# -------------------------------------------------------

def train_autoencoder(X, user_labels, epochs=EPOCHS, model_type=MODEL_TYPE):
    """
    Parameters
    ----------
    X           : np.ndarray (N, 43)   feature matrix
    user_labels : list of str (N,)     user identity per sample
    epochs      : int

    Returns
    -------
    model : trained selected autoencoder architecture
    """

    # --- encode string labels to integer indices ---
    unique_users  = sorted(set(user_labels))
    user_to_idx   = {u: i for i, u in enumerate(unique_users)}
    label_indices = [user_to_idx[u] for u in user_labels]
    num_users     = len(unique_users)

    X_tensor     = torch.FloatTensor(X)
    label_tensor = torch.LongTensor(label_indices)
    latent_dim   = 16

    # --- model and losses ---
    model         = create_model(
        model_type, input_dim=X.shape[1], latent_dim=latent_dim
    )
    mse_criterion = nn.MSELoss()

    # --- TRIPLET LOSS (active) ---
    triplet_loss  = TripletLoss(margin=MARGIN)
    optimizer     = optim.Adam(model.parameters(), lr=LR)

    # --- CENTRE LOSS (inactive) ---
    # centre_loss       = CentreLoss(num_users=num_users, latent_dim=latent_dim)
    # optimizer_model   = optim.Adam(model.parameters(),      lr=LR)
    # optimizer_centres = optim.SGD(centre_loss.parameters(), lr=LR_CENTRES)

    print(f"\n[INFO] Model     : {model_type}")
    print(f"[INFO] Training  : {epochs} epochs")
    print(f"[INFO] Users     : {num_users}")
    print(f"[INFO] Samples   : {len(X)}")
    print(f"[INFO] Loss      : MSE + TripletLoss")
    print(f"[INFO] lambda_t  : {LAMBDA_T}")
    print(f"[INFO] margin    : {MARGIN}")
    print()

    training_start = perf_counter()

    for epoch in range(epochs):

        optimizer.zero_grad()

        # --- CENTRE LOSS (inactive) ---
        # optimizer_model.zero_grad()
        # optimizer_centres.zero_grad()

        # forward pass
        if isinstance(model, VariationalAutoencoder):
            recon, latent, logvar = model(X_tensor)
            loss_kl = -0.5 * torch.mean(
                1 + logvar - latent.pow(2) - logvar.exp()
            )
        else:
            latent = model.encode(X_tensor)
            recon = model.decode(latent)
            loss_kl = torch.zeros((), device=X_tensor.device)

        # --- TRIPLET LOSS (active) ---
        loss_recon   = mse_criterion(recon, X_tensor)
        loss_triplet = triplet_loss(latent, label_tensor)
        loss_total   = (
            loss_recon + LAMBDA_T * loss_triplet + VAE_KL_WEIGHT * loss_kl
        )

        # --- CENTRE LOSS (inactive) ---
        # loss_recon  = mse_criterion(recon, X_tensor)
        # loss_centre = centre_loss(latent, label_tensor)
        # loss_total  = loss_recon + LAMBDA_C * loss_centre

        # backward pass
        loss_total.backward()
        optimizer.step()

        # --- CENTRE LOSS (inactive) ---
        # optimizer_model.step()
        # optimizer_centres.step()

        # --- logging ---
        triplet_weighted = LAMBDA_T * loss_triplet.item()
        recon_pct        = loss_recon.item()    / (loss_total.item() + 1e-8) * 100
        triplet_pct      = triplet_weighted     / (loss_total.item() + 1e-8) * 100

        print(
            f"Epoch {epoch + 1:>4}: "
            f"total={loss_total.item():.6f}  "
            f"recon={loss_recon.item():.6f} ({recon_pct:.1f}%)  "
            f"triplet={loss_triplet.item():.6f} "
            f"[weighted={triplet_weighted:.6f} ({triplet_pct:.1f}%)]"
            + (f"  kl={loss_kl.item():.6f}"
               if isinstance(model, VariationalAutoencoder) else "")
        )

        # --- CENTRE LOSS logging (inactive) ---
        # centre_weighted = LAMBDA_C * loss_centre.item()
        # recon_pct       = loss_recon.item()   / (loss_total.item() + 1e-8) * 100
        # centre_pct      = centre_weighted     / (loss_total.item() + 1e-8) * 100
        # print(
        #     f"Epoch {epoch + 1:>4}: "
        #     f"total={loss_total.item():.6f}  "
        #     f"recon={loss_recon.item():.6f} ({recon_pct:.1f}%)  "
        #     f"centre={loss_centre.item():.6f} "
        #     f"[weighted={centre_weighted:.6f} ({centre_pct:.1f}%)]"
        # )

    training_time = perf_counter() - training_start

    model_path = default_model_path(model_type)
    torch.save(model.state_dict(), model_path)

    print(f"\n[INFO] Overall training time: {training_time:.4f} seconds")

    return model


# -------------------------------------------------------
# ENTRY POINT
# -------------------------------------------------------

if __name__ == "__main__":

    print("Training autoencoder with triplet loss...")

    from dataset_loader     import load_dataset
    from feature_extraction import extract_features

    audio_paths, labels = load_dataset("dataset")

    features = []
    for path in audio_paths:
        features.append(extract_features(path))

    X = np.array(features)

    print(f"\nFeature matrix shape : {X.shape}")
    print(f"Users                : {len(set(labels))}")

    model = train_autoencoder(X, labels, epochs=EPOCHS, model_type=MODEL_TYPE)

    print("\nTraining complete.")
    print(f"Saved: {default_model_path(MODEL_TYPE)}")
