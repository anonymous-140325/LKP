"""
=========================================================
LKP MAIN PIPELINE (Sample-to-Sample Evaluation)
=========================================================

Evaluation protocol:
  - One semantic profile is generated for EACH blow-acoustic sample.
  - No statistical user profile (mean/std) is constructed.
  - Ordered genuine comparisons:
        every sample → every other sample of the same user
        (excluding itself).
  - Ordered impostor comparisons:
        every sample → every sample of every other user.
  - Supports GLOBAL and PER_USER thresholds.

Per-user threshold (KNN 1-class classifier):
  - Genuine pairwise confidence scores are collected per user.
  - KNN is fitted on those scores to characterise the genuine cluster.
  - The maximum allowed KNN distance (max_distance) is stored per user.
  - At authentication time, a probe's confidence score is queried
    against the enrolled KNN model. It is accepted if its mean
    distance to its K nearest genuine scores is within max_distance.
  - This is a proper 1-class classifier — acceptance is determined
    by proximity to the genuine cluster, NOT by a scalar threshold.

Why min(inlier_scores) is wrong:
  - min(inlier_scores) = floor of the genuine distribution.
  - Any impostor scoring above that floor is accepted.
  - Result: FRR=0, FAR≈1, accuracy collapses.

Why KNN distance comparison is correct:
  - A probe is only accepted when it sits INSIDE the genuine cluster.
  - Impostors with high confidence scores are still rejected if they
    are far from where the genuine cluster is concentrated.
=========================================================
"""

import torch
import numpy as np
from collections import defaultdict
from sklearn.neighbors import NearestNeighbors
from time import perf_counter

from dataset_loader     import load_dataset
from feature_extraction import extract_features
from model_factory      import create_model, default_model_path
from concept_mapper     import map_concepts
from knowledge_profile  import build_user_profile
from knowledge_matcher  import match_profile
from rule_engine        import evaluate_rules
from metrics            import (
    compute_accuracy,
    compute_far,
    compute_frr,
    compute_eer
)


# -------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------

DATASET_PATH = "dataset"

MODEL_TYPE = "AUTOENCODER"
# Options:
#   "AUTOENCODER"             — feed-forward deterministic autoencoder
#   "VAE"                     — variational autoencoder (mean latent vector)
#   "TRANSFORMER_AUTOENCODER" — self-attention encoder over acoustic features

MODEL_PATH = default_model_path(MODEL_TYPE)

INPUT_DIM  = 43
LATENT_DIM = 16

GLOBAL_THRESHOLD = 0.98

THRESHOLD_MODE = "GLOBAL"
# Options:
#   "GLOBAL"   — single scalar threshold applied to all comparisons
#   "PER_USER" — a threshold is defined for each user 

# KNN configuration (used only when THRESHOLD_MODE = "PER_USER")
KNN_K = 3
# Number of neighbours used to characterise the genuine score cluster.
# Recommended:
#   2–3 samples per user  → KNN_K = 2
#   4–6 samples per user  → KNN_K = 3
#   7+  samples per user  → KNN_K = 5

KNN_BOUNDARY_STD_MULTIPLIER = 0.05
# Controls how tightly the genuine cluster boundary is drawn.
#   max_distance = mean(knn_distances) + multiplier * std(knn_distances)
#   Lower  → tighter boundary → rejects more → lower FAR, higher FRR
#   Higher → looser  boundary → accepts more → higher FAR, lower FRR


# -------------------------------------------------------
# LOAD SELECTED MODEL
# -------------------------------------------------------

def load_model():

    model = create_model(
        MODEL_TYPE,
        input_dim=INPUT_DIM,
        latent_dim=LATENT_DIM
    )

    model.load_state_dict(
        torch.load(MODEL_PATH, map_location="cpu")
    )

    model.eval()

    print(f"\n[INFO] {MODEL_TYPE} loaded from {MODEL_PATH}.")

    return model


# -------------------------------------------------------
# LATENT GENERATION
# -------------------------------------------------------

def get_latent_vector(model, features):

    x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        latent = model.encode(x)

    return latent.squeeze().numpy()


# -------------------------------------------------------
# BUILD SAMPLE PROFILES
# -------------------------------------------------------

def build_sample_profiles(paths, labels, concepts_list):

    profiles = []

    for path, label, concepts in zip(paths, labels, concepts_list):

        profile = build_user_profile(
            user_id=label,
            concept_dict=concepts,
            sample_path=path
        )

        profiles.append(profile)

    return profiles


# -------------------------------------------------------
# BUILD PER-USER KNN 1-CLASS MODELS
# -------------------------------------------------------

def build_user_knn_models(sample_profiles):
    """
    For each user, fit a KNN model on their genuine pairwise confidence
    scores and compute the maximum allowed KNN distance for acceptance.

    Returns
    -------
    user_models : dict
        Keys: user_id
        Values: dict with keys
          "knn"          — fitted NearestNeighbors model (or None)
          "max_distance" — acceptance boundary in distance space
          "fallback"     — True if insufficient scores; use GLOBAL_THRESHOLD

    Authentication (in evaluate_pairwise):
      1. Compute probe confidence score.
      2. Query the user's KNN model for the probe's mean distance
         to its K nearest genuine scores.
      3. Accept if probe_distance <= max_distance.

    This is a 1-class classifier — it does not use a scalar threshold.
    Acceptance is determined by proximity to the genuine cluster,
    so impostors with high confidence scores are still rejected if
    they fall outside the cluster's learned boundary.
    """

    user_models = {}
    grouped     = defaultdict(list)

    for profile in sample_profiles:
        grouped[profile["user_id"]].append(profile)

    all_users = sorted(grouped.keys())

    print()
    print("=" * 60)
    print("BUILDING PER-USER KNN 1-CLASS MODELS")
    print("=" * 60)

    for user in all_users:

        genuine_scores = []
        user_samples   = grouped[user]

        # --- collect genuine pairwise confidence scores ---
        for i in range(len(user_samples)):
            for j in range(len(user_samples)):

                if i == j:
                    continue

                similarity, concept_scores = match_profile(
                    user_samples[i]["concepts"],
                    user_samples[j]["concepts"]
                )

                confidence = evaluate_rules(concept_scores, similarity)
                genuine_scores.append(confidence)

        # --- fallback: insufficient scores to fit KNN ---
        if len(genuine_scores) < KNN_K + 1:
            user_models[user] = {
                "knn":          None,
                "max_distance": None,
                "fallback":     True
            }
            print(
                f"{user}: insufficient genuine scores "
                f"(need >= {KNN_K + 1}, got {len(genuine_scores)}) "
                f"→ fallback to GLOBAL_THRESHOLD={GLOBAL_THRESHOLD:.4f}"
            )
            continue

        scores_array = np.array(genuine_scores).reshape(-1, 1)

        # --- fit KNN on genuine scores ---
        knn = NearestNeighbors(n_neighbors=KNN_K)
        knn.fit(scores_array)

        # --- compute intra-cluster distances ---
        distances, _  = knn.kneighbors(scores_array)
        avg_knn_dist  = distances.mean(axis=1)

        # --- cluster boundary in distance space ---
        dist_mean    = np.mean(avg_knn_dist)
        dist_std     = np.std(avg_knn_dist)
        max_distance = dist_mean + KNN_BOUNDARY_STD_MULTIPLIER * dist_std

        user_models[user] = {
            "knn":          knn,
            "max_distance": max_distance,
            "fallback":     False
        }

        print(
            f"{user}: max_distance={max_distance:.6f}  "
            f"genuine_scores={len(genuine_scores)}  "
            f"k={KNN_K}  "
            f"dist_mean={dist_mean:.6f}  "
            f"dist_std={dist_std:.6f}"
        )

    return user_models


# -------------------------------------------------------
# PAIRWISE SAMPLE-TO-SAMPLE EVALUATION
# -------------------------------------------------------

def evaluate_pairwise(sample_profiles, user_models):

    TP = TN = FP = FN = 0
    genuine_count = impostor_count = 0
    comparison_id = 0

    print("\n===================================")
    print("ORDERED SAMPLE-TO-SAMPLE AUTHENTICATION")
    print("===================================\n")

    for probe_idx, probe in enumerate(sample_profiles):

        probe_user     = probe["user_id"]
        probe_path     = probe["sample_path"]
        probe_concepts = probe["concepts"]

        for ref_idx, reference in enumerate(sample_profiles):

            if probe_idx == ref_idx:
                continue

            ref_user     = reference["user_id"]
            ref_path     = reference["sample_path"]
            ref_concepts = reference["concepts"]

            comparison_id += 1

            similarity, concept_scores = match_profile(
                probe_concepts, ref_concepts
            )
            confidence = evaluate_rules(concept_scores, similarity)

            # --- accept / reject decision ---
            if THRESHOLD_MODE == "GLOBAL":

                accepted = confidence >= GLOBAL_THRESHOLD
                decision_detail = f"Threshold       : {GLOBAL_THRESHOLD:.4f}"

            elif THRESHOLD_MODE == "PER_USER":

                model = user_models.get(ref_user, {"fallback": True})

                if model["fallback"]:
                    # Not enough genuine scores at enrolment → scalar fallback
                    accepted = confidence >= GLOBAL_THRESHOLD
                    decision_detail = (
                        f"Threshold       : {GLOBAL_THRESHOLD:.4f} (fallback)"
                    )

                else:
                    # 1-class KNN distance comparison
                    probe_arr    = np.array([[confidence]])
                    distances, _ = model["knn"].kneighbors(probe_arr)
                    probe_dist   = float(distances.mean())
                    max_dist     = model["max_distance"]
                    accepted     = probe_dist <= max_dist
                    decision_detail = (
                        f"Probe Distance  : {probe_dist:.6f}\n"
                        f"Max Distance    : {max_dist:.6f}"
                    )

            else:
                raise ValueError(f"Unknown THRESHOLD_MODE: {THRESHOLD_MODE}")

            genuine = (probe_user == ref_user)

            print("=" * 70)
            print(f"Comparison #{comparison_id}")
            print(f"Probe User      : {probe_user}")
            print(f"Probe Sample    : {probe_path}")
            print(f"Reference User  : {ref_user}")
            print(f"Reference Sample: {ref_path}")
            print(decision_detail)
            print(f"Similarity      : {similarity:.4f}")
            print(f"Confidence      : {confidence:.4f}")

            if genuine:

                genuine_count += 1

                if accepted:
                    TP += 1
                    print("Decision : ACCEPT")
                    print("Result   : TRUE POSITIVE")
                else:
                    FN += 1
                    print("Decision : REJECT")
                    print("Result   : FALSE NEGATIVE")

            else:

                impostor_count += 1

                if accepted:
                    FP += 1
                    print("Decision : ACCEPT")
                    print("Result   : FALSE POSITIVE")
                else:
                    TN += 1
                    print("Decision : REJECT")
                    print("Result   : TRUE NEGATIVE")

    return TP, TN, FP, FN, genuine_count, impostor_count


# -------------------------------------------------------
# SCORE DISTRIBUTION DIAGNOSTIC
# -------------------------------------------------------

def diagnose_score_distributions(sample_profiles):
    """
    Prints genuine vs impostor confidence score statistics.
    If the distributions overlap heavily, no threshold will work.
    This must be called before evaluate_pairwise to understand
    whether the confidence scores are discriminative at all.
    """

    grouped = defaultdict(list)
    for profile in sample_profiles:
        grouped[profile["user_id"]].append(profile)

    all_users       = sorted(grouped.keys())
    genuine_scores  = []
    impostor_scores = []

    for user_a in all_users:
        for sample_a in grouped[user_a]:
            for user_b in all_users:
                for sample_b in grouped[user_b]:

                    if sample_a is sample_b:
                        continue

                    similarity, concept_scores = match_profile(
                        sample_a["concepts"],
                        sample_b["concepts"],
                        verbose=False
                    )
                    confidence = evaluate_rules(
                        concept_scores, similarity, verbose=False
                    )

                    if user_a == user_b:
                        genuine_scores.append(confidence)
                    else:
                        impostor_scores.append(confidence)

    g  = np.array(genuine_scores)
    im = np.array(impostor_scores)

    print("\n" + "=" * 60)
    print("SCORE DISTRIBUTION DIAGNOSTIC")
    print("=" * 60)

    print(f"\nGenuine  scores  (n={len(g)})")
    print(f"  min    : {g.min():.4f}")
    print(f"  max    : {g.max():.4f}")
    print(f"  mean   : {g.mean():.4f}")
    print(f"  std    : {g.std():.4f}")
    print(f"  median : {np.median(g):.4f}")

    print(f"\nImpostor scores  (n={len(im)})")
    print(f"  min    : {im.min():.4f}")
    print(f"  max    : {im.max():.4f}")
    print(f"  mean   : {im.mean():.4f}")
    print(f"  std    : {im.std():.4f}")
    print(f"  median : {np.median(im):.4f}")

    overlap = np.sum(im >= g.min()) / len(im) * 100
    print(f"\nImpostor scores above genuine minimum : {overlap:.1f}%")
    print("(If this is high, scores are not discriminative)")

    sep = (g.mean() - im.mean()) / (
        np.sqrt((g.std() ** 2 + im.std() ** 2) / 2) + 1e-8
    )
    print(f"\nFisher separation (d') : {sep:.4f}")
    print("  d' < 1.0 → poor separation → threshold cannot fix this")
    print("  d' > 2.0 → good separation → threshold tuning will help")


# -------------------------------------------------------
# MAIN
# -------------------------------------------------------

def main():

    print("======================== //---BEGIN---// ========================")
    print("\n===================================")
    print("LKP SAMPLE-TO-SAMPLE AUTHENTICATION")
    print("===================================\n")

    # --- load dataset ---
    paths, labels = load_dataset(DATASET_PATH)
    print(f"[INFO] Samples = {len(paths)}")
    print(f"[INFO] Users   = {len(set(labels))}")

    # --- feature extraction ---
    print("\n===================================")
    print("FEATURE EXTRACTION")
    print("===================================\n")

    features_list = []
    feature_times = []

    for path in paths:
        sample_start = perf_counter()
        features_list.append(extract_features(path))
        feature_times.append(perf_counter() - sample_start)

    print(f"[INFO] Feature vectors   = {len(features_list)}")
    print(f"[INFO] Feature dimension = {features_list[0].shape}")

    # --- load selected representation model ---
    model = load_model()

    # --- latent generation ---
    print("\n===================================")
    print("LATENT GENERATION")
    print("===================================\n")

    latent_vectors = []
    inference_times = []

    for features, feature_time in zip(features_list, feature_times):
        sample_start = perf_counter()
        latent_vectors.append(get_latent_vector(model, features))
        inference_times.append(feature_time + perf_counter() - sample_start)

    print(f"[INFO] Latent vectors   = {len(latent_vectors)}")
    print(f"[INFO] Latent dimension = {latent_vectors[0].shape}")

    # --- concept mapping ---
    print("\n===================================")
    print("CONCEPT MAPPING")
    print("===================================\n")

    # compute per-dimension mean and std across all latent vectors
    # so that z-score normalisation can centre each dimension before
    # sigmoid — prevents saturation when all latent values are large
    latent_matrix = np.array(latent_vectors)          # (N, 16)
    latent_mean   = latent_matrix.mean(axis=0)        # (16,)
    latent_std    = latent_matrix.std(axis=0) + 1e-8  # (16,)

    print(f"[INFO] Latent mean (first 5 dims): "
          f"{latent_mean[:5].round(4)}")
    print(f"[INFO] Latent std  (first 5 dims): "
          f"{latent_std[:5].round(4)}")

    concepts_list = []
    for index, latent_vector in enumerate(latent_vectors):
        sample_start = perf_counter()
        concepts_list.append(
            map_concepts(latent_vector, latent_mean, latent_std)
        )
        inference_times[index] += perf_counter() - sample_start

    print(f"[INFO] Concept objects  = {len(concepts_list)}")
    print(f"[INFO] Example concepts = {concepts_list[0]}")
    print(
        f"[INFO] Average inference time per biometric sample "
        f"(feature extraction + encoding + concept mapping): "
        f"{np.mean(inference_times):.6f} seconds"
    )

    # --- build one profile per sample ---
    print("\n===================================")
    print("SAMPLE PROFILE CREATION")
    print("===================================\n")

    sample_profiles = build_sample_profiles(paths, labels, concepts_list)

    print(f"[INFO] Sample profiles = {len(sample_profiles)}")

    # --- build per-user KNN 1-class models ---
    if THRESHOLD_MODE == "PER_USER":
        user_models = build_user_knn_models(sample_profiles)
    else:
        user_models = {}

    # --- pairwise evaluation ---
    TP, TN, FP, FN, genuine_count, impostor_count = evaluate_pairwise(
        sample_profiles,
        user_models
    )

    # --- metrics ---
    accuracy = compute_accuracy(TP, TN, FP, FN)
    far      = compute_far(TP, TN, FP, FN)
    frr      = compute_frr(TP, TN, FP, FN)
    eer      = compute_eer(far, frr)

    print(f"\n===================================")
    print(f"FINAL RESULTS ({THRESHOLD_MODE}, {MODEL_TYPE})")
    print("===================================")
    print(f"Genuine Counts  = {genuine_count}")
    print(f"Impostor Counts = {impostor_count}")
    
    print("\nCONFUSION MATRIX")
    print(f"  TP = {TP}")
    print(f"  TN = {TN}")
    print(f"  FP = {FP}")
    print(f"  FN = {FN}")
    print("\nMETRICS")
    print(f"  Accuracy = {accuracy:.4f}")
    print(f"  FAR      = {far:.4f}")
    print(f"  FRR      = {frr:.4f}")
    print(f"  EER      = {eer:.4f}")

    # --- score distribution diagnostic ---
    #diagnose_score_distributions(sample_profiles)
    
    print("\nExecution Time")
    print(
        f"Average inference time per biometric sample: "
        #f"(feature extraction + encoding + concept mapping): "
        f"{np.mean(inference_times):.6f} seconds"
    )


    print("======================== //---END---// ========================")


# -------------------------------------------------------
# ENTRY POINT
# -------------------------------------------------------

if __name__ == "__main__":
    main()
