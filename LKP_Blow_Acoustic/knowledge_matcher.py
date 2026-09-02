"""
knowledge_matcher.py

Performs sample-to-sample semantic matching between two concept
dictionaries produced by concept_mapper.py.

Each concept pair is compared using a Gaussian kernel:

    similarity = exp( -0.5 * ((probe - reference) / sigma)^2 )

where sigma controls the tolerance width. The overall similarity
is the mean of all per-concept similarities.
"""

import math


def match_profile(
    probe_concepts,
    reference_concepts,
    sigma=0.20,
    verbose=True
):
    """
    Compare two semantic concept dictionaries.

    Parameters
    ----------
    probe_concepts : dict
        Concepts extracted from the probe sample.

    reference_concepts : dict
        Concepts extracted from the reference (enrolled) sample.

    sigma : float
        Gaussian bandwidth — controls how sensitive the similarity
        score is to differences between concept values.

    Returns
    -------
    overall_similarity : float
        Mean similarity across all shared concepts (0–1).

    concept_scores : dict
        Per-concept similarity scores.
    """

    concept_scores = {}
    similarities   = []

    common_keys = sorted(
        set(probe_concepts.keys()) & set(reference_concepts.keys())
    )

    for concept in common_keys:

        probe_value     = probe_concepts[concept]
        reference_value = reference_concepts[concept]

        diff = probe_value - reference_value

        similarity = math.exp(-0.5 * (diff / sigma) ** 2)
        similarity = max(0.0, min(1.0, similarity))

        concept_scores[concept] = similarity
        similarities.append(similarity)

    overall_similarity = (
        sum(similarities) / len(similarities)
        if similarities else 0.0
    )

    if verbose:
        print("\nConcept Similarities:")
        for concept in common_keys:
            print(f"  {concept}: {concept_scores[concept]:.4f}")
        print(f"\n  Knowledge Similarity: {overall_similarity:.4f}")

    return overall_similarity, concept_scores


# -------------------------------------------------------
# Standalone test
# -------------------------------------------------------

if __name__ == "__main__":

    sample_a = {
        "BlowStrength":        0.81,
        "BlowDuration":        0.76,
        "RhythmStability":     0.90,
        "SpectralConsistency": 0.84,
        "NoiseLevel":          0.11,
    }

    sample_b = {
        "BlowStrength":        0.79,
        "BlowDuration":        0.74,
        "RhythmStability":     0.88,
        "SpectralConsistency": 0.82,
        "NoiseLevel":          0.13,
    }

    similarity, concept_scores = match_profile(sample_a, sample_b)

    print("\nOverall Similarity =", similarity)
