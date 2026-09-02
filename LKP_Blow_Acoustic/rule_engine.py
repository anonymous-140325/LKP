"""
rule_engine.py

Combines per-concept scores and overall knowledge similarity into
a single confidence value used for the accept/reject decision.

Confidence formula:
    reasoning_score = mean(BlowStrength, RhythmStability,
                           SpectralConsistency, similarity)
    confidence      = 0.7 * similarity + 0.3 * reasoning_score
"""


def evaluate_rules(concept_scores, similarity, verbose=True):
    """
    Compute a final confidence score from concept scores and similarity.

    Parameters
    ----------
    concept_scores : dict
        Per-concept similarity scores from knowledge_matcher.py.

    similarity : float
        Overall knowledge similarity from knowledge_matcher.py.

    Returns
    -------
    confidence : float
        Weighted confidence score (0–1).
    """

    reasoning_score = (
        concept_scores["BlowStrength"]
        + concept_scores["RhythmStability"]
        + concept_scores["SpectralConsistency"]
        + similarity
    ) / 4.0

    confidence = 0.7 * similarity + 0.3 * reasoning_score

    if verbose:
        print(f"  Reasoning Score : {reasoning_score:.4f}")
        print(f"  Confidence      : {confidence:.4f}")

    return confidence
