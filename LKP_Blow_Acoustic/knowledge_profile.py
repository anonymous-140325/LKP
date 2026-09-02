"""
knowledge_profile.py

Builds a semantic knowledge profile for a single blow-acoustic sample.

Each sample is represented independently — no statistical aggregation
across multiple samples is performed. This supports true
sample-to-sample matching in the LKP pipeline.
"""


def build_user_profile(user_id, concept_dict, sample_path=None):
    """
    Build a semantic profile for a single blow-acoustic sample.

    Parameters
    ----------
    user_id : str
        Identity label associated with the sample.

    concept_dict : dict
        Semantic concepts produced by concept_mapper.py.

    sample_path : str, optional
        File path of the original audio sample.

    Returns
    -------
    dict
        Keys: user_id, sample_path, concepts.
    """

    profile = {
        "user_id":     user_id,
        "sample_path": sample_path,
        "concepts":    dict(concept_dict)
    }

    print("\n===================================")
    print("SAMPLE PROFILE CREATED")
    print("===================================")
    print(f"User   : {user_id}")

    if sample_path is not None:
        print(f"Sample : {sample_path}")

    print("Concepts:")

    for key, value in profile["concepts"].items():
        print(f"  {key}: {value:.6f}")

    return profile


# -------------------------------------------------------
# Standalone test
# -------------------------------------------------------

if __name__ == "__main__":

    concepts = {
        "BlowStrength":        0.8234,
        "BlowDuration":        0.7512,
        "RhythmStability":     0.9123,
        "SpectralConsistency": 0.8678,
        "NoiseLevel":          0.1187
    }

    profile = build_user_profile(
        user_id="user01",
        concept_dict=concepts,
        sample_path="dataset/user01/sample01.wav"
    )

    print("\nReturned profile:")
    print(profile)
