"""
dataset_loader.py

Loads audio file paths and user labels from the dataset directory.
Expects structure: dataset/<user_id>/<sample_file>
"""

import os


def load_dataset(dataset_path="dataset"):

    audio_paths = []
    labels = []

    print("\n=== LOADING DATASET ===")

    for user in sorted(os.listdir(dataset_path)):

        user_dir = os.path.join(dataset_path, user)

        if not os.path.isdir(user_dir):
            continue

        count = 0

        for fname in sorted(os.listdir(user_dir)):

            path = os.path.join(user_dir, fname)

            if os.path.isfile(path):
                audio_paths.append(path)
                labels.append(user)
                count += 1

        print(f"User={user}, Samples={count}")

    print(f"Total samples={len(audio_paths)}")

    return audio_paths, labels
