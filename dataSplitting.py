import os
import random
import shutil

# Path to your folder
source_folder = "data/e-waste"

def split(source_folder):
    # Paths for the splits
    train_folder = "data/e-waste_split/train"
    test_folder = "data/e-waste_split/test"
    val_folder = "data/e-waste_split/val"

    # Create destination folders if they don't exist
    for folder in [train_folder, test_folder, val_folder]:
        os.makedirs(folder, exist_ok=True)

    # Get all files
    files = os.listdir(source_folder)
    random.shuffle(files)  # Shuffle for randomness

    # Calculate split sizes
    n = len(files)
    train_end = int(0.75 * n)
    test_end = train_end + int(0.15 * n)

    # Split files
    train_files = files[:train_end]
    test_files = files[train_end:test_end]
    val_files = files[test_end:]

    # Move files
    for f in train_files:
        shutil.move(os.path.join(source_folder, f), os.path.join(train_folder, f))

    for f in test_files:
        shutil.move(os.path.join(source_folder, f), os.path.join(test_folder, f))

    for f in val_files:
        shutil.move(os.path.join(source_folder, f), os.path.join(val_folder, f))

    print(f"Train: {len(train_files)}, Test: {len(test_files)}, Val: {len(val_files)}")

split(source_folder)