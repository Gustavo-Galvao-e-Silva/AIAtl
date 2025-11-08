import os
import random
import shutil

# Path to your folder
parent_folder = "AIAtl/data"


def split(source_folder, subfolder):
    # Paths for the splits
    train_folder = f"AIAtl/splitData/train/{subfolder}"
    test_folder = f"AIAtl/splitData/test/{subfolder}"
    val_folder = f"AIAtl/splitData/val/{subfolder}"

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

for subfolder in os.listdir(parent_folder):
    subfolder_path = os.path.join(parent_folder, subfolder)
    if not os.path.isdir(subfolder_path) or subfolder.startswith('.'): #patch for us mac users LOL
        continue
    print("Processing:", subfolder_path)
    split(subfolder_path, subfolder)