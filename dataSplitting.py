import os
import random
import shutil

def split_data():
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_folder = os.path.join(script_dir, "data")
    
    if not os.path.exists(parent_folder):
        print(f"Error: Data folder not found at {parent_folder}")
        print("Please make sure your data folder exists with class subdirectories.")
        return
    
    def split(source_folder, subfolder):
        # Paths for the splits (relative to script directory)
        train_folder = os.path.join(script_dir, "splitData", "train", subfolder)
        test_folder = os.path.join(script_dir, "splitData", "test", subfolder)
        val_folder = os.path.join(script_dir, "splitData", "val", subfolder)

        # Create destination folders if they don't exist
        for folder in [train_folder, test_folder, val_folder]:
            os.makedirs(folder, exist_ok=True)

        # Get all image files (filter out hidden files and non-image files)
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
        files = [f for f in os.listdir(source_folder) 
                if not f.startswith('.') and any(f.lower().endswith(ext) for ext in image_extensions)]
        
        if len(files) == 0:
            print(f"  Warning: No image files found in {subfolder}")
            return
        
        random.shuffle(files)  # Shuffle for randomness

        # Calculate split sizes (75% train, 15% test, 10% val)
        n = len(files)
        train_end = int(0.75 * n)
        test_end = train_end + int(0.15 * n)

        # Split files
        train_files = files[:train_end]
        test_files = files[train_end:test_end]
        val_files = files[test_end:]

        # Copy files (use copy instead of move to preserve originals)
        for f in train_files:
            shutil.copy2(os.path.join(source_folder, f), os.path.join(train_folder, f))

        for f in test_files:
            shutil.copy2(os.path.join(source_folder, f), os.path.join(test_folder, f))

        for f in val_files:
            shutil.copy2(os.path.join(source_folder, f), os.path.join(val_folder, f))

        print(f"  {subfolder}: Train: {len(train_files)}, Test: {len(test_files)}, Val: {len(val_files)}")

    # Process each class folder
    print(f"Looking for data in: {parent_folder}")
    class_folders = [f for f in os.listdir(parent_folder) 
                    if os.path.isdir(os.path.join(parent_folder, f)) and not f.startswith('.')]
    
    if len(class_folders) == 0:
        print(f"Error: No class folders found in {parent_folder}")
        return
    
    print(f"Found {len(class_folders)} class folders: {class_folders}\n")
    
    for subfolder in sorted(class_folders):
        subfolder_path = os.path.join(parent_folder, subfolder)
        print(f"Processing: {subfolder}")
        split(subfolder_path, subfolder)
    
    print(f"\nData splitting complete! Split data saved to: {os.path.join(script_dir, 'splitData')}")

if __name__ == "__main__":
    split_data()