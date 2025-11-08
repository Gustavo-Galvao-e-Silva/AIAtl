import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import mobilenet_v3_small
from PIL import Image
import os
import sys

def main():
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(script_dir, "splitData")
    
    # Check if splitData exists
    if not os.path.exists(DATA_PATH):
        print(f"Error: splitData directory not found at {DATA_PATH}")
        print("\nPlease run dataSplitting.py first to organize your data.")
        print("Make sure you have a 'data' folder with class subdirectories (cardboard, glass, metal, etc.)")
        sys.exit(1)
    
    # Check if train/val/test folders exist
    train_path = os.path.join(DATA_PATH, "train")
    val_path = os.path.join(DATA_PATH, "val")
    test_path = os.path.join(DATA_PATH, "test")
    
    for path, name in [(train_path, "train"), (val_path, "val"), (test_path, "test")]:
        if not os.path.exists(path):
            print(f"Error: {name} directory not found at {path}")
            print("Please run dataSplitting.py first to create the data splits.")
            sys.exit(1)
    
    IMG_SIZE = 224
    BATCH_SIZE = 32
    EPOCHS = 10
    NUM_WORKERS = 0  # safe for Mac

    # Data augmentation for training
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    # Standard transform for validation and test (no augmentation)
    val_test_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    try:
        train_dataset = datasets.ImageFolder(train_path, transform=train_transform)
        val_dataset = datasets.ImageFolder(val_path, transform=val_test_transform)
        test_dataset = datasets.ImageFolder(test_path, transform=val_test_transform)
    except Exception as e:
        print(f"Error loading datasets: {e}")
        print("\nMake sure your splitData directory has the correct structure:")
        print("  splitData/train/class1/, splitData/train/class2/, ...")
        print("  splitData/val/class1/, splitData/val/class2/, ...")
        print("  splitData/test/class1/, splitData/test/class2/, ...")
        sys.exit(1)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Print dataset information
    print("=" * 60)
    print("RECYCLABLE OBJECT CLASSIFIER TRAINING")
    print("=" * 60)
    print(f"Dataset loaded from: {DATA_PATH}")
    print(f"Number of categories: {len(train_dataset.classes)}")
    print(f"Categories: {train_dataset.classes}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Using device: {device}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print("=" * 60 + "\n")
    
    # Check if we have at least 2 classes
    if len(train_dataset.classes) < 2:
        print("Error: Need at least 2 classes for classification.")
        print(f"Found only {len(train_dataset.classes)} class(es): {train_dataset.classes}")
        sys.exit(1)
    
    model = mobilenet_v3_small(weights="IMAGENET1K_V1")

    # Freeze backbone for faster training (unfreeze if you want to fine-tune the whole model)
    for param in model.features.parameters():
        param.requires_grad = False

    num_classes = len(train_dataset.classes)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features=in_features, out_features=num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    print("Starting training...\n")
    
    # Training loop
    best_val_acc = 0.0
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            if (i+1) % 20 == 0:
                print(f"Epoch {epoch+1}/{EPOCHS}, Batch {i+1}/{len(train_loader)}, Loss: {loss.item():.4f}")

        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        
        val_acc = 100 * val_correct / val_total if val_total > 0 else 0.0
        avg_loss = running_loss / len(train_loader) if len(train_loader) > 0 else 0.0
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{EPOCHS} finished.")
        print(f"  Average Loss: {avg_loss:.4f}")
        print(f"  Validation Accuracy: {val_acc:.2f}%")
        print(f"  Learning Rate: {scheduler.get_last_lr()[0]:.6f}\n")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_dir = os.path.join(script_dir, "model")
            os.makedirs(model_dir, exist_ok=True)
            best_model_path = os.path.join(model_dir, "mobilenetv3_recyclable_classifier_best.pth")
            torch.save(model.state_dict(), best_model_path)

    # Final evaluation on test set
    print("=" * 60)
    print("EVALUATING ON TEST SET")
    print("=" * 60)
    model.eval()
    test_correct, test_total = 0, 0
    class_correct = {cls: 0 for cls in test_dataset.classes}
    class_total = {cls: 0 for cls in test_dataset.classes}
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            test_correct += (preds == labels).sum().item()
            test_total += labels.size(0)
            
            # Per-class accuracy
            for i in range(labels.size(0)):
                label = labels[i].item()
                pred = preds[i].item()
                class_name = test_dataset.classes[label]
                class_total[class_name] += 1
                if label == pred:
                    class_correct[class_name] += 1
    
    test_acc = 100 * test_correct / test_total if test_total > 0 else 0.0
    print(f"Test Accuracy: {test_acc:.2f}% ({test_correct}/{test_total})")
    print("\nPer-class Test Accuracy:")
    for cls in test_dataset.classes:
        acc = 100 * class_correct[cls] / class_total[cls] if class_total[cls] > 0 else 0.0
        print(f"  {cls}: {acc:.2f}% ({class_correct[cls]}/{class_total[cls]})")
    print("=" * 60 + "\n")

    # Save final model
    model_dir = os.path.join(script_dir, "model")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "mobilenetv3_recyclable_classifier.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Final test accuracy: {test_acc:.2f}%")
    print(f"Model trained on {num_classes} categories: {train_dataset.classes}")

if __name__ == "__main__":
    main()