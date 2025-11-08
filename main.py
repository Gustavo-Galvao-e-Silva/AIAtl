import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

def main():
    DATA_PATH = r"D:\hcmeh\Documents\Romil Mehta - School Related\2nd Year\RecycleVision\AIAtl\data\cardboard_split"
    IMG_SIZE = 224
    BATCH_SIZE = 8
    EPOCHS = 5
    NUM_WORKERS = 0  # increase to 2–4 if your machine allows

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.Lambda(lambda im: im.convert("RGB")),  # ensure 3 channels
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(os.path.join(DATA_PATH, "train"), transform=transform)
    val_dataset   = datasets.ImageFolder(os.path.join(DATA_PATH, "val"),   transform=transform)
    test_dataset  = datasets.ImageFolder(os.path.join(DATA_PATH, "test"),  transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Use weights enum (requires internet to download on first run)
    model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)

    # Freeze backbone
    for p in model.features.parameters():
        p.requires_grad = False

    num_classes = len(train_dataset.classes)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=1e-3)

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for i, (images, labels) in enumerate(train_loader, start=1):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 10 == 0:
                print(f"Epoch {epoch+1}, Batch {i}/{len(train_loader)}, Loss: {loss.item():.4f}")

        # Validation
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                preds = outputs.argmax(1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = 100.0 * correct / total if total > 0 else 0.0
        avg_loss = running_loss / max(len(train_loader), 1)
        print(f"Epoch {epoch+1} finished. Avg Loss: {avg_loss:.4f}, Val Acc: {val_acc:.2f}%\n")

    # Optional: evaluate on test set
    model.eval()
    test_correct = test_total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(1)
            test_correct += (preds == labels).sum().item()
            test_total += labels.size(0)
    if test_total > 0:
        print(f"Test Acc: {100.0 * test_correct / test_total:.2f}%")

    torch.save(model.state_dict(), "mobilenetv3_ewaste.pth")
    print("Model saved to mobilenetv3_ewaste.pth")

if __name__ == "__main__":
    main()