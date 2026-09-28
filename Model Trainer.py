
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import timm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import kagglehub

# Automatically downloads and gets the correct absolute path
target_dir = kagglehub.dataset_download("msambare/fer2013")
print("Downloaded to:", target_dir)

if __name__ == '__main__':
    # Update this path to where your unzipped fer2013 dataset is located
    target_dir = r"C:\Users\jedas\.cache\kagglehub\datasets\msambare\fer2013\versions\1"
    
    train_dir = os.path.join(target_dir, "train")
    test_dir = os.path.join(target_dir, "test")

    if not os.path.exists(train_dir) or not os.path.exists(test_dir):
        raise FileNotFoundError(f"❌ Could not find 'train' or 'test' folders at: {target_dir}")

    print(f"✅ TARGET DATASET DIRECTORY: {target_dir}")

    # Preprocessing: FER2013 images are 48x48 grayscale.
    # Convert to 3 channels to work seamlessly with pretrained models (ResNet, etc.)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load datasets using standard PyTorch ImageFolder
    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    val_dataset = datasets.ImageFolder(root=test_dir, transform=transform)

    # Class maps directly from folder names: ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    id_to_label = {idx: name.title() for idx, name in enumerate(train_dataset.classes)}

    with open("emotion_class_mapping.json", "w") as f:
        json.dump(id_to_label, f, indent=4)

    num_detected_classes = len(id_to_label)
    print(f"Saved 'emotion_class_mapping.json' with {num_detected_classes} classes: {list(id_to_label.values())}\n")

    # Data Loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

    # Setting up the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ResNet34 with 7 output nodes
    model = timm.create_model('resnet34', pretrained=True, num_classes=num_detected_classes)
    model = model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss()

    # ----------------------------------------------------------------------
    # Training and Validation Loop
    EPOCHS = 10  # FER2013 typically requires more epochs to converge than plant diseases
    best_val_loss = float('inf')

    print(f"\nStarting training for {EPOCHS} epochs...\n")

    for epoch in range(EPOCHS):
        # --- Training Phase ---
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            if (batch_idx + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}] | Batch [{batch_idx+1}/{len(train_loader)}] | Loss: {loss.item():.4f}")

        epoch_loss = running_loss / total
        epoch_acc = correct / total

        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_loss = val_loss / val_total
        val_acc = val_correct / val_total

        print(f"\n--- Epoch {epoch+1}/{EPOCHS} Summary ---")
        print(f"Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc*100:.2f}%")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc*100:.2f}%")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_emotion_model.pth")
            print("New best checkpoint saved to 'best_emotion_model.pth'\n")
        else:
            print("\n")

    print("Training Complete. Saved best model weights to 'best_emotion_model.pth'")