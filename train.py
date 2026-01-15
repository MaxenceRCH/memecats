import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

DATA_DIR = "data"
BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-4  # Lower LR for fine-tuning
VAL_SPLIT = 0.2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 5
PATIENCE = 3  # Early stopping patience

# --- Transforms ---
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225]),
])

# --- Dataset ---
full_dataset = datasets.ImageFolder(DATA_DIR)

val_size = int(len(full_dataset) * VAL_SPLIT)
train_size = len(full_dataset) - val_size

train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

train_ds.dataset.transform = train_transform
val_ds.dataset.transform = val_transform

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

# --- Model ---
model = models.resnet50(weights="IMAGENET1K_V1")
# Unfreeze layer1 and layer2 for fine-tuning
for p in model.layer1.parameters():
    p.requires_grad = True
for p in model.layer2.parameters():
    p.requires_grad = True

model.fc = nn.Linear(2048, NUM_CLASSES)
model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
# Train all parameters (fc + unfrozen backbone layers)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# --- Training loop with early stopping ---
best_val_acc = 0.0
patience_counter = 0

for epoch in range(EPOCHS):
    # TRAIN
    model.train()
    train_loss = 0.0

    for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [train]"):
        x, y = x.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    # VALIDATION
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [val]"):
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            preds = out.argmax(dim=1)

            correct += (preds == y).sum().item()
            total += y.size(0)

    val_acc = correct / total

    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"Train loss: {train_loss:.4f} | "
        f"Val acc: {val_acc*100:.2f}%"
    )

    # Early stopping
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0
        # Save best model
        torch.save(model.state_dict(), "face_classifier_best.pt")
        print(f"✓ Best model saved with val_acc: {val_acc*100:.2f}%")
    else:
        patience_counter += 1
        print(f"Patience: {patience_counter}/{PATIENCE}")
        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping! Best val_acc: {best_val_acc*100:.2f}%")
            # Load best model
            model.load_state_dict(torch.load("face_classifier_best.pt"))
            break

torch.save(model.state_dict(), "face_classifier.pt")
print("Model saved")
