import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# ==========================
# CONFIG
# ==========================
ROOT_PATH = r"C:\Users\Professional\PycharmProjects\pythonProject"
IMAGE_PATH = os.path.join(ROOT_PATH, "images")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
NUM_CLASSES = 6
EPOCHS = 40
IMG_SIZE = 224  # Розмір для ResNet18
MODEL_SAVE_PATH = os.path.join(ROOT_PATH, "resnet18_yoga_finetuned.pth")

# ==========================
# DATASET
# ==========================
class YogaDataset(Dataset):
    def __init__(self, df, root_dir, transform=None, folder="train_images", has_labels=True):
        self.df = df.reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform
        self.folder = folder
        self.has_labels = has_labels

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.loc[idx, "image_id"]
        img_path = os.path.join(self.root_dir, self.folder, img_name)
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Не знайдено зображення: {img_path}")

        image = Image.open(img_path).convert("RGB")
        image = image.resize((IMG_SIZE, IMG_SIZE))

        if self.transform:
            image = self.transform(image)

        if self.has_labels:
            label = int(self.df.loc[idx, "class_6"])
            return image, label
        else:
            return image

# ==========================
# EVALUATION
# ==========================
def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = 100 * correct / total if total > 0 else 0
    return acc, all_preds, all_labels

# ==========================
# TRAIN ONE EPOCH
# ==========================
def train_one_epoch(model, loader, criterion, optimizer, epoch):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for inputs, labels in loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_acc = 100 * correct / total if total > 0 else 0
    avg_loss = running_loss / len(loader) if len(loader) > 0 else 0
    print(f"[Epoch {epoch}] Train loss: {avg_loss:.4f} | Train accuracy: {train_acc:.2f}%")
    return train_acc

# ==========================
# MAIN
# ==========================
if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()

    # --- Load CSV ---
    csv_path = os.path.join(ROOT_PATH, "train.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Не знайдено {csv_path}")
    df = pd.read_csv(csv_path)

    # --- Перевірка класів ---
    print("Унікальні класи:", df["class_6"].unique())
    print("Кількість прикладів по класах:\n", df["class_6"].value_counts().sort_index(), "\n")

    # --- Train/Val split ---
    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df["class_6"],
        random_state=42
    )

    # --- WeightedRandomSampler для балансування ---
    counts = train_df["class_6"].value_counts().sort_index().values
    inv_counts = 1.0 / counts
    sample_weights = train_df["class_6"].map(lambda x: inv_counts[x]).values
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    # --- Transforms ---
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize(int(IMG_SIZE * 1.14)),  # 256 → центральний кроп 224
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # --- Datasets & Loaders ---
    train_dataset = YogaDataset(
        df=train_df,
        root_dir=IMAGE_PATH,
        transform=train_transform,
        folder="train_images",
        has_labels=True
    )
    val_dataset = YogaDataset(
        df=val_df,
        root_dir=IMAGE_PATH,
        transform=val_transform,
        folder="train_images",
        has_labels=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # --- Завантажуємо ResNet18 pretrained та розморожуємо layer3/4 + fc ---
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    for name, param in model.named_parameters():
        if not (name.startswith("layer3") or name.startswith("layer4") or name.startswith("fc")):
            param.requires_grad = False

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
    model = model.to(DEVICE)

    # --- Class weights & Loss & Optimizer & Scheduler ---
    class_weights = inv_counts / inv_counts.sum()
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

    optimizer = optim.Adam([
        {'params': model.layer3.parameters(), 'lr': 1e-5},
        {'params': model.layer4.parameters(), 'lr': 1e-5},
        {'params': model.fc.parameters(),    'lr': 3e-4}
    ])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3
    )

    # --- Перед тренуванням ---
    print(f"Device: {DEVICE} | Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}\n")

    best_val_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        train_acc = train_one_epoch(model, train_loader, criterion, optimizer, epoch)
        val_acc, preds, labels = evaluate(model, val_loader)
        print(f"[Epoch {epoch}] Validation Accuracy: {val_acc:.2f}%\n")

        scheduler.step(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"→ Збережено нову BEST модель: Val acc = {best_val_acc:.2f}%\n")

    print(f"Навчання завершено. Найкраща валід. точність: {best_val_acc:.2f}%")

    # --- Confusion Matrix після останньої епохи ---
    cm = confusion_matrix(labels, preds, labels=list(range(NUM_CLASSES)))
    print("Confusion Matrix:\n", cm)
    print("\nClassification Report:\n",
          classification_report(labels, preds, digits=4))
