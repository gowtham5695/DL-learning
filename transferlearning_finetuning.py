import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)
torch.manual_seed(0)

# ---------- 1. Data ----------
# the pretrained model expects 224x224 images normalized like ImageNet
train_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),            # small random change = more variety (augmentation)
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
test_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

train_full = datasets.CIFAR10(root="data", train=True, download=True, transform=train_transform)
test_full = datasets.CIFAR10(root="data", train=False, download=True, transform=test_transform)

# keep only a small random part, to act like a small dataset
train_idx = torch.randperm(len(train_full))[:2000].tolist()
test_idx = torch.randperm(len(test_full))[:1000].tolist()
train_data = Subset(train_full, train_idx)
test_data = Subset(test_full, test_idx)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)
print("train images:", len(train_data), " test images:", len(test_data))

# ---------- 2. Load the pretrained model ----------
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)   # trained on ImageNet

# freeze everything
for param in model.parameters():
    param.requires_grad = False

# replace the head: 1000 ImageNet classes -> our 10 classes
model.fc = nn.Linear(512, 10)

model = model.to(device)

total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("total weights:", total, " trainable weights:", trainable)

loss_fn = nn.CrossEntropyLoss()

# ---------- 3. Two small helper functions (so we don't repeat the loops) ----------
def train_one_epoch(optimizer):
    model.train()
    total_loss = 0.0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        loss = loss_fn(model(images), labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(train_loader)

def test_accuracy():
    model.eval()
    correct = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            predictions = model(images).argmax(dim=1)
            correct += (predictions == labels).sum().item()
    return 100 * correct / len(test_data)

# ---------- 4. PHASE 1: feature extraction (train only the new head) ----------
print("\nPHASE 1: only the new head is trained")
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)

for epoch in range(3):
    loss = train_one_epoch(optimizer)
    acc = test_accuracy()
    print("epoch", epoch + 1, "| train loss", round(loss, 4), "| test accuracy", round(acc, 2), "%")

# ---------- 5. PHASE 2: fine-tuning (also unfreeze the last block) ----------
print("\nPHASE 2: last block (layer4) + head are trained")
for param in model.layer4.parameters():
    param.requires_grad = True

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("trainable weights now:", trainable)

optimizer = torch.optim.Adam([
    {"params": model.layer4.parameters(), "lr": 0.0001},   # gentle on pretrained layers
    {"params": model.fc.parameters(),     "lr": 0.0005},   # faster on the head
])

for epoch in range(3):
    loss = train_one_epoch(optimizer)
    acc = test_accuracy()
    print("epoch", epoch + 1, "| train loss", round(loss, 4), "| test accuracy", round(acc, 2), "%")