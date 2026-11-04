import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

# ---------- 1. Data: MNIST (70,000 handwritten digits, 0 to 9) ----------
transform = transforms.Compose([
    transforms.ToTensor(),                       # picture -> tensor with values 0 to 1
    transforms.Normalize((0.1307,), (0.3081,)),  # scale like StandardScaler (mean, std of MNIST)
])

train_data = datasets.MNIST(root="data", train=True, download=True, transform=transform)
test_data = datasets.MNIST(root="data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=256, shuffle=False)

print("training images:", len(train_data), " test images:", len(test_data))

# ---------- 2. The model (a class, like MyNet in Topic 6) ----------
class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)   # 1 -> 6 channels
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)             # 6 -> 16 channels
        self.pool = nn.MaxPool2d(2)                              # halves height and width
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)                             # 10 digits

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))    # (batch, 6, 14, 14)
        x = self.pool(self.relu(self.conv2(x)))    # (batch, 16, 5, 5)
        x = x.flatten(1)                           # (batch, 400)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)                         # raw scores, NO softmax (Topic 4 rule)

model = LeNet().to(device)

# quick shape check: feed one fake image and print the output shape
fake = torch.zeros(1, 1, 28, 28).to(device)
print("output shape:", model(fake).shape)          # torch.Size([1, 10])

total_weights = sum(p.numel() for p in model.parameters())
print("total weights:", total_weights)             # 61,706

# ---------- 3. Loss and optimizer ----------
loss_fn = nn.CrossEntropyLoss()                    # raw scores + integer labels
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
git 
# ---------- 4. Train and test ----------
for epoch in range(5):
    # --- training ---
    model.train()
    total_loss = 0.0
    for images, labels in train_loader:
        images = images.to(device)                 # data must be on the same device as the model
        labels = labels.to(device)

        outputs = model(images)
        loss = loss_fn(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # --- testing ---
    model.eval()
    correct = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            predictions = outputs.argmax(dim=1)    # the digit with the highest score
            correct += (predictions == labels).sum().item()

    accuracy = 100 * correct / len(test_data)
    print("epoch", epoch + 1,
          "| train loss", round(total_loss / len(train_loader), 4),
          "| test accuracy", round(accuracy, 2), "%")

# ---------- 5. Look at some predictions ----------
images, labels = next(iter(test_loader))
model.eval()
with torch.no_grad():
    predictions = model(images.to(device)).argmax(dim=1).cpu()

plt.figure(figsize=(12, 3))
for i in range(8):
    plt.subplot(1, 8, i + 1)
    plt.imshow(images[i][0], cmap="gray")
    plt.title("pred " + str(predictions[i].item()))
    plt.axis("off")
plt.show()