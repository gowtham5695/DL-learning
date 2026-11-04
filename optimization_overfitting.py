import torch
import torch.nn as nn
import matplotlib.pyplot as plt

torch.manual_seed(0)    # same random numbers every run, so results are repeatable

# ----- data: true rule is sin(x), plus some noise -----
x_train = torch.rand(30, 1) * 6 - 3                       # 30 points between -3 and 3
y_train = torch.sin(x_train) + 0.3 * torch.randn(30, 1)   # sin(x) + noise

x_val = torch.rand(100, 1) * 6 - 3                        # 100 NEW points (the real exam)
y_val = torch.sin(x_val) + 0.3 * torch.randn(100, 1)

# ----- a model that is too big for 30 points -----
model = nn.Sequential(
    nn.Linear(1, 128),
    nn.ReLU(),
    nn.Dropout(0.2),  # dropout layer: randomly sets 50% of the neurons to zero during training
    nn.Linear(128, 128),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(128, 1),
)

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01,weight_decay=0.01)

train_losses = []
val_losses = []

# ----- training loop -----
for epoch in range(2000):
    model.train()                        # "training mode" (matters for dropout later)
    pred = model(x_train)
    loss = loss_fn(pred, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model.eval()                         # "exam mode"
    with torch.no_grad():                # don't track gradients: we only want to measure
        val_loss = loss_fn(model(x_val), y_val)

    train_losses.append(loss.item())
    val_losses.append(val_loss.item())

    if epoch % 200 == 0:
        print("epoch", epoch, "train", round(loss.item(), 4), "val", round(val_loss.item(), 4))

# ----- picture 1: loss curves, picture 2: what the model learned -----
x_line = torch.linspace(-3, 3, 200).reshape(-1, 1)
model.eval()
with torch.no_grad():
    y_line = model(x_line)

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label="train loss")
plt.plot(val_losses, label="validation loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.title("Loss curves")

plt.subplot(1, 2, 2)
plt.scatter(x_train, y_train, label="train points")
plt.plot(x_line, torch.sin(x_line), label="true sin(x)")
plt.plot(x_line, y_line, label="model")
plt.legend()
plt.title("What the model learned")

plt.show()