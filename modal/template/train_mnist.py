# Author: CLAUDE

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def build_model():
    return nn.Sequential(
        nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
        nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
        nn.MaxPool2d(2),                                   # 28 -> 14
        nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
        nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
        nn.MaxPool2d(2),                                   # 14 -> 7
        nn.Flatten(),
        nn.Linear(64 * 7 * 7, 256), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(256, 10),
    )


@torch.no_grad()
def evaluate(model, dl, device):
    model.eval()
    correct = 0
    for x, y in dl:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(1) == y).sum().item()
    return correct / len(dl.dataset)


def main(epochs=5, batch_size=128, max_lr=3e-3, data_dir="data", out_dir="."):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(42)
    print(f"Using {device}")

    # --- data ---
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train = datasets.MNIST(data_dir, train=True, download=True, transform=tf)
    test = datasets.MNIST(data_dir, train=False, download=True, transform=tf)
    train_dl = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=2)
    test_dl = DataLoader(test, batch_size=1000)

    # --- model ---
    model = build_model().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=max_lr, total_steps=epochs * len(train_dl))

    # --- train ---
    acc = 0.0
    for epoch in range(epochs):
        model.train()
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            loss = F.cross_entropy(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        acc = evaluate(model, test_dl, device)
        print(f"epoch {epoch + 1}: loss {loss.item():.4f}  test acc {acc:.4f}")

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "mnist_cnn.pt")
    torch.save(model.state_dict(), path)
    print(f"saved {path}")
    return acc


if __name__ == "__main__":
    main()