import torch
import torch.nn as nn
from torchvision import models
from common import get_loaders, IMAGENET_MEAN, IMAGENET_STD

def train_with_best_save(net, train_loader, val_loader, epochs=5, lr=1e-3, save_path="best_model.pth"):
    device = "cpu"
    net.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=lr)

    best_acc = 0.0
    for epoch in range(epochs):
        net.train()
        for X, Y in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(net(X), Y)
            loss.backward()
            optimizer.step()

        net.eval()
        correct = total = 0
        with torch.no_grad():
            for X, Y in val_loader:
                correct += (net(X).argmax(1) == Y).sum().item()
                total += Y.size(0)
        acc = correct / total
        print(f"Epoch [{epoch+1}/{epochs}] Val Acc: {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            torch.save(net.state_dict(), save_path)
            print(f"  ^ 最佳模型已保存到 {save_path}")
    print(f"最佳准确率: {best_acc:.4f}")

if __name__ == '__main__':
    net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    net.fc = nn.Linear(net.fc.in_features, 2)
    for p in net.parameters():
        p.requires_grad = False
    for p in net.fc.parameters():
        p.requires_grad = True
    
    train_loader, val_loader = get_loaders("./datasets", batch_size=16)
    train_with_best_save(net, train_loader, val_loader, epochs=5, lr=1e-3)