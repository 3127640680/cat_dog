import torch
import torch.nn as nn
from torchvision import models
from common import get_loaders, train_and_eval

class AlexNet(nn.Module):
    def __init__(self):
        """初始化"""
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Flatten(),
            nn.Linear(6400, 4096), nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096), nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 2))
    
    def forward(self, X):
        """前向传播"""
        return self.net(X)

def init_weight(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.xavier_uniform_(m.weight)

if __name__ == '__main__':
    batch_size = 16
    lr = 0.0001
    epochs = 30
    device = 'cpu'

    net = AlexNet()
    net.apply(init_weight)
    net.to(device)

    train_loader, val_loader = get_loaders("./datasets", batch_size=batch_size)
    
    print("开始训练...")
    acc = train_and_eval(net, train_loader, val_loader, epochs=epochs, lr=lr, device=device)
    print(f"\n【第一步 手动搭建+从零训练】最终准确率: {acc:.4f}")