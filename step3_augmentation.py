import torch
import torch.nn as nn
from torchvision import models, transforms
from common import get_loaders, train_and_eval, IMAGENET_MEAN, IMAGENET_STD
from step2_pretrained import AlexNetManual

def main():
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    train_loader, val_loader = get_loaders("./datasets", batch_size=16, train_transform=train_transform)

    net = AlexNetManual(num_classes=1000)
    official = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
    net.load_state_dict(official.state_dict(), strict=False)
    net.classifier[6] = nn.Linear(4096, 2)
    
    for name, param in net.named_parameters():
        param.requires_grad = name.startswith("classifier")

    acc = train_and_eval(net, train_loader, val_loader, epochs=30, lr=1e-3)
    print(f"\n【第三步 预训练+数据增强】最终准确率: {acc:.4f}")

if __name__ == '__main__':
    main()