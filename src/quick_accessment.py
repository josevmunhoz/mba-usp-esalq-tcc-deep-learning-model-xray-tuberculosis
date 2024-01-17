import torch
from torchvision.transforms import v2
from torchvision import datasets

path = "../archive/data/classified"

train_data = datasets.ImageFolder(root=path,
                                transform=v2.ToTensor(),
                                target_transform=None)

print(train_data)