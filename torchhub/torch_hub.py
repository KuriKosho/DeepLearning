import torch
# Khai báo hàm load mô hình ResNet18 từ torch.hub
from torch import hub
import torchvision
# print(torchvision.__version__)
# Load mô hình ResNet18 từ torch.hub
# https:///github.com/
resnet18_model = hub.load('pytorch/vision:main', 'resnet18', pretrained=True)
# In ra mô hình ResNet18
print(resnet18_model)
