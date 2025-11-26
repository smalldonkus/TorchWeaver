import torch
import torch.nn as nn
import torch.nn.functional as F

class GeneratedModel(nn.Module):
    def __init__(self):
        super(GeneratedModel, self).__init__()

        self.layer1 = nn.Linear(in_features=2, out_features=8, bias=True)
        self.layer3 = nn.Linear(in_features=1, out_features=6, bias=True)
        self.activator1 = nn.ReLU()
        self.activator2 = nn.ReLU()
        self.layer2 = nn.Linear(in_features=8, out_features=4, bias=True)
        self.layer4 = nn.Linear(in_features=6, out_features=4, bias=True)
        self.activator3 = nn.ReLU()
        self.activator4 = nn.ReLU()
        self.activator5 = nn.ReLU()

    def forward(self, x):
        input1 = x
        tensorop1a, tensorop1b = torch.split(input1, split_size_or_sections=[2, 1], dim=1)
        layer1 = self.layer1(tensorop1a)
        layer3 = self.layer3(tensorop1b)
        activator1 = self.activator1(layer1)
        activator2 = self.activator2(layer3)
        layer2 = self.layer2(activator1)
        layer4 = self.layer4(activator2)
        activator3 = self.activator3(layer2)
        activator4 = self.activator4(layer4)
        tensorop2 = torch.cat([activator3, activator4], dim=1)
        activator5 = self.activator5(tensorop2)
        return activator5

model = GeneratedModel()
x = torch.randn(5, 3)
output = model(x)
print("Forward Output:", output.shape)