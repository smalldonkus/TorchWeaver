import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

x = torch.randn(5, 3)
y = torch.randn(5, 4)

class GeneratedModel(nn.Module):
    def __init__(self):
        super(GeneratedModel, self).__init__()

        self.layer1 = nn.Linear(in_features=3, out_features=16, bias=True)
        self.layer3 = nn.Linear(in_features=3, out_features=8, bias=True)
        self.activator1 = nn.ReLU()
        self.activator3 = nn.ReLU()
        self.layer2 = nn.Linear(in_features=16, out_features=8, bias=True)
        self.activator2 = nn.ReLU()
        self.layer4 = nn.Linear(in_features=16, out_features=4, bias=True)
        self.activator4 = nn.ReLU()

    def forward(self, x):
        input1 = x
        layer1 = self.layer1(input1)
        layer3 = self.layer3(input1)
        activator1 = self.activator1(layer1)
        activator3 = self.activator3(layer3)
        layer2 = self.layer2(activator1)
        activator2 = self.activator2(layer2)
        tensorop1 = torch.cat([activator2, activator3], dim=1)
        layer4 = self.layer4(tensorop1)
        activator4 = self.activator4(layer4)
        return activator4


model = GeneratedModel()

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

optimizer.zero_grad()
pred = model(x)
loss = criterion(pred, y)
loss.backward()
optimizer.step()

print ("Foward Output:", pred)
print("Loss:", loss.item())