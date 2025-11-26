import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

x = torch.randn(5, 3)
y = torch.randn(5, 1)

class GeneratedModel(nn.Module):
    def __init__(self):
        super(GeneratedModel, self).__init__()

        self.layer2 = nn.Linear(in_features=3, out_features=1, bias=True)

    def forward(self, x):
        input2 = x
        layer2 = self.layer2(input2)
        return layer2
    
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