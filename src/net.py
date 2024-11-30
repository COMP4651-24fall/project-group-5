import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self, input_dim=32, output_dim=1):
        super(Net, self).__init__()
        self.embedding = nn.Embedding(360000, 64)
        self.fc1 = nn.Linear(input_dim*64, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, output_dim)
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        x = torch.flatten(x, start_dim=1)
        x = self.leaky_relu(self.fc1(x))
        x = self.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        x = torch.sigmoid(x)
        return x