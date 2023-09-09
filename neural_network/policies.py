import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class SimplePolicy(nn.Module):
    def __init__(self, action_space_size):
        super().__init__()

        # Define your layers
        self.fc1 = nn.Linear(2, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, action_space_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))  # Your layers
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        # Softmax activation
        x = F.softmax(x, dim=1)

        return x


class CriticPolicy(nn.Module):
    def __init__(self):
        super().__init__()

        # Define your layers
        self.fc1 = nn.Linear(2, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x.squeeze(-1)


