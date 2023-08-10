import torch.nn as nn
import torch.nn.functional as F


class CNNPolicy(nn.Module):
    def __init__(self, action_space_size):
        super(CNNPolicy, self).__init__()

        # Define the convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2)

        # Apply max pooling to reduce the size further
        self.pool = nn.MaxPool2d(2, 2)

        # Define the fully connected layers
        self.fc1 = nn.Linear(
            128 * 27 * 27, 512
        )  # Adjust the size according to the output of conv3 and pooling
        self.fc2 = nn.Linear(512, action_space_size)

    def forward(self, x):
        # Apply the convolutional layers with ReLU activations
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x)  # Apply max pooling

        # Flatten the output for the fully connected layers
        x = x.view(x.size(0), -1)

        # Apply the fully connected layers with ReLU activations
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        # Apply a softmax to get the action probabilities
        x = F.softmax(x, dim=1)

        return x
