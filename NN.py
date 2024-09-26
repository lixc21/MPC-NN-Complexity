import hashlib
import torch.nn as nn


class FullyConnectedNN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_width, hidden_depth):
        super(FullyConnectedNN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_width = hidden_width
        self.hidden_depth = hidden_depth

        layers = []
        layers.append(nn.Linear(input_dim, hidden_width))
        layers.append(nn.ReLU())
        for _ in range(hidden_depth - 1):
            layers.append(nn.Linear(hidden_width, hidden_width))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_width, output_dim))
        self.fc_layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.fc_layers(x)
    

class ResidualBlock(nn.Module):
    def __init__(self, hidden_width):
        super(ResidualBlock, self).__init__()
        self.hidden_width = hidden_width
        self.fc1 = nn.Linear(hidden_width, hidden_width)
        self.fc2 = nn.Linear(hidden_width, hidden_width)
        self.fc3 = nn.Linear(hidden_width, hidden_width)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out += x
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_width, num_blocks):
        super(ResNet, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_width = hidden_width
        self.num_blocks = num_blocks

        layers = []
        layers.append(nn.Linear(input_dim, hidden_width))
        layers.append(nn.ReLU())
        for _ in range(num_blocks):
            layers.append(ResidualBlock(hidden_width))
        layers.append(nn.Linear(hidden_width, output_dim))
        self.resnet_layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.resnet_layers(x)
    
