import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter

class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(128, 1024)
        self.fc2 = nn.Linear(1024, 1)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = x.permute(0, 2, 1)  # Change shape to (batch_size, features, sequence_length)
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = out.mean(2)  # Global Average Pooling
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out