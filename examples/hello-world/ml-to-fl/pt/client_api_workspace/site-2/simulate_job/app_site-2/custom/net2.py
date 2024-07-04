import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import random
import numpy as np

class Net(nn.Module):
    def __init__(self, seed=None):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.seed_layer = nn.Linear(84, 100)  # New layer with 100 nodes
        self.mask = None  # To store the mask

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        seed_output = self.seed_layer(x)
        
        # Create a random mask of the same shape as seed_output
        if self.training:  # Apply dropout only during training
            self.mask = (torch.rand(seed_output.size()) > 0.5).float()  # Example dropout rate of 0.5
            masked_seed_output = self.mask * seed_output
        else:
            masked_seed_output = seed_output * 0.5  # During inference, multiply by the keep probability
        
        x = self.fc3(masked_seed_output)
        return x

    def apply_mask_to_weights(self):
        if self.mask is not None:
            for name, param in self.named_parameters():
                if 'seed_layer' in name:
                    param.data *= self.mask    