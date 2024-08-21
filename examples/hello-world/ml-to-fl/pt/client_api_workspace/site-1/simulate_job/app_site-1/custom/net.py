# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import random
import numpy as np
import torch.nn.init as init

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
class Net_new123(nn.Module): 
    def __init__(self, num_filters=32, dropout_conv=0.25, dropout_dense=0.5, img_rows=32, img_cols=32, channels=3, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, num_filters, kernel_size=3, padding='same')
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding='same')
        self.bn2 = nn.BatchNorm2d(num_filters)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout_conv = nn.Dropout2d(dropout_conv)
        
        self.conv3 = nn.Conv2d(num_filters, 2*num_filters, kernel_size=3, padding='same')
        self.bn3 = nn.BatchNorm2d(2*num_filters)
        self.conv4 = nn.Conv2d(2*num_filters, 2*num_filters, kernel_size=3, padding='same')
        self.bn4 = nn.BatchNorm2d(2*num_filters)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv5 = nn.Conv2d(2*num_filters, 4*num_filters, kernel_size=3, padding='same')
        self.bn5 = nn.BatchNorm2d(4*num_filters)
        self.conv6 = nn.Conv2d(4*num_filters, 4*num_filters, kernel_size=3, padding='same')
        self.bn6 = nn.BatchNorm2d(4*num_filters)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(4*num_filters * (img_rows // 8) * (img_cols // 8), 512)
        self.bn7 = nn.BatchNorm1d(512)
        self.dropout_dense = nn.Dropout(dropout_dense)
        self.fc2 = nn.Linear(512, num_classes)

        # Initialize the random mask for fc1
        self.p = dropout_dense  # -> Retention probability
        self.random_mask_fc1 = self.random_mask(self.fc1.weight.shape, p=self.p)

    def random_mask(self, shape, p=0.5):
        """ Generate a random mask with probability `p` of keeping each element. """
        return torch.bernoulli(torch.ones(shape) * p)
    
    def forward(self, x, apply_mask=True):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.dropout_conv(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        x = self.dropout_conv(x)
        
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.pool3(x)
        x = self.dropout_conv(x)
        
        x = torch.flatten(x, 1)

        # -> Apply the random mask to the first Linear layer (fc1) and scale weights by p
        masked_weights_fc1 = self.fc1.weight * self.random_mask_fc1
        if apply_mask:
            x = F.linear(x, masked_weights_fc1, self.fc1.bias)
        else:
            x = F.linear(x, self.fc1.weight * self.p, self.fc1.bias)  # Scale weights by p during inference
        x = F.relu(self.bn7(x))
        x = self.dropout_dense(x)

        # No mask applied to the second Linear layer (fc2)
        x = self.fc2(x)

        return x
    
class Net2(nn.Module):
    def __init__(self, num_filters=32, dropout_conv=0, dropout_dense=0.5, img_rows=32, img_cols=32, channels=3, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, num_filters, kernel_size=3, padding='same')
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding='same')
        self.bn2 = nn.BatchNorm2d(num_filters)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout_conv = nn.Dropout2d(dropout_conv)
        
        self.conv3 = nn.Conv2d(num_filters, 2*num_filters, kernel_size=3, padding='same')
        self.bn3 = nn.BatchNorm2d(2*num_filters)
        self.conv4 = nn.Conv2d(2*num_filters, 2*num_filters, kernel_size=3, padding='same')
        self.bn4 = nn.BatchNorm2d(2*num_filters)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv5 = nn.Conv2d(2*num_filters, 4*num_filters, kernel_size=3, padding='same')
        self.bn5 = nn.BatchNorm2d(4*num_filters)
        self.conv6 = nn.Conv2d(4*num_filters, 4*num_filters, kernel_size=3, padding='same')
        self.bn6 = nn.BatchNorm2d(4*num_filters)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(4*num_filters * (img_rows // 8) * (img_cols // 8), 512) # Increase from 512 to 2048
        self.bn7 = nn.BatchNorm1d(512)
        self.dropout_dense = nn.Dropout(dropout_dense)
        self.fc2 = nn.Linear(512, num_classes)
    
    def forward(self, x, apply_mask=False):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.dropout_conv(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        x = self.dropout_conv(x)
        
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.pool3(x)
        x = self.dropout_conv(x)
        
        x = torch.flatten(x, 1)
        x = F.relu(self.bn7(self.fc1(x)))
        x = self.dropout_dense(x)
        # x = self.fc2(x)

        # Apply random mask on the last Linear layer
        mask = torch.bernoulli(torch.ones(self.fc2.weight.shape) * 0.5)
        x = F.linear(x, self.fc2.weight * mask, self.fc2.bias)

        return x
