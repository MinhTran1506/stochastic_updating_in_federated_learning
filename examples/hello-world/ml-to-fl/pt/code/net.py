import torch
import torch.nn as nn
import torch.nn.functional as F
DEVICE = "cuda:0" if torch.cuda.is_available() else 'cpu'


    
class Net_original(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x, apply_mask=False):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    

# If using CIFAR -> img_rows=32, img_cols=32, channels=3
# If using MNIST -> img_rows=28, img_cols=28, channels=1
class Net_old_cifar(nn.Module): 
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
        
        self.fc1 = nn.Linear(4*num_filters * (img_rows // 8) * (img_cols // 8), 512)
        self.bn7 = nn.BatchNorm1d(512)
        self.dropout_dense = nn.Dropout(dropout_dense)
        self.fc2 = nn.Linear(512, num_classes)

        # Initialize the random mask for fc1
        self.p = 0.5  # -> Retention probability
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
class NetNoStochasticUpdating(nn.Module):
    def __init__(self, num_filters=32, dropout_conv=0.5, dropout_dense=0.5, img_rows=32, img_cols=32, channels=3, num_classes=10, device=DEVICE):
        super().__init__()
        self.device = device
        self.num_filters = num_filters
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
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x, apply_mask=False):
        # Pass through convolutional layers
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
        
        # Flatten and pass through dense layers
        x = x.view(-1, 4*self.num_filters * (x.size(2) // 1) * (x.size(3) // 1))
        x = F.relu(self.bn7(self.fc1(x)))
        x = self.fc2(x)
        
        return x


# prev : dropout_conv=0.25, dropout_dense=0.25
class Net(nn.Module):
    def __init__(self, num_filters=32, dropout_conv=0.5, dropout_dense=0.5, img_rows=32, img_cols=32, channels=3, num_classes=10, device=DEVICE):
        super().__init__()
        self.device = device
        self.num_filters = num_filters
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
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout_dense = dropout_dense

        # Placeholder for dropout mask for the dense layer
        self.dense_mask = None
    
    def forward(self, x, apply_mask=False):
        # Pass through convolutional layers
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
        
        # Flatten and pass through dense layers
        x = x.view(-1, 4*self.num_filters * (x.size(2) // 1) * (x.size(3) // 1))
        x = F.relu(self.bn7(self.fc1(x)))
        if apply_mask and self.dense_mask is not None:
            self.dense_mask = self.dense_mask.to(x.device)
            x = x * self.dense_mask
            x = x / (1 - self.dropout_dense)
        x = self.fc2(x)
        
        return x
    
    def resample_dropout_masks(self, x):
        # Resample dropout mask for the dense layer
        self.dense_mask = torch.bernoulli(torch.ones(self.fc1.out_features) * (1 - self.dropout_dense)).to(self.device)
        # self.dense_mask /= (1 - self.dropout_dense)  # Scale by 1/(1-p) to maintain expected values

