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
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from net import Net, Net_ori, Net2
from helper_evaluation import set_all_seeds, set_deterministic, compute_accuracy
from helper_train import train_model
from helper_plotting import plot_training_loss, plot_accuracy, plot_data_distribution

# (optional) set a fix place so we don't need to download everytime
DATASET_PATH = "/tmp/nvflare/data"
# (optional) We change to use GPU to speed things up.
# if you want to use CPU, change DEVICE="cpu"
# DEVICE = "cuda:0"
DEVICE = "cuda:0" if torch.cuda.is_available() else 'cpu'


def main():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 4
    epochs = 2

    print(f"------------Train on {DEVICE}-----------")

    train_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    test_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # train_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    # test_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    download = not os.path.exists(os.path.join(DATASET_PATH, 'cifar'))

    batch_size = 4
    epochs = 50

    # Load the datasets
    full_dataset = torchvision.datasets.CIFAR10(root=DATASET_PATH, train=True, download=download, transform=train_transforms)
    trainset = torchvision.datasets.CIFAR10(root=DATASET_PATH, train=True, download=download, transform=train_transforms)
    testset = torchvision.datasets.CIFAR10(root=DATASET_PATH, train=False, download=download, transform=test_transforms)

    # Define validation set
    validation_fraction = 0.1
    num_val_samples = int(validation_fraction * len(trainset))
    valid_indices = torch.arange(len(trainset) - num_val_samples, len(trainset))
    validset = torch.utils.data.Subset(trainset, valid_indices)

    # validation_fraction = 0.1
    # num = int(validation_fraction * 50000)
    # train_indices = torch.arange(0, 50000 - num)
    # valid_indices = torch.arange(50000 - num, 50000)

    # train_sampler = SubsetRandomSampler(train_indices)
    # valid_sampler = SubsetRandomSampler(valid_indices)

    # trainset = torchvision.datasets.CIFAR10(root=DATASET_PATH, train=True, download=download, transform=train_transforms)
    trainloader = torch.utils.data.DataLoader(dataset=trainset,
                                  batch_size=batch_size,
                                  num_workers=2,
                                  drop_last=True)
    # validset = torchvision.datasets.CIFAR10(root=DATASET_PATH, train=True, download=download, transform=test_transforms)
    validloader = torch.utils.data.DataLoader(dataset=validset,
                                  batch_size=batch_size,
                                  num_workers=2)
    # testset = torchvision.datasets.CIFAR10(root=DATASET_PATH, train=False, download=download, transform=test_transforms)
    testloader = torch.utils.data.DataLoader(dataset=testset,
                                batch_size=batch_size,
                                num_workers=2,
                                shuffle=False)

    net = Net_ori()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # (optional) use GPU to speed things up
    net.to(DEVICE)

    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            # (optional) use GPU to speed things up
            inputs, labels = data[0].to(DEVICE), data[1].to(DEVICE)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs, apply_mask=True)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}")
                running_loss = 0.0

    print("Finished Training")

    PATH = "./cifar_net.pth"
    torch.save(net.state_dict(), PATH)

    net = Net2()

    

    net.load_state_dict(torch.load(PATH))
    # (optional) use GPU to speed things up
    net.to(DEVICE)

    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            # (optional) use GPU to speed things up
            images, labels = data[0].to(DEVICE), data[1].to(DEVICE)
            # calculate outputs by running images through the network
            outputs = net(images, apply_mask=False)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Accuracy of the network on the 10000 test images: {100 * correct // total} %")


if __name__ == "__main__":
    main()
