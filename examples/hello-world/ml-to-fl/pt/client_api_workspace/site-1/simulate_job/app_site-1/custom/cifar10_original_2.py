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

import os
import torch# type: ignore
import torch.nn as nn# type: ignore
import torch.optim as optim# type: ignore
import torchvision# type: ignore
import torchvision.transforms as transforms# type: ignore
import matplotlib.pyplot as plt # type: ignore
from net import Net, Net2, Net3, Net4
from helper_evaluation import set_all_seeds, set_deterministic, compute_accuracy
from helper_train import train_model
from helper_plotting import plot_training_loss, plot_accuracy, show_examples
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR # type: ignore

# (optional) set a fix place so we don't need to download everytime
#DATASET_PATH = "/tmp/nvflare/data"
DATASET_PATH = "NVFlare/examples/hello-world/ml-to-fl/pt/code/data"

# (optional) We change to use GPU to speed things up.
# if you want to use CPU, change DEVICE="cpu"
# DEVICE = "cuda:0"
DEVICE = "cpu"
RANDOM_SEED = 42

def main():
    set_all_seeds(RANDOM_SEED)
    set_deterministic()

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


    # Add data augmentation
    # transform_train = transforms.Compose([
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomCrop(32, padding=4),
    # transforms.RandomRotation(15),
    # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    # transforms.ToTensor(),
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    # ])

    # transform_test = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    # ])

    batch_size = 4
    epochs = 2

    trainset = torchvision.datasets.CIFAR10(root=DATASET_PATH, train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    validset = torchvision.datasets.CIFAR10(root=DATASET_PATH, train=False, download=True, transform=transform)
    validloader = torch.utils.data.DataLoader(validset, batch_size=batch_size, shuffle=False, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root=DATASET_PATH, train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    # trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    # testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    # # Split training set into training and validation sets
    # train_size = int(0.8 * len(trainset))
    # val_size = len(trainset) - train_size
    # train_subset, val_subset = torch.utils.data.random_split(trainset, [train_size, val_size])

    # trainloader = torch.utils.data.DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2)
    # validloader = torch.utils.data.DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=2)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    net = Net4()

    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.AdamW(net.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=200)

    # (optional) use GPU to speed things up
    net.to(DEVICE)


    minibatch_loss_list, train_acc_list, valid_acc_list = train_model(
        model=net,
        num_epochs=epochs,
        train_loader=trainloader,
        valid_loader=validloader,
        test_loader=testloader,
        optimizer=optimizer,
        criterion=criterion,
        device=DEVICE,
        input_model=None,
        summary_writer=None,
        scheduler=scheduler
    )
    results_dir = "./results"
    print("Saving results...")
    os.makedirs(results_dir, exist_ok=True)

    plot_training_loss(minibatch_loss_list, epochs, len(trainloader), results_dir)
    plot_accuracy(train_acc_list, valid_acc_list, results_dir)
    print("Finished Training")

    PATH = "./cifar_net.pth"
    torch.save(net.state_dict(), PATH)


    


if __name__ == "__main__":
    main()
