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
from net import Net
from helper_evaluation import set_all_seeds, set_deterministic, compute_accuracy
from helper_train import train_model
from helper_plotting import plot_training_loss, plot_accuracy, show_examples
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR # type: ignore
from torch.utils.data import SubsetRandomSampler

# (optional) set a fix place so we don't need to download everytime
#DATASET_PATH = "/tmp/nvflare/data"
DATASET_PATH = "NVFlare/examples/hello-world/ml-to-fl/pt/code/data"

# (optional) We change to use GPU to speed things up.
# if you want to use CPU, change DEVICE="cpu"
# DEVICE = "cuda:0"
DEVICE = "cuda:0" if torch.cuda.is_available() else 'cpu'
RANDOM_SEED = 42

def main():
    set_all_seeds(RANDOM_SEED)
    set_deterministic()

    train_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    test_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    download = not os.path.exists(os.path.join(DATASET_PATH, 'cifar-10-batches-py'))

    batch_size = 4
    epochs = 50

    print(f"------------Train on {DEVICE}-----------")

    train_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    test_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # train_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    # test_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    download = not os.path.exists(os.path.join(DATASET_PATH, 'cifar'))

    trainset = torchvision.datasets.CIFAR10(root=DATASET_PATH, train=True, download=download, transform=train_transforms)
    validset = torchvision.datasets.CIFAR10(root=DATASET_PATH, train=True, download=download, transform=test_transforms)
    testset = torchvision.datasets.CIFAR10(root=DATASET_PATH, train=False, download=download, transform=test_transforms)


    validation_fraction = 0.1
    num = int(validation_fraction * len(trainset))
    train_indices = torch.arange(0, len(trainset) - num)
    valid_indices = torch.arange(len(trainset) - num, len(trainset))

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(valid_indices)

    trainloader = torch.utils.data.DataLoader(dataset=trainset,
                                  batch_size=batch_size,
                                  num_workers=2,
                                  drop_last=True,
                                  sampler=train_sampler)
    validloader = torch.utils.data.DataLoader(dataset=validset,
                                  batch_size=batch_size,
                                  num_workers=2,
                                  sampler=valid_sampler)

    testloader = torch.utils.data.DataLoader(dataset=testset,
                                batch_size=batch_size,
                                num_workers=2,
                                shuffle=False)

    net = Net()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    # optimizer = optim.AdamW(net.parameters(), lr=0.001, weight_decay=1e-4)
    # scheduler = CosineAnnealingLR(optimizer, T_max=200)
    PATH = "./cifar_net.pth"

    # (optional) use GPU to speed things up
    net.to(DEVICE)


    minibatch_loss_list, train_acc_list, valid_acc_list, accuracy = train_model(
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
        # scheduler=scheduler
    )
    results_dir = "./results"
    print("Saving results...")
    os.makedirs(results_dir, exist_ok=True)

    plot_training_loss(minibatch_loss_list, epochs, len(trainloader), results_dir)
    plot_accuracy(train_acc_list, valid_acc_list, results_dir)
    print("Finished Training")

    torch.save(net.state_dict(), PATH)


    


if __name__ == "__main__":
    main()
