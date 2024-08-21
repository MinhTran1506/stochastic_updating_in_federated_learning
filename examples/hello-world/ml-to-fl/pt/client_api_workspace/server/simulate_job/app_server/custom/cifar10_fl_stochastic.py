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
import random
import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.optim as optim # type: ignore
import torchvision # type: ignore
import torchvision.transforms as transforms  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
from net import Net
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR  # type: ignore
from torch.utils.data import SubsetRandomSampler
# (1) import nvflare client API
import nvflare.client as flare

# (optional) metrics
from nvflare.client.tracking import SummaryWriter

# Importing helper functions
from helper_evaluation import set_all_seeds, set_deterministic, compute_accuracy
from helper_train import train_model
from helper_plotting import plot_training_loss, plot_accuracy, show_examples

# (optional) set a fixed place so we don't need to download every time
DATASET_PATH = "/tmp/nvflare/data"
# (optional) We change to use GPU to speed things up.
# if you want to use CPU, change DEVICE="cpu"
DEVICE = "cuda:0" if torch.cuda.is_available() else 'cpu'
RANDOM_SEED = 42

def main():
    # Set seeds and deterministic behavior
    set_all_seeds(RANDOM_SEED)
    set_deterministic()

    # Transform for MNIST
    # train_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    # test_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    # Transform for CIFAR10
    train_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    test_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    download = not os.path.exists(os.path.join(DATASET_PATH, 'cifar-10-batches-py'))

    batch_size = 4
    epochs = 10

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

    # (2) initializes NVFlare client API
    flare.init()

    summary_writer = SummaryWriter()
    while flare.is_running():
        # (3) receives FLModel from NVFlare
        input_model = flare.receive()
        print(f"\n[Current Round={input_model.current_round}, Site = {flare.get_site_name()}]\n")


        # (4) loads model from NVFlare
        try:
            net.load_state_dict(input_model.params)
            print("Model loaded successfully")
        except RuntimeError as e:
            net.load_state_dict(input_model.params, strict=False)
            print(f"Error loading model state_dict: {e}. Initializing new model weights.")

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
        # optimizer = optim.Adam(net.parameters(), lr=0.0005, weight_decay=1e-4)
        # optimizer = optim.AdamW(net.parameters(), lr=0.001)
        # scheduler = CosineAnnealingLR(optimizer, T_max=2)
        scheduler = None
        net.to(DEVICE)

        PATH = "./cifar_net_stochastic.pth"
        # Use the train_model function from helper_train.py
        minibatch_loss_list, train_acc_list, valid_acc_list, accuracy = train_model(
            model=net,
            num_epochs=epochs,
            train_loader=trainloader,
            valid_loader=validloader,
            test_loader=testloader,
            optimizer=optimizer,
            criterion=criterion,
            device=DEVICE,
            input_model=input_model,
            summary_writer=summary_writer,
            scheduler=scheduler
        )

        print("Finished Training")

        torch.save(net.state_dict(), PATH)

        # (5) wraps evaluation logic into a method to re-use for
        #       evaluation on both trained and received model
        # def evaluate(input_weights):
        #     net = Net()
        #     net.load_state_dict(input_weights, strict=False)
        #     net.to(DEVICE)

        #     accuracy = compute_accuracy(net, testloader, DEVICE)
        #     print(f"Accuracy of the network on the 10000 test images: {accuracy:.2f} %")
        #     return accuracy

        # # (6) evaluate on received model for model selection
        # accuracy = evaluate(input_model.params)
        # (7) construct trained FL model
        output_model = flare.FLModel(
            params=net.cpu().state_dict(),
            metrics={"accuracy": accuracy},
            meta={"NUM_STEPS_CURRENT_ROUND": epochs * len(trainloader)},
        )
        # (8) send model back to NVFlare
        flare.send(output_model)

    # Plot the training loss and accuracy
    results_dir = "./results"

    plot_training_loss(minibatch_loss_list, epochs, len(trainloader), results_dir, averaging_iterations=20)
    plot_accuracy(train_acc_list, valid_acc_list, results_dir)



if __name__ == "__main__":
    main()
