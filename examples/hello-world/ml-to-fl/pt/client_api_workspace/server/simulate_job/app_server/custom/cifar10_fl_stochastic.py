

import os
import random
import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.optim as optim # type: ignore
import torchvision # type: ignore
import torchvision.transforms as transforms  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
from net import Net
from torch.optim.lr_scheduler import CosineAnnealingLR  # type: ignore
from torch.utils.data import SubsetRandomSampler
# (1) import nvflare client API
import nvflare.client as flare

# (optional) metrics
from nvflare.client.tracking import SummaryWriter

# Importing helper functions
from helper_evaluation import set_all_seeds, set_deterministic, compute_accuracy
from helper_train import train_model
from helper_plotting import plot_training_loss, plot_accuracy, plot_data_distribution

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

    print(f"------------Train on {DEVICE}-----------")
    # Transform for MNIST
    # train_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    # test_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    # Transform for CIFAR10
    train_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    test_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


    download = not os.path.exists(os.path.join(DATASET_PATH, 'cifar10-batches-py'))

    batch_size = 4
    epochs = 10

    full_dataset = torchvision.datasets.CIFAR10(root=DATASET_PATH, train=True, download=download, transform=train_transforms)
    trainset = torchvision.datasets.CIFAR10(root=DATASET_PATH, train=True, download=download, transform=train_transforms)
    testset = torchvision.datasets.CIFAR10(root=DATASET_PATH, train=False, download=download, transform=test_transforms)
    validset = torchvision.datasets.CIFAR10(root=DATASET_PATH, train=True, download=download, transform=test_transforms)

    # # Define validation set
    # validation_fraction = 0.1
    # num_val_samples = int(validation_fraction * len(trainset))
    # valid_indices = torch.arange(len(trainset) - num_val_samples, len(trainset))
    # validset = torch.utils.data.Subset(trainset, valid_indices)

    # # trainset = torchvision.datasets.CIFAR10(root=DATASET_PATH, train=True, download=download, transform=train_transforms)
    # trainloader = torch.utils.data.DataLoader(dataset=trainset,
    #                               batch_size=batch_size,
    #                               num_workers=2,
    #                               drop_last=True)
    # # validset = torchvision.datasets.CIFAR10(root=DATASET_PATH, train=True, download=download, transform=test_transforms)
    # validloader = torch.utils.data.DataLoader(dataset=validset,
    #                               batch_size=batch_size,
    #                               num_workers=2)
    # # testset = torchvision.datasets.CIFAR10(root=DATASET_PATH, train=False, download=download, transform=test_transforms)
    # testloader = torch.utils.data.DataLoader(dataset=testset,
    #                             batch_size=batch_size,
    #                             num_workers=2,
    #                             shuffle=False)

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
    
    results_dir = "./results"

    if not os.path.exists(results_dir):
        os.makedirs(results_dir, exist_ok=True)
    # Plot data distribution for the full dataset before splitting
    plot_data_distribution(full_dataset, "Full Dataset", os.path.join(results_dir, "full_dataset_distribution.png"))

    # Plot data distribution for each set
    plot_data_distribution(trainset, "Training Set", os.path.join(results_dir, "trainset_distribution.png"))
    plot_data_distribution(validset, "Validation Set", os.path.join(results_dir, "validset_distribution.png"))
    plot_data_distribution(testset, "Test Set", os.path.join(results_dir, "testset_distribution.png"))

    net = Net(device=DEVICE)

    # (2) initializes NVFlare client API
    flare.init()

    summary_writer = SummaryWriter()
    while flare.is_running():
        # (3) receives FLModel from NVFlare
        input_model = flare.receive()
        print(f"\n[Current Round={input_model.current_round}, Site = {flare.get_site_name()}]\n")


        # (4) loads model from NVFlare
        try:
            net.load_state_dict(input_model.params, strict=False)
            print("Model loaded successfully with strict=False")
        except RuntimeError as e:
            print(f"Error loading model state_dict: {e}. Initializing new model weights.")

        criterion = nn.CrossEntropyLoss()
        # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
        optimizer = optim.SGD(net.parameters(), lr=0.0005, momentum=0.9, weight_decay=5e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)

        net.to(DEVICE)

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
            scheduler=scheduler,
            stochastic=True
        )

        print("Finished Training")

        PATH = "./cifar_net_stochastic.pth"
        torch.save(net.state_dict(), PATH)

        output_model = flare.FLModel(
            params=net.cpu().state_dict(),
            metrics={"accuracy": accuracy},
            meta={"NUM_STEPS_CURRENT_ROUND": epochs * len(trainloader)},
        )
        # (8) send model back to NVFlare
        flare.send(output_model)

    # Plot the training loss and accuracy
    plot_training_loss(minibatch_loss_list, epochs, len(trainloader), results_dir, averaging_iterations=20)
    plot_accuracy(train_acc_list, valid_acc_list, results_dir)



if __name__ == "__main__":
    main()
