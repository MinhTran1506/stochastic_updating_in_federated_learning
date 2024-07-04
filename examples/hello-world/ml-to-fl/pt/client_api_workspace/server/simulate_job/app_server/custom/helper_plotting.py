import os
import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_training_loss(minibatch_loss_list, num_epochs, iter_per_epoch,
                       results_dir=None, averaging_iterations=100):
    """
    Plot the training loss over iterations and epochs, with an optional running average,
    and save the plot to a file if specified.

    Args:
        minibatch_loss_list (list of float): List of loss values recorded at each minibatch iteration.
        num_epochs (int): Total number of epochs the model was trained for.
        iter_per_epoch (int): Number of iterations (minibatches) per epoch.
        results_dir (str, optional): Directory path to save the plot as a PDF. If None, the plot is not saved.
        averaging_iterations (int, optional): Number of iterations over which to average the loss for a smoother plot.
                                              Default is 100.

    The function performs the following steps:
    1. Creates a plot of the minibatch loss values over the iterations.
    2. Optionally adjusts the y-axis limits if there are more than 1000 loss entries to zoom in on the later part of the training.
    3. Adds a running average of the loss values to the plot for smoother trend observation.
    4. Adds a secondary x-axis to represent epochs instead of iterations, providing additional insight into the training progress over time.
    5. Adjusts the layout of the plot to ensure all elements fit properly.
    6. Saves the plot as a PDF file in the specified directory if `results_dir` is provided.

    The plot includes:
    - The raw minibatch loss values plotted against iteration numbers.
    - A running average of the minibatch loss values.
    - A secondary x-axis showing the epoch numbers.

    Returns:
        None
    """
    plt.figure()
    ax1 = plt.subplot(1, 1, 1)
    
    # Plot the minibatch loss against the iteration number.
    ax1.plot(range(len(minibatch_loss_list)),
             (minibatch_loss_list), label='Minibatch Loss')
    
    # This code helps in zooming in on the later part of the training loss.
    if len(minibatch_loss_list) > 1000:
        ax1.set_ylim([
            0, np.max(minibatch_loss_list[1000:])*1.5
        ])
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Loss')

    # Calculate the running average using convolutional function
    # Helps smooth out the noise in the loss values, making it easier to observer the overall trend
    ax1.plot(np.convolve(minibatch_loss_list,
                         np.ones(averaging_iterations,)/averaging_iterations,
                         mode='valid'),
                         label='Running Average')
    ax1.legend()

    ###############
    # Sharing the same y-axis with ax1
    ax2 = ax1.twiny()
    newlabel = list(range(num_epochs+1))

    newpos = [e*iter_per_epoch for e in newlabel]

    ax2.set_xticks(newpos[::10])
    ax2.set_xticklabels(newlabel[::10])

    ax2.xaxis.set_ticks_position('bottom')
    ax2.xaxis.set_label_position('bottom')
    ax2.spines['bottom'].set_position(('outward', 45))
    ax2.set_xlabel('Epochs')
    ax2.set_xlim(ax1.get_xlim())
    ###############

    plt.tight_layout()

    if results_dir is not None:
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        image_path = os.path.join(results_dir, 'plot_training_loss.png')
        plt.savefig(image_path)
        print(f"Image created and saved in {image_path}")


def plot_accuracy(train_acc_list, valid_acc_list, results_dir):
    """
    Plot training and validation accuracy over epochs and save the plot to a file if specified.

    Args:
        train_acc_list (list of float): List of training accuracy values recorded at the end of each epoch.
        valid_acc_list (list of float): List of validation accuracy values recorded at the end of each epoch.
        results_dir (str, optional): Directory path to save the plot as a PDF. If None, the plot is not saved.

    The function performs the following steps:
    1. Plots training and validation accuracy over epochs.
    2. Labels the x-axis as 'Epoch' and the y-axis as 'Accuracy'.
    3. Adds a legend to differentiate between training and validation accuracy.
    4. Adjusts the layout of the plot to ensure all elements fit properly.
    5. Saves the plot as a PDF file in the specified directory if `results_dir` is provided.

    The plot includes:
    - Training accuracy plotted against epoch numbers.
    - Validation accuracy plotted against epoch numbers.

    Returns:
        None
    """
    num_epochs = len(train_acc_list)
    
    plt.plot(np.arange(1, num_epochs+1),
                train_acc_list, label='Training')
    plt.plot(np.arange(1, num_epochs+1),
                valid_acc_list, label='Validation')

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()

    if results_dir is not None:
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        image_path = os.path.join(results_dir, 'plot_acc_training_validation.png')
        plt.savefig(image_path)
        print(f"Image created and saved in {image_path}")


def show_examples(model, data_loader):

    for batch_idx, (features, targets) in enumerate(data_loader):

        with torch.no_grad():
            features = features
            targets = targets
            logits = model(features)
            predictions = torch.argmax(logits, dim=1)
        break

    fig, axes = plt.subplot(nrows=3, ncols=5,
                            sharex=True, sharey=True)
    
    nhwc_img = np.transpose(features, axes=(0, 2, 3, 1))
    nhw_img = np.squeeze(nhwc_img.numpy(), axis=3)

    for idx, ax in enumerate(axes.ravel()):
        ax.imshow(nhw_img[idx], cmap='binary')
        ax.title.set_text(f'P: {predictions[idx]} | T: {targets[idx]}')
        ax.axison = False
    plt.tight_layout()
    plt.show()



