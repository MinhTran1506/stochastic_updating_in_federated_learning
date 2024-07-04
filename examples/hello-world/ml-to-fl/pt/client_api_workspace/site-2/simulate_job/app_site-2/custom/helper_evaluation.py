import os
import numpy as np
import random
import torch
from distutils.version import LooseVersion as Version

def set_all_seeds(seed):
    """
    Set the seed for various random number generators to ensure reproducibility.

    Args:
        seed (int): The seed value to set for the random number generators.
    
    This function sets the seed for:
    - Environment variable (PL_GLOBAL_SEED)
    - Python's built-in random module
    - NumPy's random number generator
    - PyTorch's CPU and CUDA random number generators
    """
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_deterministic():
    """
    Configure PyTorch to operate in a deterministic mode to ensure reproducible results.
    
    This function sets various flags and options within PyTorch to ensure that operations 
    involving randomness produce the same results every time they are run. This includes:
    - Disabling CuDNN auto-tuner
    - Forcing CuDNN to use deterministic algorithms
    - Setting the deterministic flag for PyTorch (depending on the version)
    """
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    
    if torch.__version__ <= Version("1.7"):
        torch.set_deterministic(True)
    else:
        torch.use_deterministic_algorithms(True)


def compute_accuracy(model, data_loader, device):
    """
    Compute the accuracy of a model on a given dataset.

    Args:
        model (torch.nn.Module): The model to evaluate.
        data_loader (torch.utils.data.DataLoader): DataLoader for the dataset to evaluate.
        device (torch.device): The device on which to perform computations (e.g., 'cpu' or 'cuda').

    Returns:
        float: The accuracy of the model as a percentage.
    
    This function evaluates the model's accuracy by iterating over the provided DataLoader, 
    making predictions, and comparing them to the true labels. The accuracy is computed as 
    the number of correct predictions divided by the total number of examples, multiplied by 100.
    """
    with torch.no_grad():

        correct_pred, num_examples = 0, 0

        for i, (features, targets) in enumerate(data_loader):

            features = features.to(device)
            targets = targets.float().to(device)

            logits = model(features)
            _, predicted_labels = torch.max(logits, 1)

            num_examples += targets.size(0)
            correct_pred += (predicted_labels == targets).sum()

    return correct_pred.float() / num_examples * 100
