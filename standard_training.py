import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import pandas as pd
from tqdm import tqdm
import itertools
from dist_robust_training import (
    ColoredMnistDataset, ColoredMnistNetwork, evaluate_model, plot_loss_graphs,
    save_color_palette_image, show_dataset_examples, create_dataloaders
)


def train_standard(model, device, train_loader, optimizer, epoch, loss_fn, log_interval=5):
    """
    Train the model with standard training.

    Parameters:
    model (nn.Module): The neural network model to be trained.
    device (torch.device): The device to run the training on (CPU or GPU).
    train_loader (DataLoader): DataLoader for the training dataset.
    optimizer (torch.optim.Optimizer): Optimizer for updating the model parameters.
    epoch (int): The current epoch number.
    loss_fn (callable): Loss function for the network.
    log_interval (int, optional): Interval for logging the training progress. Default is 5.

    Returns:
    list: List of tuples containing epoch, batch index, and training loss.
    """
    model.train()
    training_losses = []
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for batch_idx, (data, target) in enumerate(progress_bar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        training_losses.append((epoch, batch_idx, loss.item()))
        if batch_idx % log_interval == 0:
            progress_bar.set_postfix_str(f"[{batch_idx * len(data)}/{len(train_loader.dataset)}] Train-Loss: {loss.item():.6f}")
    return training_losses


def save_training_results(exp_dir, learning_rate_network, loss_fn_network, batch_size, epochs, train_color_palette, test_color_palette, training_losses, test_losses, test_accuracies, model):
    """
    Save the training results and plots to the specified directory.

    Parameters:
    exp_dir (str): Directory to save the results.
    learning_rate_network (float): Learning rate for the network optimizer.
    loss_fn_network (callable): Loss function for network optimization.
    batch_size (int): Batch size for training and testing.
    epochs (int): Number of epochs for training.
    train_color_palette (list): Color palette for training data.
    test_color_palette (list): Color palette for test data.
    training_losses (list): List of training loss values.
    test_losses (list): List of test loss values.
    test_accuracies (list): List of test accuracy values.
    model (nn.Module): Trained neural network model.
    train_loader (DataLoader): DataLoader for the training dataset.
    device (torch.device): Device used for training.
    """
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    save_model(model, exp_dir)
    results = prepare_results_dataframe(learning_rate_network, loss_fn_network, batch_size, epochs, train_color_palette, test_color_palette, training_losses, test_losses, test_accuracies)
    results.to_excel(os.path.join(exp_dir, 'results.xlsx'), index=False)
    plot_loss_graphs(training_losses, test_losses, exp_dir)
    save_color_palette_image(train_color_palette, test_color_palette, exp_dir, show=False)


def save_model(model, exp_dir):
    """
    Save the model to the specified directory.

    Parameters:
    model (nn.Module): The trained neural network model.
    exp_dir (str): Directory to save the model.
    """
    model_path = os.path.join(exp_dir, 'model.pth')
    torch.save(model.state_dict(), model_path)


def prepare_results_dataframe(learning_rate_network, loss_fn_network, batch_size, epochs, train_color_palette, test_color_palette, training_losses, test_losses, test_accuracies):
    """
    Prepare a DataFrame with the results of the training.

    Parameters:
    learning_rate_network (float): Learning rate for the network optimizer.
    loss_fn_network (callable): Loss function for network optimization.
    batch_size (int): Batch size for training and testing.
    epochs (int): Number of epochs for training.
    train_color_palette (list): Color palette for training data.
    test_color_palette (list): Color palette for test data.
    training_losses (list): List of training loss values.
    test_losses (list): List of test loss values.
    test_accuracies (list): List of test accuracy values.

    Returns:
    pd.DataFrame: DataFrame with the results.
    """
    num_entries = max(len(training_losses), len(test_losses))
    training_epochs, training_batch_indices, training_loss_values = zip(*training_losses)
    test_loss_values, test_epochs, test_batch_indices = zip(*test_losses)

    results = {
        'learning_rate_network': ([learning_rate_network] + [None] * (num_entries - 1))[:num_entries],
        'loss_fn_network': ([loss_fn_network.__name__] + [None] * (num_entries - 1))[:num_entries],
        'batch_size': ([batch_size] + [None] * (num_entries - 1))[:num_entries],
        'epochs': ([epochs] + [None] * (num_entries - 1))[:num_entries],
        'train_color_palette': ([train_color_palette] + [None] * (num_entries - 1))[:num_entries],
        'test_color_palette': ([test_color_palette] + [None] * (num_entries - 1))[:num_entries],
        'training_epochs': list(training_epochs) + [None] * (num_entries - len(training_epochs)),
        'training_batch_indices': list(training_batch_indices) + [None] * (num_entries - len(training_batch_indices)),
        'training_loss_values': list(training_loss_values) + [None] * (num_entries - len(training_loss_values)),
        'test_epochs': list(test_epochs) + [None] * (num_entries - len(test_epochs)),
        'test_batch_indices': list(test_batch_indices) + [None] * (num_entries - len(test_batch_indices)),
        'test_loss_values': list(test_loss_values) + [None] * (num_entries - len(test_loss_values)),
        'test_accuracies': test_accuracies + [None] * (num_entries - len(test_accuracies))
    }

    return pd.DataFrame(results)


def run_standard_experiments(param_grid):
    """
    Run experiments with different parameter combinations. For each combination the training and 
    evaluation process is executed and a folder is created to save the results. Training will run three
    times for each parameter combination to account for randomness in the initialization.
    parameters needed (each one as a list of all possible values):
    - learning_rate_network (float): Learning rate for the network optimizer.
    - loss_fn_network (callable): Loss function for network optimization.
    - batch_size (int): Batch size for training and testing.
    - epochs (int): Number of epochs for training.
    - train_color_palette (list): Color palette for training data: 10x3 list of lists of RGB values. Each inner list contains a RGB
    color mask for a class label.
    - test_color_palette (list): Color palette for test data: same format as above.
    - palette_name (str): Name of the color palette.

    Parameters:
    param_grid (dict): Dictionary where keys are parameter names and values are lists of parameter values.
    """
    keys, values = zip(*param_grid.items())
    for value_combination in itertools.product(*values):
        params = dict(zip(keys, value_combination))
        run_single_standard_experiment(params)


def run_single_standard_experiment(params):
    """
    Run a single standard training experiment with the given parameters.

    Parameters:
    params (dict): Dictionary of parameters for the experiment.
    """
    exp_dir, device, train_loader, test_loader = setup_experiment(params)
    for exp_num in range(1, 4):
        model, optimizer = setup_model_and_optimizer(params, device)
        exp_subdir = create_experiment_subdirectory(exp_dir, exp_num)
        training_losses, test_losses, test_accuracies = train_and_evaluate(model, device, train_loader, test_loader, optimizer, params)
        
        filtered_params = {key: value for key, value in params.items() if key != 'palette_name'}
        
        save_training_results(exp_subdir, **filtered_params, training_losses=training_losses, test_losses=test_losses, test_accuracies=test_accuracies, model=model)


def setup_experiment(params):
    """
    Set up the experiment directory and data loaders.

    Parameters:
    params (dict): Dictionary of parameters for the experiment.

    Returns:
    tuple: Experiment directory, device, train_loader, and test_loader.
    """
    exp_dir = f"Standard_exp_lrNet{params['learning_rate_network']}_lossNet{params['loss_fn_network'].__name__}_batch{params['batch_size']}_epochs{params['epochs']}_palette{params['palette_name']}"
    os.makedirs(exp_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = create_dataloaders(params['train_color_palette'], params['test_color_palette'], params['batch_size'])
    return exp_dir, device, train_loader, test_loader


def setup_model_and_optimizer(params, device):
    """
    Set up the model and optimizer for training.

    Parameters:
    params (dict): Dictionary of parameters for the experiment.
    device (torch.device): Device to run the training on.

    Returns:
    tuple: Model and optimizer.
    """
    model = ColoredMnistNetwork().to(device)
    optimizer = optim.SGD(model.parameters(), lr=params['learning_rate_network'])
    return model, optimizer


def create_experiment_subdirectory(exp_dir, exp_num):
    """
    Create a subdirectory for each experiment repetition.

    Parameters:
    exp_dir (str): Base experiment directory.
    exp_num (int): Experiment repetition number.

    Returns:
    str: Subdirectory path.
    """
    exp_subdir = os.path.join(exp_dir, f"Exp{exp_num}")
    os.makedirs(exp_subdir, exist_ok=True)
    return exp_subdir


def train_and_evaluate(model, device, train_loader, test_loader, optimizer, params):
    """
    Train and evaluate the model.

    Parameters:
    model (nn.Module): The neural network model to be trained.
    device (torch.device): Device to run the training on.
    train_loader (DataLoader): DataLoader for the training dataset.
    test_loader (DataLoader): DataLoader for the testing dataset.
    optimizer (torch.optim.Optimizer): Optimizer for updating the model parameters.
    params (dict): Dictionary of parameters for the experiment.

    Returns:
    tuple: Training losses, test losses, and test accuracies.
    """
    training_losses = []
    test_losses = []
    test_accuracies = []
    for epoch in range(1, params['epochs'] + 1):
        epoch_training_losses = train_standard(model, device, train_loader, optimizer, epoch, params['loss_fn_network'])
        training_losses.extend(epoch_training_losses)
        test_loss, accuracy, epoch_test_losses = evaluate_model(model, device, test_loader, epoch)
        test_losses.extend(epoch_test_losses)
        test_accuracies.append(accuracy)
    return training_losses, test_losses, test_accuracies


def visualize_color_palettes_standard(train_color_palette, test_color_palette):
    """
    Visualize the color palettes for training and testing data.

    Parameters:
    train_color_palette (list): Color palette for training data.
    test_color_palette (list): Color palette for test data.
    """
    train_loader = DataLoader(ColoredMnistDataset(datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor()), train_color_palette), batch_size=64, shuffle=True)
    test_loader = DataLoader(ColoredMnistDataset(datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor()), test_color_palette), batch_size=64, shuffle=False)
    show_dataset_examples(train_loader, test_loader, show=True, save_path=None)