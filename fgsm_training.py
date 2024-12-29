import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import pandas as pd
from tqdm import tqdm
import itertools
from dist_robust_training import show_dataset_examples, ColoredMnistDataset, ColoredMnistNetwork, evaluate_model, plot_loss_graphs, save_color_palette_image, show_adversarial_example_images


def fgsm_attack(model, data, target, epsilon, loss_fn):
    """
    Perform FGSM attack to generate adversarial examples.

    Parameters:
    model (nn.Module): The neural network model.
    data (torch.Tensor): Input data.
    target (torch.Tensor): True labels for the data.
    epsilon (float): Perturbation magnitude.
    loss_fn (callable): Loss function for adversarial attack.

    Returns:
    torch.Tensor: Adversarial examples.
    """
    data.requires_grad = True
    output = model(data)
    loss = loss_fn(output, target)
    model.zero_grad()
    loss.backward()
    data_grad = data.grad.data
    perturbed_data = data + epsilon * data_grad.sign()
    return torch.clamp(perturbed_data, 0, 1)


def train_fgsm(model, device, train_loader, optimizer, epoch, epsilon, loss_fn, log_interval=5):
    """
    Train the model with FGSM adversarial examples.

    Parameters:
    model (nn.Module): The neural network model to be trained.
    device (torch.device): The device to run the training on (CPU or GPU).
    train_loader (DataLoader): DataLoader for the training dataset.
    optimizer (torch.optim.Optimizer): Optimizer for updating the model parameters.
    epoch (int): The current epoch number.
    epsilon (float): Perturbation magnitude for FGSM attack.
    loss_fn (callable): Loss function for adversarial attack.
    log_interval (int, optional): Interval for logging the training progress. Default is 10.

    Returns:
    list: List of tuples containing epoch, batch index, and training loss.
    """
    model.train()
    training_losses = []
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for batch_idx, (data, target) in enumerate(progress_bar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        perturbed_data = fgsm_attack(model, data, target, epsilon, loss_fn)
        loss = compute_loss(model, perturbed_data, target)
        optimizer.step()
        training_losses.append((epoch, batch_idx, loss.item()))
        if batch_idx % log_interval == 0:
            progress_bar.set_postfix_str(f"[{batch_idx * len(data)}/{len(train_loader.dataset)}] Train-Loss: {loss.item():.6f}")
    return training_losses


def compute_loss(model, data, target):
    """
    Compute the loss for the given data and target.

    Parameters:
    model (nn.Module): The neural network model.
    data (torch.Tensor): Input data.
    target (torch.Tensor): True labels for the data.

    Returns:
    torch.Tensor: The loss value.
    """
    output = model(data)
    loss = F.nll_loss(output, target)
    loss.backward()
    return loss


def save_training_results(exp_dir, epsilon, learning_rate_network, loss_fn_network, loss_fn_fgsm, batch_size, epochs, train_color_palette, test_color_palette, training_losses, test_losses, test_accuracies, model, train_loader, device):
    """
    Save the training results and plots to the specified directory.

    Parameters:
    exp_dir (str): Directory to save the results.
    epsilon (float): Perturbation magnitude for FGSM attack.
    learning_rate_network (float): Learning rate for the network optimizer.
    loss_fn_network (callable): Loss function for network optimization.
    loss_fn_fgsm (callable): Loss function for FGSM adversarial attack.
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
    results = prepare_results_dataframe(epsilon, learning_rate_network, loss_fn_network, loss_fn_fgsm, batch_size, epochs, train_color_palette, test_color_palette, training_losses, test_losses, test_accuracies)
    results.to_excel(os.path.join(exp_dir, 'results.xlsx'), index=False)
    plot_loss_graphs(training_losses, test_losses, exp_dir)
    save_color_palette_image(train_color_palette, test_color_palette, exp_dir, show=False)
    save_adversarial_example_image(model, epsilon, exp_dir, train_loader, device, show=False)


def save_model(model, exp_dir):
    """
    Save the model to the specified directory.

    Parameters:
    model (nn.Module): The trained neural network model.
    exp_dir (str): Directory to save the model.
    """
    model_path = os.path.join(exp_dir, 'model.pth')
    torch.save(model.state_dict(), model_path)


def prepare_results_dataframe(epsilon, learning_rate_network, loss_fn_network, loss_fn_fgsm, batch_size, epochs, train_color_palette, test_color_palette, training_losses, test_losses, test_accuracies):
    """
    Prepare a DataFrame with the results of the training.

    Parameters:
    epsilon (float): Perturbation magnitude for FGSM attack.
    learning_rate_network (float): Learning rate for the network optimizer.
    loss_fn_network (callable): Loss function for network optimization.
    loss_fn_fgsm (callable): Loss function for FGSM adversarial attack.
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
        'epsilon': ([epsilon] + [None] * (num_entries - 1))[:num_entries],
        'learning_rate_network': ([learning_rate_network] + [None] * (num_entries - 1))[:num_entries],
        'loss_fn_network': ([loss_fn_network.__name__] + [None] * (num_entries - 1))[:num_entries],
        'loss_fn_fgsm': ([loss_fn_fgsm.__name__] + [None] * (num_entries - 1))[:num_entries],
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


def save_adversarial_example_image(model, epsilon, exp_dir, train_loader, device, show=False):
    """
    Save an image showing original and adversarial examples.

    Parameters:
    model (nn.Module): Trained neural network model.
    epsilon (float): Perturbation magnitude for FGSM attack.
    exp_dir (str): Directory to save the image.
    train_loader (DataLoader): DataLoader for the training dataset.
    device (torch.device): Device used for training.
    show (bool): Whether to show the image or save it.
    """
    original_img = train_loader.dataset[0][0].unsqueeze(0).to(device)
    target = torch.tensor([train_loader.dataset[0][1]]).to(device)
    adversarial_img = fgsm_attack(model, original_img, target, epsilon, F.nll_loss)

    # Detach tensors before passing to the show_adversarial_example_images function
    original_img = original_img.detach()
    adversarial_img = adversarial_img.detach()
    
    adv_img_path = os.path.join(exp_dir, 'adversarial_example.png')
    show_adversarial_example_images(original_img[0].to(device), adversarial_img[0], title="Example: Original vs Adversarial Image", show=show, save_path=adv_img_path if not show else None)


def run_fgsm_experiments(param_grid):
    """
    Run experiments with different parameter combinations. For each combination the training and 
    evaluation process is executed and a folder is created to save the results. Training will run three
    times for each parameter combination to account for randomness in the initialization.
    parameters needed (each one as a list of all possible values):
    - epsilon (float): Perturbation magnitude for FGSM attack.
    - learning_rate_network (float): Learning rate for the network optimizer.
    - loss_fn_network (callable): Loss function for network optimization.
    - loss_fn_fgsm (callable): Loss function for FGSM adversarial attack.
    - batch_size (int): Batch size for training and testing.
    - epochs (int): Number of epochs for training.
    - train_color_palette (list): Color palette for training data: 10x3 list of lists of RGB values. Each inner list contains a RGB
    color mask for a class label.
    - test_color_palette (list): Color palette for test data: same format as train_color_palette.
    - palette_name (str): Name of the color palette.

    Parameters:
    param_grid (dict): Dictionary where keys are parameter names and values are lists of parameter values.
    """
    keys, values = zip(*param_grid.items())
    for value_combination in itertools.product(*values):
        params = dict(zip(keys, value_combination))
        run_single_fgsm_experiment(params)


def run_single_fgsm_experiment(params):
    """
    Run a single FGSM experiment with the given parameters.

    Parameters:
    params (dict): Dictionary of parameters for the experiment.
    """
    exp_dir, device, train_loader, test_loader = setup_experiment(params)
    for exp_num in range(1, 4):
        model, optimizer = setup_model_and_optimizer(params, device)
        exp_subdir = create_experiment_subdirectory(exp_dir, exp_num)
        training_losses, test_losses, test_accuracies = train_and_evaluate(model, device, train_loader, test_loader, optimizer, params)
        # Remove palette_name from params before passing to save_training_results
        filtered_params = {key: value for key, value in params.items() if key != 'palette_name'}
        save_training_results(exp_subdir, **filtered_params, training_losses=training_losses, test_losses=test_losses, test_accuracies=test_accuracies, model=model, train_loader=train_loader, device=device)


def setup_experiment(params):
    """
    Set up the experiment directory and data loaders.

    Parameters:
    params (dict): Dictionary of parameters for the experiment.

    Returns:
    tuple: Experiment directory, device, train_loader, and test_loader.
    """
    exp_dir = f"FGSM_exp_epsilon{params['epsilon']}_lrNet{params['learning_rate_network']}_lossNet{params['loss_fn_network'].__name__}_lossFGSM{params['loss_fn_fgsm'].__name__}_batch{params['batch_size']}_epochs{params['epochs']}_palette{params['palette_name']}"
    os.makedirs(exp_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = get_data_loaders(params)
    return exp_dir, device, train_loader, test_loader


def get_data_loaders(params):
    """
    Get the data loaders for the experiment.

    Parameters:
    params (dict): Dictionary of parameters for the experiment.
    device (torch.device): Device to run the training on (CPU or GPU).

    Returns:
    tuple: train_loader and test_loader.
    """
    train_mnist = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
    test_mnist = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
    train_dataset = ColoredMnistDataset(train_mnist, params['train_color_palette'])
    test_dataset = ColoredMnistDataset(test_mnist, params['test_color_palette'])
    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=False)
    return train_loader, test_loader


def setup_model_and_optimizer(params, device):
    """
    Set up the model and optimizer.

    Parameters:
    params (dict): Dictionary of parameters for the experiment.
    device (torch.device): Device to run the training on (CPU or GPU).

    Returns:
    tuple: Model and optimizer.
    """
    model = ColoredMnistNetwork().to(device)
    optimizer = optim.SGD(model.parameters(), lr=params['learning_rate_network'])
    return model, optimizer


def create_experiment_subdirectory(exp_dir, exp_num):
    """
    Create a subdirectory for an individual experiment.

    Parameters:
    exp_dir (str): Base directory for the experiment.
    exp_num (int): Experiment number.

    Returns:
    str: Path to the subdirectory.
    """
    exp_subdir = os.path.join(exp_dir, f"Exp{exp_num}")
    os.makedirs(exp_subdir, exist_ok=True)
    return exp_subdir


def train_and_evaluate(model, device, train_loader, test_loader, optimizer, params):
    """
    Train and evaluate the model for the given parameters.

    Parameters:
    model (nn.Module): The neural network model.
    device (torch.device): Device to run the training on (CPU or GPU).
    train_loader (DataLoader): DataLoader for the training dataset.
    test_loader (DataLoader): DataLoader for the testing dataset.
    optimizer (torch.optim.Optimizer): Optimizer for updating the model parameters.
    params (dict): Dictionary of parameters for the experiment.

    Returns:
    tuple: Lists of training losses, test losses, and test accuracies.
    """
    training_losses = []
    test_losses = []
    test_accuracies = []
    for epoch in range(1, params['epochs'] + 1):
        epoch_training_losses = train_fgsm(model, device, train_loader, optimizer, epoch, params['epsilon'], params['loss_fn_fgsm'])
        training_losses.extend(epoch_training_losses)
        test_loss, accuracy, epoch_test_losses = evaluate_model(model, device, test_loader, epoch)
        test_losses.extend(epoch_test_losses)
        test_accuracies.append(accuracy)
    return training_losses, test_losses, test_accuracies


def visualize_fgsm_adversarial_example(epsilon, loss_fn_fgsm, train_color_palette):
    """
    Visualize an adversarial example with the given parameters using FGSM.

    Parameters:
    epsilon (float): Perturbation magnitude for FGSM attack.
    learning_rate_network (float): Learning rate for the network optimizer.
    loss_fn_fgsm (callable): Loss function for FGSM adversarial attack.
    train_color_palette (list): Color palette for training data.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ColoredMnistNetwork().to(device)
    model.eval()
    data, target = get_single_data_example(train_color_palette, device)
    adversarial_examples = fgsm_attack(model, data, target, epsilon, loss_fn_fgsm)

    # Detach tensors before passing to the show_adversarial_example_images function
    data = data.detach()
    adversarial_examples = adversarial_examples.detach()
    
    show_adversarial_example_images(data[0].to(device), adversarial_examples[0], title="FGSM Adversarial Example", show=True, save_path=None)


def get_single_data_example(train_color_palette, device):
    """
    Get a single data example from the training dataset.

    Parameters:
    train_color_palette (list): Color palette for training data.
    device (torch.device): Device to run the training on (CPU or GPU).

    Returns:
    tuple: Data and target tensors.
    """
    train_mnist = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
    train_dataset = ColoredMnistDataset(train_mnist, train_color_palette)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    data, target = next(iter(train_loader))
    return data.to(device), target.to(device)


def visualize_color_palettes_fgsm(train_color_palette, test_color_palette):
    """
    Visualize the color palettes for training and testing data using FGSM.

    Parameters:
    train_color_palette (list): Color palette for training data.
    test_color_palette (list): Color palette for test data.
    """
    train_loader, test_loader = get_color_palette_loaders(train_color_palette, test_color_palette)
    show_dataset_examples(train_loader, test_loader, show=True, save_path=None)


def get_color_palette_loaders(train_color_palette, test_color_palette):
    """
    Get data loaders for visualizing color palettes.

    Parameters:
    train_color_palette (list): Color palette for training data.
    test_color_palette (list): Color palette for test data.

    Returns:
    tuple: train_loader and test_loader.
    """
    train_loader = DataLoader(ColoredMnistDataset(datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor()), train_color_palette), batch_size=64, shuffle=True)
    test_loader = DataLoader(ColoredMnistDataset(datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor()), test_color_palette), batch_size=64, shuffle=False)
    return train_loader, test_loader