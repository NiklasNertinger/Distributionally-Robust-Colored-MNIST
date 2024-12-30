import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import os
import pandas as pd
from tqdm import tqdm
import itertools
import numpy as np


class ColoredMnistNetwork(nn.Module):
    """
    Convolutional Neural Network for the Colored MNIST dataset.

    The network consists of two convolutional layers followed by max pooling and ELU activation,
    a dropout layer, and two fully connected layers.

    Methods:
    - forward(x): Forward pass through the network.

    Attributes:
    - conv1: First convolutional layer.
    - conv2: Second convolutional layer.
    - conv2_drop: Dropout layer applied after the second convolutional layer.
    - fc1: First fully connected layer.
    - fc2: Second fully connected layer (output layer).
    """
    def __init__(self):
        super(ColoredMnistNetwork, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(64 * 7 * 7, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.elu(F.max_pool2d(self.conv1(x), 2))
        x = F.elu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 64 * 7 * 7)
        x = F.elu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class ColoredMnistDataset(Dataset):
    """
    Custom Dataset for the Colored MNIST dataset.

    The dataset applies specific color palettes to the grayscale MNIST images to create colored versions.

    Methods:
    - __getitem__(index): Get the colored image and its label at the specified index.
    - __len__(): Get the total number of samples in the dataset.

    Attributes:
    - mnist_dataset: Original MNIST dataset.
    - color_palette: List of color values for each class label.
    """
    def __init__(self, mnist_dataset, color_palette):
        self.mnist_dataset = mnist_dataset
        self.color_palette = color_palette

    def __getitem__(self, index):
        img, label = self.mnist_dataset[index]
        img = img.repeat(3, 1, 1)
        color_mask = torch.tensor(self.color_palette[label], dtype=torch.float32)
        colored_img = img * color_mask.view(3, 1, 1)
        return colored_img, label

    def __len__(self):
        return len(self.mnist_dataset)


def collect_examples(loader, offset):
    """
    Collect examples from the loader for visualization.

    Parameters:
    loader (DataLoader): DataLoader for the dataset.
    offset (int): Offset to apply to the labels for distinguishing train and test examples.

    Returns:
    dict: Dictionary containing example images.
    """
    examples = {}
    for images, labels in loader:
        for i, label in enumerate(labels):
            label = label.item()
            if label not in examples or (label + offset) not in examples:
                examples[label + offset] = images[i]
            if len(examples) >= 20:
                return examples
        if len(examples) >= 20:
            break
    return examples


def show_images(examples, title, save_path=None):
    """
    Display or save a set of images.

    Parameters:
    examples (dict): Dictionary of images to display.
    title (str): Title of the plot.
    save_path (str, optional): Path to save the image. If None, the image will be shown.
    """
    plt.figure(figsize=(20, 4))
    plt.suptitle(title)
    for i in range(10):
        plt.subplot(2, 10, i + 1)
        plt.imshow(np.clip(examples[i].permute(1, 2, 0).numpy(), 0, 1))
        plt.title(f'Train Label: {i}')
        plt.axis('off')
        plt.subplot(2, 10, i + 11)
        plt.imshow(np.clip(examples[i + 10].permute(1, 2, 0).numpy(), 0, 1))
        plt.title(f'Test Label: {i}')
        plt.axis('off')
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()


def show_dataset_examples(train_loader, test_loader, show=True, save_path=None):
    """
    Display example images from the train and test loaders side by side.

    Parameters:
    train_loader (DataLoader): DataLoader for the training dataset.
    test_loader (DataLoader): DataLoader for the testing dataset.
    show (bool): Whether to show the image or save it.
    save_path (str, optional): Path to save the image. If None, the image will be shown instead.
    """
    examples = {}
    examples.update(collect_examples(train_loader, 0))
    examples.update(collect_examples(test_loader, 10))
    show_images(examples, "Train and Test Dataset Examples Side by Side", save_path if not show else None)


def train_model_with_robustness(model, device, train_loader, optimizer, epoch, gamma, epsilon, learning_rate_adversarial, max_steps, adversarial_variance, loss_fn_network, loss_fn_adversarial, log_interval=5):
    """
    Train the model with adversarial robustness considerations.

    Parameters:
    model (nn.Module): The neural network model to be trained.
    device (torch.device): The device to run the training on (CPU or GPU).
    train_loader (DataLoader): DataLoader for the training dataset.
    optimizer (torch.optim.Optimizer): Optimizer for updating the model parameters.
    epoch (int): The current epoch number.
    gamma (float): Regularization parameter for the adversarial training.
    epsilon (float): Threshold for stopping the adversarial maximization process.
    learning_rate_adversarial (float): Learning rate for the adversarial maximizer.
    max_steps (int): Maximum number of steps for adversarial maximization.
    adversarial_variance (float): Variance for the normal distribution to generate noise for adversarial examples.
    loss_fn_network (callable): Loss function for network optimization.
    loss_fn_adversarial (callable): Loss function for adversarial maximization.
    log_interval (int, optional): Interval for logging the training progress. Default is 5.

    Returns:
    list: List of tuples containing epoch, batch index, and training loss.
    """
    model.train()
    training_losses = []
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for batch_idx, (data, target) in enumerate(progress_bar):
        data, target = data.to(device), target.to(device)
        adversarial_examples = find_adversarial_maximizer(model, data, target, gamma, epsilon, learning_rate_adversarial, max_steps, adversarial_variance, loss_fn_adversarial)
        model_loss = update_model(model, optimizer, adversarial_examples, target, loss_fn_network)
        training_losses.append((epoch, batch_idx, model_loss.item()))
        if batch_idx % log_interval == 0:
            progress_bar.set_postfix_str(f"[{batch_idx * len(data)}/{len(train_loader.dataset)}] Train-Loss: {model_loss.item():.6f}")
    return training_losses


def update_model(model, optimizer, adversarial_examples, target, loss_fn_network):
    """
    Update the model parameters using the optimizer.

    Parameters:
    model (nn.Module): The neural network model to be updated.
    optimizer (torch.optim.Optimizer): Optimizer for updating the model parameters.
    adversarial_examples (torch.Tensor): Adversarial examples.
    target (torch.Tensor): True labels for the data.
    loss_fn_network (callable): Loss function for network optimization.

    Returns:
    torch.Tensor: The loss value after the model update.
    """
    optimizer.zero_grad()
    output = model(adversarial_examples)
    model_loss = loss_fn_network(output, target)
    model_loss.backward()
    optimizer.step()
    return model_loss


def find_adversarial_maximizer(model, data, target, gamma, epsilon, learning_rate_adversarial, max_steps, adversarial_variance, loss_fn_adversarial):
    """
    Find the adversarial maximizer for the input data.

    Parameters:
    model (nn.Module): The neural network model.
    data (torch.Tensor): Input data.
    target (torch.Tensor): True labels for the data.
    gamma (float): Regularization parameter for the adversarial training.
    epsilon (float): Threshold for stopping the adversarial maximization process.
    learning_rate_adversarial (float): Learning rate for the adversarial maximizer.
    max_steps (int): Maximum number of steps for adversarial maximization.
    adversarial_variance (float): Variance for the normal distribution to generate noise for adversarial examples.
    loss_fn_adversarial (callable): Loss function for adversarial maximization.

    Returns:
    torch.Tensor: The adversarial examples.
    """
    noise = adversarial_variance * torch.randn_like(data)
    adversarial_examples = (data + noise).detach().requires_grad_(True)
    optimizer = optim.SGD([adversarial_examples], lr=learning_rate_adversarial)
    prev_loss = None

    for i in range(max_steps):
        optimizer.zero_grad()
        output = model(adversarial_examples)
        loss = loss_fn_adversarial(output, target, reduction='mean')
        regularization = gamma * torch.mean(torch.norm(adversarial_examples - data, p=2, dim=[1, 2, 3]))
        adversarial_loss = - loss + regularization
        adversarial_loss.backward()
        optimizer.step()
        if prev_loss is not None and abs(adversarial_loss.item() - prev_loss) < epsilon:
            break
        prev_loss = adversarial_loss.item()

    return adversarial_examples.detach()


def show_adversarial_example_images(original, adversarial, title="Original and Adversarial Images", show=True, save_path=None):
    """
    Display original and adversarial images side by side.

    Parameters:
    original (torch.Tensor): Original input image.
    adversarial (torch.Tensor): Adversarially modified image.
    title (str, optional): Title for the plot. Default is "Original and Adversarial Images".
    show (bool): Whether to show the image or save it.
    save_path (str, optional): Path to save the image. If None, the image will be shown instead.
    """
    difference = torch.abs(adversarial - original)
    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(np.clip(original.permute(1, 2, 0).cpu().numpy(), 0, 1), interpolation='nearest')
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(np.clip(adversarial.permute(1, 2, 0).cpu().numpy(), 0, 1), interpolation='nearest')
    plt.title("Adversarial Image")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(np.clip(difference.permute(1, 2, 0).cpu().numpy(), 0, 1), cmap='hot', interpolation='nearest')
    plt.title("Difference Image")
    plt.axis('off')

    plt.suptitle(title)
    if show:
        plt.show()
    elif save_path:
        plt.savefig(save_path)
    plt.close()


def evaluate_model(model, device, test_loader, epoch):
    """
    Evaluate the model on the test dataset.

    Parameters:
    model (nn.Module): The neural network model to be evaluated.
    device (torch.device): The device to run the evaluation on (CPU or GPU).
    test_loader (DataLoader): DataLoader for the testing dataset.
    epoch (int): The current epoch number.

    Returns:
    tuple: A tuple containing the average test loss, accuracy, and list of test losses per batch.
    """
    model.eval()
    test_loss = 0
    correct_predictions = 0
    test_losses = []
    with torch.inference_mode():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = F.nll_loss(output, target, reduction='mean').item()
            test_loss += loss * data.size(0)
            predictions = output.argmax(dim=1, keepdim=True)
            correct_predictions += predictions.eq(target.view_as(predictions)).sum().item()
            test_losses.append((loss, epoch, batch_idx))
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct_predictions / len(test_loader.dataset)
    print(f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {accuracy:.0f}%\n")
    return test_loss, accuracy, test_losses


def save_training_results(exp_dir, gamma, learning_rate_adversarial, epsilon, learning_rate_network, adversarial_variance, loss_fn_network, loss_fn_adversarial, batch_size, epochs, max_steps, train_color_palette, test_color_palette, training_losses, test_losses, test_accuracies, model, train_loader, device):
    """
    Save the training results and plots to the specified directory.

    Parameters:
    exp_dir (str): Directory to save the results.
    gamma (float): Regularization parameter for the adversarial training.
    learning_rate_adversarial (float): Learning rate for the adversarial maximizer.
    epsilon (float): Threshold for stopping the adversarial maximization process.
    learning_rate_network (float): Learning rate for the network optimizer.
    adversarial_variance (float): Variance for the normal distribution to generate noise for adversarial examples.
    loss_fn_network (callable): Loss function for network optimization.
    loss_fn_adversarial (callable): Loss function for adversarial maximization.
    batch_size (int): Batch size for training and testing.
    epochs (int): Number of epochs for training.
    max_steps (int): Maximum number of steps for adversarial maximization.
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

    # Save model
    save_model(model, exp_dir)

    # Prepare training loss data
    results = prepare_results_dict(
        gamma, learning_rate_adversarial, epsilon, learning_rate_network, adversarial_variance, loss_fn_network, loss_fn_adversarial,
        batch_size, epochs, max_steps, train_color_palette, test_color_palette, training_losses, test_losses, test_accuracies
    )

    save_results_to_excel(results, exp_dir)

    # Save training and test loss plots
    plot_loss_graphs(training_losses, test_losses, exp_dir)
    save_color_palette_image(train_color_palette, test_color_palette, exp_dir, show=False)
    save_adversarial_example_image(model, gamma, epsilon, learning_rate_adversarial, max_steps, adversarial_variance, exp_dir, train_loader, device, loss_fn_adversarial, show=False)


def save_model(model, exp_dir):
    """
    Save the trained model to the specified directory.

    Parameters:
    model (nn.Module): Trained neural network model.
    exp_dir (str): Directory to save the model.
    """
    model_path = os.path.join(exp_dir, 'model.pth')
    torch.save(model.state_dict(), model_path)


def prepare_results_dict(gamma, learning_rate_adversarial, epsilon, learning_rate_network, adversarial_variance, loss_fn_network, loss_fn_adversarial,
                         batch_size, epochs, max_steps, train_color_palette, test_color_palette, training_losses, test_losses, test_accuracies):
    """
    Prepare the results dictionary for saving to Excel.

    Parameters:
    (All parameters required for preparing the results dictionary)

    Returns:
    dict: Dictionary of results.
    """
    num_entries = max(len(training_losses), len(test_losses))
    training_epochs = [epoch for epoch, _, _ in training_losses]
    training_batch_indices = [batch_idx for _, batch_idx, _ in training_losses]
    training_loss_values = [loss for _, _, loss in training_losses]

    test_loss_values, test_epochs, test_batch_indices = zip(*test_losses)
    test_loss_values = list(test_loss_values)
    test_epochs = list(test_epochs)
    test_batch_indices = list(test_batch_indices)

    #Make all entries the same length
    results = {
        'gamma': ([gamma] + [None] * (num_entries - 1))[:num_entries],
        'learning_rate_adversarial': ([learning_rate_adversarial] + [None] * (num_entries - 1))[:num_entries],
        'epsilon': ([epsilon] + [None] * (num_entries - 1))[:num_entries],
        'learning_rate_network': ([learning_rate_network] + [None] * (num_entries - 1))[:num_entries],
        'adversarial_variance': ([adversarial_variance] + [None] * (num_entries - 1))[:num_entries],
        'loss_fn_network': ([loss_fn_network.__name__] + [None] * (num_entries - 1))[:num_entries],
        'loss_fn_adversarial': ([loss_fn_adversarial.__name__] + [None] * (num_entries - 1))[:num_entries],
        'batch_size': ([batch_size] + [None] * (num_entries - 1))[:num_entries],
        'epochs': ([epochs] + [None] * (num_entries - 1))[:num_entries],
        'max_steps': ([max_steps] + [None] * (num_entries - 1))[:num_entries],
        'train_color_palette': ([train_color_palette] + [None] * (num_entries - 1))[:num_entries],
        'test_color_palette': ([test_color_palette] + [None] * (num_entries - 1))[:num_entries],
        'training_epochs': (training_epochs + [None] * (num_entries - len(training_epochs)))[:num_entries],
        'training_batch_indices': (training_batch_indices + [None] * (num_entries - len(training_batch_indices)))[:num_entries],
        'training_loss_values': (training_loss_values + [None] * (num_entries - len(training_loss_values)))[:num_entries],
        'test_epochs': (test_epochs + [None] * (num_entries - len(test_epochs)))[:num_entries],
        'test_batch_indices': (test_batch_indices + [None] * (num_entries - len(test_batch_indices)))[:num_entries],
        'test_loss_values': (test_loss_values + [None] * (num_entries - len(test_loss_values)))[:num_entries],
        'test_accuracies': (test_accuracies + [None] * (num_entries - len(test_accuracies)))[:num_entries]
    }

    return results


def save_results_to_excel(results, exp_dir):
    """
    Save the results dictionary to an Excel file.

    Parameters:
    results (dict): Dictionary of results.
    exp_dir (str): Directory to save the Excel file.
    """
    df = pd.DataFrame(results)
    results_path = os.path.join(exp_dir, 'results.xlsx')
    df.to_excel(results_path, index=False)


def plot_loss_graphs(training_losses, test_losses, exp_dir, show=False):
    """
    Plot and save the training and test loss graphs.

    Parameters:
    training_losses (list): List of training loss values.
    test_losses (list): List of test loss values.
    exp_dir (str): Directory to save the plot.
    show (bool): Whether to show the image or save it.
    """
    # Calculate average loss per epoch
    epochs = list(set([epoch for epoch, _, _ in training_losses]))
    avg_training_losses = [np.mean([loss for e, _, loss in training_losses if e == epoch]) for epoch in epochs]
    avg_test_losses = [np.mean([loss for loss, e, _ in test_losses if e == epoch]) for epoch in epochs]

    plt.figure()
    plt.plot(epochs, avg_training_losses, label='Training Loss')
    plt.plot(epochs, avg_test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss')
    plt.legend()
    plot_path = os.path.join(exp_dir, 'loss_plot.png')
    if show:
        plt.show()
    else:
        plt.savefig(plot_path)
    plt.close()


def save_color_palette_image(train_color_palette, test_color_palette, exp_dir, show=False):
    """
    Save the color palette image.

    Parameters:
    train_color_palette (list): Color palette for training data.
    test_color_palette (list): Color palette for test data.
    exp_dir (str): Directory to save the image.
    show (bool): Whether to show the image or save it.
    """
    train_loader = DataLoader(ColoredMnistDataset(datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor()), train_color_palette), batch_size=64, shuffle=True)
    test_loader = DataLoader(ColoredMnistDataset(datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor()), test_color_palette), batch_size=64, shuffle=False)
    palette_path = os.path.join(exp_dir, 'color_palette.png')
    show_dataset_examples(train_loader, test_loader, show=show, save_path=palette_path if not show else None)


def save_adversarial_example_image(model, gamma, epsilon, learning_rate_adversarial, max_steps, adversarial_variance, exp_dir, train_loader, device, loss_fn_adversarial, show=False):
    """
    Save an image showing original and adversarial examples.

    Parameters:
    model (nn.Module): Trained neural network model.
    gamma (float): Regularization parameter for the adversarial training.
    epsilon (float): Threshold for stopping the adversarial maximization process.
    learning_rate_adversarial (float): Learning rate for the adversarial maximizer.
    max_steps (int): Maximum number of steps for adversarial maximization.
    adversarial_variance (float): Variance for the normal distribution to generate noise for adversarial examples.
    exp_dir (str): Directory to save the image.
    train_loader (DataLoader): DataLoader for the training dataset.
    device (torch.device): Device used for training.
    loss_fn_adversarial (callable): Loss function for adversarial maximization.
    show (bool): Whether to show the image or save it.
    """
    original_img = train_loader.dataset[0][0]
    adversarial_img = find_adversarial_maximizer(
        model, original_img.unsqueeze(0).to(device), 
        torch.tensor([train_loader.dataset[0][1]]).to(device), 
        gamma, epsilon, learning_rate_adversarial, max_steps, adversarial_variance, loss_fn_adversarial
    )
    adv_img_path = os.path.join(exp_dir, 'adversarial_example.png')
    show_adversarial_example_images(original_img.to(device), adversarial_img[0], title="Example: Original vs Adversarial Image", show=show, save_path=adv_img_path if not show else None)


def run_experiments(param_grid):
    """
    Run experiments with different parameter combinations. For each combination the training and 
    evaluation process is executed and a folder is created to save a visualization of an adversarial example,
    a visualization of the color palette, a loss plot, the final model, and an excel file containing information
    about the training run and parameters. Training will run three
    times for each parameter combination to account for randomness in the initialization.
    parameters needed (each one as a list of all possible values, experiments will run for all combinations):
    - gamma (float): Regularization parameter for the adversarial training.
    - learning_rate_adversarial (float): Learning rate for the adversarial maximizer.
    - epsilon (float): Threshold for stopping the adversarial maximization process.
    - learning_rate_network (float): Learning rate for the network optimizer.
    - adversarial_variance (float): Variance for the normal distribution to generate noise for adversarial examples.
    - loss_fn_network (callable): Loss function for network optimization.
    - loss_fn_adversarial (callable): Loss function for adversarial maximization.
    - batch_size (int): Batch size for training and testing.
    - epochs (int): Number of epochs for training.
    - max_steps (int): Maximum number of steps for adversarial maximization.
    - train_color_palette (list): Color palette for training data: 10x3 list of lists of RGB values. Each inner list contains a RGB
    color mask for a class label.
    - test_color_palette (list): Color palette for test data: Same format as above
    - palette_name (str): Self-chosen name of the color palette for saving the results.


    Parameters:
    param_grid (dict): Dictionary where keys are parameter names and values are lists of parameter values.
    """
    keys, values = zip(*param_grid.items())
    for value_combination in itertools.product(*values):
        params = dict(zip(keys, value_combination))
        run_single_experiment(params)


def run_single_experiment(params):
    """
    Run a single experiment with the given parameters.

    Parameters:
    params (dict): Dictionary of parameters for the experiment.
    """
    gamma = params['gamma']
    learning_rate_adversarial = params['learning_rate_adversarial']
    epsilon = params['epsilon']
    learning_rate_network = params['learning_rate_network']
    adversarial_variance = params['adversarial_variance']
    loss_fn_network = params['loss_fn_network']
    loss_fn_adversarial = params['loss_fn_adversarial']
    batch_size = params['batch_size']
    epochs = params['epochs']
    max_steps = params['max_steps']
    train_color_palette = params['train_color_palette']
    test_color_palette = params['test_color_palette']
    palette_name = params['palette_name']

    exp_dir = f"Dist_Robust_exp_gamma{gamma}_lrAdv{learning_rate_adversarial}_lrNet{learning_rate_network}_var{adversarial_variance}_lossNet{loss_fn_network.__name__}_lossAdv{loss_fn_adversarial.__name__}_batch{batch_size}_epochs{epochs}_steps{max_steps}_palette{palette_name}"
    os.makedirs(exp_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_loader, test_loader = create_dataloaders(train_color_palette, test_color_palette, batch_size)
    
    for exp_num in range(1, 4):
        model = ColoredMnistNetwork().to(device)
        optimizer = optim.SGD(model.parameters(), lr=learning_rate_network)
        exp_subdir = os.path.join(exp_dir, f"Exp{exp_num}")
        os.makedirs(exp_subdir, exist_ok=True)

        training_losses, test_losses, test_accuracies = train_and_evaluate(
            model, device, train_loader, test_loader, optimizer, gamma, epsilon, learning_rate_adversarial, max_steps, adversarial_variance, epochs,
            loss_fn_network, loss_fn_adversarial
        )

        save_training_results(
            exp_subdir, gamma, learning_rate_adversarial, epsilon, learning_rate_network, adversarial_variance, loss_fn_network, loss_fn_adversarial,
            batch_size, epochs, max_steps, train_color_palette, test_color_palette, training_losses, test_losses,
            test_accuracies, model, train_loader, device
        )


def create_dataloaders(train_color_palette, test_color_palette, batch_size):
    """
    Create dataloaders for training and testing datasets.

    Parameters:
    train_color_palette (list): Color palette for training data.
    test_color_palette (list): Color palette for test data.
    batch_size (int): Batch size for training and testing.

    Returns:
    tuple: Tuple containing training and testing dataloaders.
    """
    train_mnist = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
    test_mnist = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())

    train_dataset = ColoredMnistDataset(train_mnist, train_color_palette)
    test_dataset = ColoredMnistDataset(test_mnist, test_color_palette)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def train_and_evaluate(model, device, train_loader, test_loader, optimizer, gamma, epsilon, learning_rate_adversarial, max_steps, adversarial_variance, epochs, loss_fn_network, loss_fn_adversarial):
    """
    Train and evaluate the model.

    Parameters:
    (All parameters required for training and evaluation)

    Returns:
    tuple: Tuple containing training losses, test losses, and test accuracies.
    """
    training_losses = []
    test_losses = []
    test_accuracies = []

    for epoch in range(1, epochs + 1):
        epoch_training_losses = train_model_with_robustness(
            model, device, train_loader, optimizer, epoch, gamma, epsilon, learning_rate_adversarial, max_steps, adversarial_variance,
            loss_fn_network, loss_fn_adversarial
        )
        training_losses.extend(epoch_training_losses)
        test_loss, accuracy, epoch_test_losses = evaluate_model(model, device, test_loader, epoch)
        test_losses.extend(epoch_test_losses)
        test_accuracies.append(accuracy)

    return training_losses, test_losses, test_accuracies


def visualize_adversarial_example(gamma, learning_rate_adversarial, epsilon, loss_fn_adversarial, train_color_palette, adversarial_variance, max_steps=40):
    """
    Visualize an adversarial example with the given parameters.

    Parameters:
    gamma (float): Regularization parameter for the adversarial training.
    learning_rate_adversarial (float): Learning rate for the adversarial maximizer.
    epsilon (float): Threshold for stopping the adversarial maximization process.
    loss_fn_adversarial (callable): Loss function for adversarial maximization.
    train_color_palette (list): Color palette for training data.
    adversarial_variance (float): Variance for the normal distribution to generate noise for adversarial examples.
    max_steps (int): Maximum number of steps for adversarial maximization. Default is 40.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ColoredMnistNetwork().to(device)
    model.eval()

    train_loader = create_single_loader(train_color_palette)

    data, target = next(iter(train_loader))
    data, target = data.to(device), target.to(device)
    adversarial_examples = find_adversarial_maximizer(model, data, target, gamma, epsilon, learning_rate_adversarial, max_steps, adversarial_variance, loss_fn_adversarial)
    
    show_adversarial_example_images(data[0].to(device), adversarial_examples[0], title="Adversarial Example", show=True, save_path=None)


def create_single_loader(color_palette):
    """
    Create a DataLoader for a single example.

    Parameters:
    color_palette (list): Color palette for the dataset.

    Returns:
    DataLoader: DataLoader for a single example.
    """
    mnist_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
    dataset = ColoredMnistDataset(mnist_dataset, color_palette)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    return loader


def visualize_color_palettes(train_color_palette, test_color_palette):
    """
    Visualize the color palettes for training and testing data.

    Parameters:
    train_color_palette (list): Color palette for training data.
    test_color_palette (list): Color palette for test data.
    """
    train_loader = DataLoader(ColoredMnistDataset(datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor()), train_color_palette), batch_size=64, shuffle=True)
    test_loader = DataLoader(ColoredMnistDataset(datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor()), test_color_palette), batch_size=64, shuffle=False)
    show_dataset_examples(train_loader, test_loader, show=True, save_path=None)
