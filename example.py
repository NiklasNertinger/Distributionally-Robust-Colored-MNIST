import torch.nn.functional as F
from dist_robust_training import run_experiments, visualize_adversarial_example, visualize_color_palettes
from fgsm_training import run_fgsm_experiments, visualize_fgsm_adversarial_example
from standard_training import run_standard_experiments


# This file is an example of what relevant functions are included in this project and how they are to be used.
# There are six main functions:
#  - run_experiments: This function runs the distributionally robust training algorithm on the color palette dataset.
#  - visualize_adversarial_example: This function visualizes an adversarial example generated using the distributionally robust training algorithm.
#  - visualize_color_palettes: This function visualizes the color palettes used in the training and testing sets.
#  - run_fgsm_experiments: This function runs the fast gradient sign method on the color palette dataset.
#  - visualize_fgsm_adversarial_example: This function visualizes an adversarial example generated using the fast gradient sign method.
#  - run_standard_experiments: This function runs the standard training algorithm on the color palette dataset.


#------------------------------ Fixed Parameters ------------------------------


loss_fn_network = F.nll_loss
batch_size = 64
learning_rate_network = 0.01
epochs = 10
loss_fn_adversarial = F.nll_loss
epsilon_dist_robust = 0.0001
loss_fn_fgsm = F.nll_loss
train_color_palette = [[0.08, 0.0, 0.6], [0.7, 0.24, 0.0], [0.35, 0.8, 0.23], [0.62, 0.13, 0.4], [0.8, 0.55, 0.0], [0.73, 0.89, 0.4], [0.68, 0.3, 0.12], [1.0, 0.45, 0.37], [0.05, 0.6, 0.6], [0.6, 0.2, 0.63]]
test_color_palette = [[0.0, 0.12, 0.5], [0.67, 0.19, 0.06], [0.3, 0.7, 0.51], [0.49, 0.11, 0.58], [0.72, 0.6, 0.05], [0.69, 0.9, 0.23], [0.54, 0.37, 0.22], [0.77, 0.52, 0.6], [0.1, 0.58, 0.71], [0.66, 0.0, 0.6]]
palette_name = 'SimilarColors'


#------------------------------ Distributionally Robust Training ------------------------------


gamma = 0.3
learning_rate_adversarial = 0.1
epsilon_dist_robust = 0.0001
max_steps = 100
adversarial_variance = 0.2

param_grid_dist_robust = {
'gamma': [gamma],
'learning_rate_adversarial': [learning_rate_adversarial],
'epsilon': [epsilon_dist_robust],
'learning_rate_network': [learning_rate_network],
'loss_fn_network': [loss_fn_network],
'loss_fn_adversarial': [loss_fn_adversarial],
'batch_size': [batch_size],
'epochs': [epochs],
'max_steps': [max_steps],
'adversarial_variance': [adversarial_variance],
'train_color_palette': [train_color_palette],
'test_color_palette': [test_color_palette],
'palette_name': [palette_name]
}

# We visualize the color palettes, showing what color is assigned to what label in each of the test and training set.
visualize_color_palettes(train_color_palette, test_color_palette)
# We visualize an adversial example for the distributionally robust training implemented in this project.
visualize_adversarial_example(gamma, learning_rate_adversarial, epsilon_dist_robust, loss_fn_adversarial, train_color_palette, adversarial_variance, max_steps)
run_experiments(param_grid_dist_robust)


#------------------------------ Fast Gradient Sign Method ------------------------------


epsilon_fgsm = 0.2

param_grid_fgsm = {
    'epsilon': [epsilon_fgsm],
    'learning_rate_network': [learning_rate_network],
    'loss_fn_network': [loss_fn_network],
    'loss_fn_fgsm': [loss_fn_fgsm],
    'batch_size': [batch_size],
    'epochs': [epochs],
    'train_color_palette': [train_color_palette],
    'test_color_palette': [test_color_palette],
    'palette_name': [palette_name]
}

#We visualize an adversarial example using standard FGSM.
visualize_fgsm_adversarial_example(epsilon_fgsm, loss_fn_fgsm, train_color_palette)
run_fgsm_experiments(param_grid_fgsm)


#------------------------------ Standard Training ------------------------------


param_grid_standard = {
    'learning_rate_network': [learning_rate_network],
    'loss_fn_network': [loss_fn_network],
    'batch_size': [batch_size],
    'epochs': [epochs],
    'train_color_palette': [train_color_palette],
    'test_color_palette': [test_color_palette],
    'palette_name': [palette_name]
}

run_standard_experiments(param_grid_standard)