# **Distributionally Robust Colored MNIST**

This repository implements a **Distributionally Robust Optimization (DRO)** algorithm to enhance adversarial robustness, applying it to a custom **Colored MNIST dataset**. The code is designed for evaluation of its performance against the **Fast Gradient Sign Method (FGSM)** and **standard training (no adversarial defense).**

## **Table of Contents**
1. [Project Overview](#1-project-overview)
2. [Core Features](#2-core-features)
3. [Key Functions](#3-key-functions)
4. [Visualizations](#4-visualizations)

---

## **1. Project Overview**

Deep learning models are often vulnerable to adversarial attacks or spurious correlations. This project tackles this challenge by:
- Implementing **DRO training**, inspired by [*Certifying Some Distributional Robustness with Principled Adversarial Training*](https://arxiv.org/abs/1710.10571) by Sinha et al.
- Applying DRO on a **Colored MNIST dataset**, where colors are spuriously correlated with digit labels.
- Offering tools to compare **DRO**, **FGSM adversarial training**, and **standard training**.

### **Note on the DRO Implementation**
The DRO algorithm implemented in this project differs from the algorithm presented in the paper by Sinha et al. The original algorithm would often fail to "move away" from the starting example when attempting to create an adversarial example using the modified loss function. 

To address this:
- The starting point for the Stochastic Gradient Descent (SGD) that maximizes the modified loss is sampled from a Gaussian distribution with:
  - A **mean** equal to the original image.
  - A **variance** controlled by the parameter `adversarial_variance`.
- The optimization of the modified loss runs until:
  - The difference between steps in the modified loss is smaller than the parameter `epsilon`, or
  - A maximum number of steps (`max_steps`) is reached.

---

## **2. Core Features**

- **Custom Dataset**: Creates a Colored MNIST dataset with spurious correlations provided color palettes for the test and training set.
- **DRO Training**: Implements a distributionally robust training algorithm to defend against adversarial perturbations.
- **FGSM Adversarial Training**: Implements FGSM to test its robustness against adversarial attacks.
- **Standard Training**: Provides a baseline without adversarial defenses.
- **Visualization Tools**: Includes functions for visualizing adversarial examples and dataset characteristics.

---

## **3. Key Functions**

### 1. `run_experiments`
- **Purpose**: Runs the **Distributionally Robust Optimization (DRO)** training algorithm on the Colored MNIST dataset.
- **Details**:
  - Executes the training process **three times for each parameter combination** to account for randomness in the model initialization.
  - Saves all outputs described in the [What is Saved](#what-is-saved) section.
- **Output**: Trained models using DRO, saved data for each training run, and performance metrics for analysis.

### 2. `visualize_adversarial_example`
- **Purpose**: Visualizes **adversarial examples** generated using a model trained with DRO.
- **Details**:
  - Displays the original image, the adversarially perturbed image, and the difference between them.
  - Useful for understanding the effect of adversarial perturbations on models trained with DRO.
- **Output**: Visualization of adversarial examples.

### 3. `visualize_color_palettes`
- **Purpose**: Visualizes the **color palettes** used in the training and testing datasets.
- **Details**:
  - Provides a side-by-side comparison of the colors assigned to digit labels in the training and test datasets.
- **Output**: A visualization showing how spurious correlations are introduced through color labeling.

### 4. `run_fgsm_experiments`
- **Purpose**: Runs the **Fast Gradient Sign Method (FGSM)** adversarial training algorithm on the Colored MNIST dataset.
- **Details**:
  - Executes FGSM training **three times for each parameter combination** to account for randomness in initialization.
  - Saves all outputs described in the [What is Saved](#what-is-saved) section, including adversarial example visualizations.
- **Output**: Trained models using FGSM, saved data for each training run, and performance metrics for analysis.

### 5. `visualize_fgsm_adversarial_example`
- **Purpose**: Visualizes **adversarial examples** generated using a model trained with FGSM.
- **Details**:
  - Displays the original image, the adversarially perturbed image, and the difference between them.
  - Useful for comparing the effects of adversarial training on the model's robustness.
- **Output**: Visualization of adversarial examples.

### 6. `run_standard_experiments`
- **Purpose**: Runs the **standard training** algorithm (without adversarial defense) on the Colored MNIST dataset.
- **Details**:
  - Executes standard training **three times for each parameter combination** to account for randomness in initialization.
  - Saves all outputs described in the [What is Saved](#what-is-saved) section, except for adversarial example visualizations.
- **Output**: Trained models without adversarial defenses, saved data for each training run, and performance metrics for analysis.

---

### **What is Saved**

For each training run (DRO, FGSM, and standard training), the following files and artifacts are saved for every parameter combination and each of the three runs:

- **Color Palette Visualization**: Images showcasing the color palette used for training and testing.
- **Loss Plot**: A graph of the training and test loss over training epochs.
- **Final Model**: The trained model saved as a file.
- **Details Spreadsheet**: An Excel file summarizing parameters and metrics of the training run.

#### Additional Saved Outputs
- For **DRO and FGSM**: Adversarial example visualizations (images showing original, adversarial, and difference examples).
- For **Standard Training**: No adversarial example visualizations are saved.

---

## **4. Visualizations**

### **Adversarial Example: DRO**
This image visualizes an adversarial example generated using the **Distributionally Robust Optimization (DRO)** training algorithm. Similar to FGSM, it displays the original image, the adversarially perturbed image, and the difference between the two.

#### Parameters:
- `gamma = 0.3`
- `learning_rate_adversarial = 0.1`
- `epsilon = 0.0001`
- `loss_fn_adversarial = nll_loss`
- `adversarial_variance = 0.2`
- `max_steps = 100`

![DRO Adversarial Example](images/adversarial_example_dro.png)

---

### **Adversarial Example: FGSM**
The following image demonstrates the effect of the **Fast Gradient Sign Method (FGSM)** on the Colored MNIST dataset. The original image, the adversarially perturbed image, and the difference between the two are shown side-by-side.

#### Parameters:
- `epsilon = 0.2`

![FGSM Adversarial Example](images/adversarial_example_fgsm.png)

---

### **Colored MNIST Dataset Visualization**
Below is a visualization of a possible training and testing dataset. Colors are spuriously correlated with digit labels to introduce a challenging dataset for robustness evaluation. Users can specify custom **color palettes** for both the training and test datasets, allowing flexibility in how spurious correlations are introduced.

#### Parameters:
- **Train Color Palette**:
  ```python
  train_color_palette = [
      [0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.5], 
      [0.5, 0.0, 0.5], [1.0, 0.5, 0.0], [1.0, 1.0, 0.0], 
      [0.6, 0.4, 0.2], [1.0, 0.6, 0.6], [0.0, 1.0, 1.0], 
      [1.0, 0.0, 1.0]
  ]
- **Test Color Palette**:
   ```python
  test_color_palette = [
      [0.5, 0.0, 0.5], [1.0, 0.5, 0.0], [0.0, 0.8, 0.0], 
      [1.0, 0.0, 1.0], [1.0, 1.0, 0.0], [0.5, 1.0, 0.5], 
      [1.0, 1.0, 0.0], [1.0, 0.7, 0.8], [0.0, 0.0, 1.0], 
      [1.0, 0.0, 0.0]
  ]

![Dataset Visualization](images/color_palette.png)

