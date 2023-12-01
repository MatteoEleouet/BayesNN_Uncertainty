import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Function Definitions


def nonlinearity(x):
    return torch.relu(x)


def log_gaussian(x, mu, sigma):
    sigma_tensor = torch.tensor(sigma, dtype=x.dtype, device=x.device)
    return -0.5 * np.log(2 * np.pi) - torch.log(torch.abs(sigma_tensor)) - (x - mu) ** 2 / (2 * sigma_tensor ** 2)


def log_gaussian_logsigma(x, mu, logsigma):
    return -0.5 * np.log(2 * np.pi) - logsigma - (x - mu) ** 2 / (2. * torch.exp(logsigma) ** 2.)


def get_random(shape, avg, std, device):
    return torch.randn(shape, device=device) * std + avg


def log_categ(y, y_hat):
    ll = 1e-8
    y_hat = y_hat.clamp(min=ll, max=1)
    return torch.sum(y * torch.log(y_hat), axis=1)

# Bayesian Neural Network Class


class BayesianNeuralNetwork(nn.Module):
    def __init__(self, n_input, n_hidden_1, n_hidden_2, n_output):
        super(BayesianNeuralNetwork, self).__init__()
        self.W1_mu = nn.Parameter(torch.randn(n_input, n_hidden_1) * 0.1)
        self.W1_logsigma = nn.Parameter(torch.randn(n_input, n_hidden_1) * 0.1)
        self.b1_mu = nn.Parameter(torch.zeros(n_hidden_1))
        self.b1_logsigma = nn.Parameter(torch.zeros(n_hidden_1))
        self.W2_mu = nn.Parameter(torch.randn(n_hidden_1, n_hidden_2) * 0.1)
        self.W2_logsigma = nn.Parameter(
            torch.randn(n_hidden_1, n_hidden_2) * 0.1)
        self.b2_mu = nn.Parameter(torch.zeros(n_hidden_2))
        self.b2_logsigma = nn.Parameter(torch.zeros(n_hidden_2))
        self.W3_mu = nn.Parameter(torch.randn(n_hidden_2, n_output) * 0.1)
        self.W3_logsigma = nn.Parameter(
            torch.randn(n_hidden_2, n_output) * 0.1)
        self.b3_mu = nn.Parameter(torch.zeros(n_output))
        self.b3_logsigma = nn.Parameter(torch.zeros(n_output))

    def forward(self, x, epsilon_prior):
        epsilon_w1 = get_random(self.W1_mu.size(), avg=0.,
                                std=epsilon_prior, device=x.device)
        epsilon_b1 = get_random(self.b1_mu.size(), avg=0.,
                                std=epsilon_prior, device=x.device)
        W1 = self.W1_mu + torch.log1p(torch.exp(self.W1_logsigma)) * epsilon_w1
        b1 = self.b1_mu + torch.log1p(torch.exp(self.b1_logsigma)) * epsilon_b1

        epsilon_w2 = get_random(self.W2_mu.size(), avg=0.,
                                std=epsilon_prior, device=x.device)
        epsilon_b2 = get_random(self.b2_mu.size(), avg=0.,
                                std=epsilon_prior, device=x.device)
        W2 = self.W2_mu + torch.log1p(torch.exp(self.W2_logsigma)) * epsilon_w2
        b2 = self.b2_mu + torch.log1p(torch.exp(self.b2_logsigma)) * epsilon_b2

        epsilon_w3 = get_random(self.W3_mu.size(), avg=0.,
                                std=epsilon_prior, device=x.device)
        epsilon_b3 = get_random(self.b3_mu.size(), avg=0.,
                                std=epsilon_prior, device=x.device)
        W3 = self.W3_mu + torch.log1p(torch.exp(self.W3_logsigma)) * epsilon_w3
        b3 = self.b3_mu + torch.log1p(torch.exp(self.b3_logsigma)) * epsilon_b3

        a1 = nonlinearity(torch.matmul(x, W1) + b1)
        a2 = nonlinearity(torch.matmul(a1, W2) + b2)
        h = torch.softmax(nonlinearity(torch.matmul(a2, W3) + b3), dim=1)

        return h, W1, W2, W3, b1, b2, b3

# Function to calculate the entropy of predictions
def predictive_entropy(y_hat):
    return -torch.sum(y_hat * torch.log(y_hat + 1e-8), axis=1)

# Function to visualize weight distributions
def plot_weight_distributions(weights, epoch):
    fig, axs = plt.subplots(1, len(weights), figsize=(15, 3))
    for i, weight in enumerate(weights):
        axs[i].hist(weight.detach().cpu().numpy().flatten(), bins=50, alpha=0.7)
        axs[i].set_title(f'Layer {i+1} Weight Distribution at Epoch {epoch+1}')
    plt.tight_layout()
    plt.show()


# Main Script
if __name__ == '__main__':
    # Load MNIST dataset
    mnist = fetch_openml('mnist_784', version=1)
    data = np.float32(mnist.data) / 255
    target = np.int32(mnist.target).reshape(-1, 1)

    # Sample a subset of data
    N = 30000
    idx = np.random.choice(data.shape[0], N)
    data = data[idx]
    target = target[idx]

    # Split into training and testing sets
    train_idx, test_idx = train_test_split(np.array(range(N)), test_size=0.05)
    train_data = torch.tensor(data[train_idx], dtype=torch.float32).to("cuda")
    test_data = torch.tensor(data[test_idx], dtype=torch.float32).to("cuda")
    train_target = torch.tensor(OneHotEncoder(sparse=False).fit_transform(
        target[train_idx]), dtype=torch.float32).to("cuda")
    test_target = torch.tensor(target[test_idx], dtype=torch.long).to("cuda")

    # Define model parameters
    n_input = train_data.shape[1]
    n_hidden_1 = 1024
    n_hidden_2 = 1024
    n_output = 10
    sigma_prior = 1.0
    epsilon_prior = torch.tensor(0.001).to("cuda")
    n_samples = 1
    learning_rate = 0.001
    n_epochs = 10
    batch_size = 100

    # Initialize the model and move it to the GPU
    model = BayesianNeuralNetwork(
        n_input, n_hidden_1, n_hidden_2, n_output).to("cuda")
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    validation_accuracies = []
    weight_distributions = {i: [] for i in range(1, 4)}  # Assuming three layers for simplicity

    # Training loop
    for epoch in range(n_epochs):
        for i in range(0, len(train_data), batch_size):
            x_batch = train_data[i:i + batch_size]
            y_batch = train_target[i:i + batch_size]

            optimizer.zero_grad()
            log_pw, log_qw, log_likelihood = 0., 0., 0.
            for _ in range(n_samples):
                h, W1, W2, W3, b1, b2, b3 = model(x_batch, epsilon_prior)

                sample_log_pw, sample_log_qw, sample_log_likelihood = 0., 0., 0.
                for W, b, W_mu, W_logsigma, b_mu, b_logsigma in [
                    (W1, b1, model.W1_mu, model.W1_logsigma,
                     model.b1_mu, model.b1_logsigma),
                    (W2, b2, model.W2_mu, model.W2_logsigma,
                     model.b2_mu, model.b2_logsigma),
                    (W3, b3, model.W3_mu, model.W3_logsigma,
                     model.b3_mu, model.b3_logsigma)
                ]:
                    sample_log_pw += torch.sum(log_gaussian(W,
                                               0., sigma_prior))
                    sample_log_pw += torch.sum(log_gaussian(b,
                                               0., sigma_prior))
                    sample_log_qw += torch.sum(log_gaussian_logsigma(W,
                                               W_mu, W_logsigma * 2))
                    sample_log_qw += torch.sum(log_gaussian_logsigma(b,
                                               b_mu, b_logsigma * 2))

                sample_log_likelihood = torch.sum(log_categ(y_batch, h))

                log_pw += sample_log_pw
                log_qw += sample_log_qw
                log_likelihood += sample_log_likelihood

            log_qw /= n_samples
            log_pw /= n_samples
            log_likelihood /= n_samples

            # Define the scaling factor 'pi'
            pi = 1. / (N / float(batch_size))

            # Objective function
            objective = torch.sum(pi * (log_qw - log_pw)) - \
                log_likelihood / float(batch_size)
            objective.backward()
            optimizer.step()

        with torch.no_grad():
            correct = 0
            for i in range(0, len(test_data), batch_size):
                x_test_batch = test_data[i:i + batch_size]
                y_test_batch = test_target[i:i + batch_size]
                h, _, _, _, _, _, _ = model(x_test_batch, epsilon_prior)
                pred = torch.argmax(h, axis=1)
                correct += (pred == y_test_batch.flatten()).sum().item()
            acc = correct / float(test_data.shape[0])
            print(f'Epoch {epoch+1}, Accuracy: {acc}')

    # Visualization of uncertainty and interpretability
    # Prediction Confidence Histogram
    confidence_scores = []
    for i in range(len(test_data)):
        h, _, _, _, _, _, _ = model(test_data[i].unsqueeze(0), epsilon_prior)
        confidence, predicted_class = torch.max(h, 1)
        confidence_scores.append(confidence.item())

    plt.figure()
    plt.hist(confidence_scores, bins=20, alpha=0.7,
             color='blue', edgecolor='black')
    plt.title('Histogram of Prediction Confidence Scores')
    plt.xlabel('Confidence Score')
    plt.ylabel
    plt.ylabel('Frequency')
    plt.show()

    # Uncertainty vs. Error Plot
    uncertainties = []
    errors = []
    for i in range(len(test_data)):
        h, _, _, _, _, _, _ = model(test_data[i].unsqueeze(0), epsilon_prior)
        predicted_class = torch.argmax(h, 1)
        error = (predicted_class != test_target[i]).item()
        uncertainty = torch.std(h).item()
        uncertainties.append(uncertainty)
        errors.append(error)

    plt.figure()
    plt.scatter(uncertainties, errors, alpha=0.7, color='red')
    plt.title('Uncertainty vs. Error')
    plt.xlabel('Uncertainty (Standard Deviation)')
    plt.ylabel('Error (0 = Correct, 1 = Incorrect)')
    plt.show()

    # Weight Distributions Visualization
    plt.figure(figsize=(12, 4))
    layer_index = 1
    for param in model.parameters():
        if len(param.size()) > 1:  # Ensure we only plot weights, not biases
            plt.subplot(1, 6, layer_index)
            plt.hist(param.detach().cpu().numpy().flatten(), bins=30,
                     alpha=0.7, color='green', edgecolor='black')
            plt.title(f'Layer {layer_index} Weights')
            layer_index += 1
    plt.tight_layout()
    plt.show()

    # Feature Importance for 5 Samples
    num_samples = 5
    sample_indices = np.random.choice(len(test_data), num_samples, replace=False)
    fig = plt.figure(figsize=(9, 2 * num_samples))  # Adjust the figure size to accommodate three columns
    gs = gridspec.GridSpec(num_samples, 3, width_ratios=[1, 1, 1])  # Now we have three columns in GridSpec

    for i, idx in enumerate(sample_indices):
        sample_data = test_data[idx]
        sample_data.requires_grad_(True)

        # Obtain model output
        output = model(sample_data.unsqueeze(0), epsilon_prior)[0]
        model.zero_grad()
        output.max().backward()

        # Get gradients for input_grad
        input_grad = sample_data.grad.detach().cpu().numpy().reshape(28, 28)
        heatmap = np.abs(input_grad)

        # Original image
        ax1 = plt.subplot(gs[i, 0])
        ax1.imshow(sample_data.detach().cpu().numpy().reshape(28, 28), cmap='gray')
        ax1.set_title('Actual Image')
        ax1.axis('off')

        # Heatmap of feature importance
        ax2 = plt.subplot(gs[i, 1])
        ax2.imshow(heatmap, cmap='hot')
        ax2.set_title('Feature Importance')
        ax2.axis('off')

        # Overlay of the actual image and the heatmap
        ax3 = plt.subplot(gs[i, 2])
        ax3.imshow(sample_data.detach().cpu().numpy().reshape(28, 28), cmap='gray', alpha=0.5)
        ax3.imshow(heatmap, cmap='hot', alpha=0.5)
        ax3.set_title('Overlay')
        ax3.axis('off')

    plt.tight_layout()
    plt.show()
