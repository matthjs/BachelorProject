import numpy as np
import matplotlib.pyplot as plt
import torch
from botorch.models.gpytorch import GPyTorchModel
from gpytorch.mlls import ExactMarginalLogLikelihood
from typing import Tuple, List, Optional

"""
Note: these visualization functions only work for GP models mapping functions f : R x R x Z -> R
"""


def generate_dummy_data(num_samples: int = 50) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates dummy data for training the GP model.

    :param num_samples: Number of samples to generate.
    :return: Tuple of input tensor and target tensor.
    """
    x = np.random.uniform(0, 1, (num_samples, 2))
    d = np.random.randint(0, 3, num_samples).reshape(-1, 1)  # Discrete dimension with values in {0, 1, 2}
    inputs = np.hstack([x, d])
    targets = np.sin(2 * np.pi * inputs[:, 0]) + np.cos(2 * np.pi * inputs[:, 1]) + 0.1 * inputs[:,
                                                                                          2] + 0.1 * np.random.randn(
        num_samples)
    return torch.tensor(inputs, dtype=torch.float64), torch.tensor(targets, dtype=torch.float64).unsqueeze(-1)


# noinspection DuplicatedCode
def plot_gp_contours_with_uncertainty(gp_model: GPyTorchModel, x_range: Tuple[float, float],
                                      y_range: Tuple[float, float], d_values: List[int],
                                      highlight_point: Optional[Tuple[float, float]] = None,
                                      title: str = '') -> None:
    """
    Plots the contour of the GP mean and uncertainty for different discrete values.

    :param gp_model: Trained GP model.
    :param x_range: Range of x values.
    :param y_range: Range of y values.
    :param d_values: List of discrete values.
    :param highlight_point: Tuple of (x1, x2) values to highlight on the plot.
    :param title: Title for the plots.
    """
    x = np.linspace(x_range[0], x_range[1], 100)
    y = np.linspace(y_range[0], y_range[1], 100)
    X, Y = np.meshgrid(x, y)
    xy = np.vstack([X.ravel(), Y.ravel()]).T

    fig, axes = plt.subplots(2, len(d_values), figsize=(15, 10))

    for i, d in enumerate(d_values):
        d_tensor = torch.tensor([d] * xy.shape[0]).view(-1, 1)
        xy_tensor = torch.tensor(xy)
        input_tensor = torch.cat((xy_tensor, d_tensor), dim=1)

        with torch.no_grad():
            posterior = gp_model.posterior(input_tensor)
            gp_mean, gp_var = posterior.mean, posterior.variance

        Z_mean = gp_mean.view(X.shape).numpy()
        Z_std = gp_var.view(X.shape).sqrt().numpy()  # Standard deviation

        # Mean contour plot
        ax_mean = axes[0, i]
        contour_mean = ax_mean.contourf(X, Y, Z_mean, cmap='viridis', alpha=0.8)
        ax_mean.contour(X, Y, Z_mean, colors='k', linewidths=0.5)
        ax_mean.set_title(f'{title} Mean, a={d}')
        fig.colorbar(contour_mean, ax=ax_mean)
        ax_mean.set_xlabel('s1')
        ax_mean.set_ylabel('s2')

        # Standard deviation contour plot
        ax_std = axes[1, i]
        contour_std = ax_std.contourf(X, Y, Z_std, cmap='viridis', alpha=0.8)
        ax_std.contour(X, Y, Z_std, colors='k', linewidths=0.5)
        ax_std.set_title(f'{title} Std Dev, a={d}')
        fig.colorbar(contour_std, ax=ax_std)
        ax_std.set_xlabel('s1')
        ax_std.set_ylabel('s2')

        # Highlight the specified point if provided
        if highlight_point:
            ax_mean.scatter(*highlight_point, color='red', label='Highlight Point')
            ax_std.scatter(*highlight_point, color='red', label='Highlight Point')

    plt.show()


# noinspection DuplicatedCode
def plot_gp_surface_with_uncertainty(gp_model: GPyTorchModel, x_range: Tuple[float, float],
                                     y_range: Tuple[float, float], d_values: List[int], title: str = '') -> None:
    """
    Plots the 3D surface of the GP mean and uncertainty for different discrete values.

    :param gp_model: Trained GP model.
    :param x_range: Range of x values.
    :param y_range: Range of y values.
    :param d_values: List of discrete values.
    :param title: Title for the plots.
    """
    x = np.linspace(x_range[0], x_range[1], 100)
    y = np.linspace(y_range[0], y_range[1], 100)
    X, Y = np.meshgrid(x, y)
    xy = np.vstack([X.ravel(), Y.ravel()]).T

    fig = plt.figure(figsize=(15, 10))

    for i, d in enumerate(d_values):
        d_tensor = torch.tensor([d] * xy.shape[0]).view(-1, 1)
        xy_tensor = torch.tensor(xy)
        input_tensor = torch.cat((xy_tensor, d_tensor), dim=1)

        with torch.no_grad():
            posterior = gp_model.posterior(input_tensor)
            gp_mean, gp_var = posterior.mean, posterior.variance

        Z_mean = gp_mean.view(X.shape).numpy()
        Z_std = gp_var.view(X.shape).sqrt().numpy()  # Standard deviation

        # Mean surface plot
        ax_mean = fig.add_subplot(2, len(d_values), i + 1, projection='3d')
        ax_mean.plot_surface(X, Y, Z_mean, cmap='viridis')
        ax_mean.set_title(f'{title} Mean, a={d}')
        ax_mean.set_xlabel('s1')
        ax_mean.set_ylabel('s2')
        ax_mean.set_zlabel('μ(s1, s2, a)')

        # Standard deviation surface plot
        ax_std = fig.add_subplot(2, len(d_values), i + 1 + len(d_values), projection='3d')
        ax_std.plot_surface(X, Y, Z_std, cmap='viridis')
        ax_std.set_title(f'{title} Std Dev, a={d}')
        ax_std.set_xlabel('s1')
        ax_std.set_ylabel('s2')
        ax_std.set_zlabel('Std Dev σ(s1, s2, a)')

    plt.show()


# noinspection DuplicatedCode
def plot_sample_from_gp_posterior(gp_model: GPyTorchModel, x_range: Tuple[float, float],
                                  y_range: Tuple[float, float], d_values: List[int], num_samples: int = 1) -> None:
    """
    Samples and plots functions from the GP posterior for different discrete values.

    :param gp_model: Trained GP model.
    :param x_range: Range of x values.
    :param y_range: Range of y values.
    :param d_values: List of discrete values.
    :param num_samples: Number of samples to draw from the posterior.
    """
    x = np.linspace(x_range[0], x_range[1], 100)
    y = np.linspace(y_range[0], y_range[1], 100)
    X, Y = np.meshgrid(x, y)
    xy = np.vstack([X.ravel(), Y.ravel()]).T

    fig = plt.figure(figsize=(15, 5))

    for i, d in enumerate(d_values):
        d_tensor = torch.tensor([d] * xy.shape[0]).view(-1, 1)
        xy_tensor = torch.tensor(xy)
        input_tensor = torch.cat((xy_tensor, d_tensor), dim=1)

        gp_model.eval()  # Put the model in evaluation mode
        gp_model.likelihood.eval()  # Put the likelihood in evaluation mode

        with torch.no_grad():
            posterior = gp_model.posterior(input_tensor)
            samples = posterior.rsample(sample_shape=torch.Size([num_samples]))

        for j in range(num_samples):
            Z_sample = samples[j].view(X.shape).numpy()

            ax = fig.add_subplot(1, len(d_values), i + 1, projection='3d')
            ax.plot_surface(X, Y, Z_sample, cmap='viridis')
            ax.set_title(f'Sample action-value function {j + 1}, a={d}')
            ax.set_xlabel('s1')
            ax.set_ylabel('s2')
            ax.set_zlabel('q(s1, s2, a)')

    plt.show()


# noinspection DuplicatedCode
def plot_gp_surface(gp_model: GPyTorchModel, x_range: Tuple[float, float], y_range: Tuple[float, float],
                    d_values: List[int], title: str = '') -> None:
    """
    Plots the 3D surface of the GP mean for different discrete values.

    :param gp_model: Trained GP model.
    :param x_range: Range of x values.
    :param y_range: Range of y values.
    :param d_values: List of discrete values.
    :param title: Title for the plots.
    """
    x = np.linspace(x_range[0], x_range[1], 100)
    y = np.linspace(y_range[0], y_range[1], 100)
    X, Y = np.meshgrid(x, y)
    xy = np.vstack([X.ravel(), Y.ravel()]).T

    fig = plt.figure(figsize=(15, 5))

    for i, d in enumerate(d_values):
        d_tensor = torch.tensor([d] * xy.shape[0]).view(-1, 1)
        xy_tensor = torch.tensor(xy)
        input_tensor = torch.cat((xy_tensor, d_tensor), dim=1)

        with torch.no_grad():
            gp_mean = gp_model.posterior(input_tensor).mean

        Z = gp_mean.view(X.shape).numpy()

        ax = fig.add_subplot(1, len(d_values), i + 1, projection='3d')
        ax.plot_surface(X, Y, Z, cmap='viridis')
        ax.set_title(f'{title}, d={d}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('f(x, y, d)')

    plt.show()
