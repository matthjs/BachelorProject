import numpy as np
import matplotlib.pyplot as plt
import torch
from botorch.models.gpytorch import GPyTorchModel
from gpytorch.mlls import ExactMarginalLogLikelihood
from typing import Tuple, List, Optional

from util.fetchdevice import fetch_device


# noinspection DuplicatedCode
def plot_gp_contours_with_uncertainty2(gp_model: GPyTorchModel, x_range: Tuple[float, float],
                                       y_range: Tuple[float, float], d_values: int,
                                       highlight_point: Optional[Tuple[float, float, float]] = None,
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

    fig, axes = plt.subplots(2, d_values, figsize=(15, 10))

    for d in range(d_values):
        d_tensor = torch.tensor([d] * xy.shape[0]).view(-1, 1)
        xy_tensor = torch.tensor(xy)
        input_tensor = torch.cat((xy_tensor, d_tensor), dim=1).to(device=fetch_device())

        with torch.no_grad():
            posterior = gp_model.posterior(input_tensor)
            gp_mean, gp_var = posterior.mean, posterior.variance

        Z_mean = gp_mean.view(X.shape).numpy()
        Z_std = gp_var.view(X.shape).sqrt().numpy()  # Standard deviation

        # Mean contour plot
        ax_mean = axes[0, d]
        contour_mean = ax_mean.contourf(X, Y, Z_mean, cmap='viridis', alpha=0.8)
        ax_mean.contour(X, Y, Z_mean, colors='k', linewidths=0.5)
        ax_mean.set_title(f'{title} Mean, a={d}')
        fig.colorbar(contour_mean, ax=ax_mean)
        ax_mean.set_xlabel('s1')
        ax_mean.set_ylabel('s2')

        # Standard deviation contour plot
        ax_std = axes[1, d]
        contour_std = ax_std.contourf(X, Y, Z_std, cmap='viridis', alpha=0.8)
        ax_std.contour(X, Y, Z_std, colors='k', linewidths=0.5)
        ax_std.set_title(f'{title} Std Dev, a={d}')
        fig.colorbar(contour_std, ax=ax_std)
        ax_std.set_xlabel('s1')
        ax_std.set_ylabel('s2')

        # Highlight the specified point if provided
        if highlight_point:
            ax_mean.scatter(highlight_point[0], highlight_point[1], color='red' if d == highlight_point[2] else 'blue',
                            label='Highlight Point')
            ax_std.scatter(highlight_point[0], highlight_point[1], color='red' if d == highlight_point[2] else 'blue',
                           label='Highlight Point')

    plt.show()


# noinspection DuplicatedCode
def plot_gp_surface_with_uncertainty2(gp_model: GPyTorchModel, x_range: Tuple[float, float],
                                      y_range: Tuple[float, float], d_values: int,
                                      highlight_point: Optional[Tuple[float, float, float]] = None,
                                      title: str = '') -> None:
    """
    Plots the 3D surface of the GP mean and uncertainty for different discrete values.

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

    fig = plt.figure(figsize=(15, 10))

    # Calculate the minimum and maximum of the GP mean and standard deviation
    z_mean_min = np.inf
    z_mean_max = -np.inf
    z_std_min = np.inf
    z_std_max = -np.inf

    for d in range(d_values):
        d_tensor = torch.tensor([d] * xy.shape[0]).view(-1, 1)
        xy_tensor = torch.tensor(xy)
        input_tensor = torch.cat((xy_tensor, d_tensor), dim=1).to(device=fetch_device())

        with torch.no_grad():
            posterior = gp_model.posterior(input_tensor)
            gp_mean, gp_var = posterior.mean, posterior.variance

        Z_mean = gp_mean.view(X.shape).numpy()
        Z_std = gp_var.view(X.shape).sqrt().numpy()  # Standard deviation

        # Update the minimum and maximum values
        z_mean_min = min(z_mean_min, np.min(Z_mean))
        z_mean_max = max(z_mean_max, np.max(Z_mean))
        z_std_min = min(z_std_min, np.min(Z_std))
        z_std_max = max(z_std_max, np.max(Z_std))

    for d in range(d_values):
        d_tensor = torch.tensor([d] * xy.shape[0]).view(-1, 1)
        xy_tensor = torch.tensor(xy)
        input_tensor = torch.cat((xy_tensor, d_tensor), dim=1).to(device=fetch_device())

        with torch.no_grad():
            posterior = gp_model.posterior(input_tensor)
            gp_mean, gp_var = posterior.mean, posterior.variance

        Z_mean = gp_mean.view(X.shape).numpy()
        Z_std = gp_var.view(X.shape).sqrt().numpy()  # Standard deviation

        # Mean surface plot
        ax_mean = fig.add_subplot(2, d_values, d + 1, projection='3d')
        ax_mean.plot_surface(X, Y, Z_mean, cmap='viridis')
        ax_mean.set_title(f'{title} Mean, a={d}')
        ax_mean.set_xlabel('s1')
        ax_mean.set_ylabel('s2')
        ax_mean.set_zlabel('μ(s1, s2, a)')
        # ax_mean.set_zlim(z_mean_min, z_mean_max)  # Adjust z-range

        # Standard deviation surface plot
        ax_std = fig.add_subplot(2, d_values, d + 1 + d_values, projection='3d')
        ax_std.plot_surface(X, Y, Z_std, cmap='viridis')
        ax_std.set_title(f'{title} Std Dev, a={d}')
        ax_std.set_xlabel('s1')
        ax_std.set_ylabel('s2')
        ax_std.set_zlabel('σ(s1, s2, a)')
        # ax_std.set_zlim(z_std_min, z_std_max)  # Adjust z-range

        # Highlight the specified point if provided
        if highlight_point:
            x_val = highlight_point[0]
            y_val = highlight_point[1]
            z_range_mean = np.linspace(z_mean_min, z_mean_max, 100)
            z_range_std = np.linspace(z_std_min, z_std_max, 100)
            ax_mean.plot(np.full_like(z_range_mean, x_val), np.full_like(z_range_mean, y_val), z_range_mean,
                         color='red' if d == highlight_point[2] else 'blue', linewidth=2, alpha=1)
            ax_std.plot(np.full_like(z_range_std, x_val), np.full_like(z_range_std, y_val), z_range_std,
                        color='red' if d == highlight_point[2] else 'blue', linewidth=2, alpha=1)

    plt.show()


# noinspection DuplicatedCode
def plot_sample_from_gp_posterior2(gp_model: GPyTorchModel, x_range: Tuple[float, float],
                                   y_range: Tuple[float, float], d_values: int, num_samples: int = 1) -> None:
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

    for d in range(d_values):
        d_tensor = torch.tensor([d] * xy.shape[0]).view(-1, 1)
        xy_tensor = torch.tensor(xy)
        input_tensor = torch.cat((xy_tensor, d_tensor), dim=1).to(device=fetch_device())

        gp_model.eval()  # Put the model in evaluation mode
        gp_model.likelihood.eval()  # Put the likelihood in evaluation mode

        with torch.no_grad():
            posterior = gp_model.posterior(input_tensor)
            samples = posterior.rsample(sample_shape=torch.Size([num_samples]))

        for j in range(num_samples):
            Z_sample = samples[j].view(X.shape).numpy()

            ax = fig.add_subplot(1, d_values, d + 1, projection='3d')
            ax.plot_surface(X, Y, Z_sample, cmap='viridis')
            ax.set_title(f'Sample action-value function {j + 1}, a={d}')
            ax.set_xlabel('s1')
            ax.set_ylabel('s2')
            ax.set_zlabel('q(s1, s2, a)')

    plt.show()
