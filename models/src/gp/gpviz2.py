from typing import Tuple, Optional

import torch
import numpy as np
import matplotlib.pyplot as plt
from botorch.models.gpytorch import GPyTorchModel

from util.fetchdevice import fetch_device


def generate_nd_grid(n_dims: int, num_points: int = 100, ranges=None):
    if ranges is None:
        ranges = [(0, 1)] * n_dims  # Assuming unit cube for simplicity

    grids = [np.linspace(r[0], r[1], num_points) for r in ranges]
    mesh = np.meshgrid(*grids)
    grid_points = np.vstack([m.ravel() for m in mesh]).T
    return grid_points


def plot_gp_contours_with_uncertainty2(gp_model: GPyTorchModel, state_dim: int, d_values: int,
                                       plot_dims: tuple[int, int],
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
    grid_points = generate_nd_grid(n_dims=state_dim)

    fig, axes = plt.subplots(2, d_values, figsize=(15, 10))

    for d in range(d_values):
        d_tensor = torch.tensor([d] * grid_points.shape[0]).view(-1, 1)
        grid_tensor = torch.tensor(grid_points)
        input_tensor = torch.cat((grid_tensor, d_tensor), dim=1).to(device=fetch_device())

        with torch.no_grad():
            posterior = gp_model.posterior(input_tensor)
            gp_mean, gp_var = posterior.mean, posterior.variance

        Z_mean = gp_mean.numpy()
        Z_std = gp_var.sqrt().numpy()  # Standard deviation

        # Mean contour plot
        points_2d = grid_points[:, [plot_dims[0], plot_dims[1]]]
        ax_mean = axes[0, d]
        contour_mean = ax_mean.contourf(points_2d[:, 0], points_2d[:, 1], Z_mean, cmap='viridis', alpha=0.8)
        ax_mean.contour(points_2d[:, 0], points_2d[:, 1], Z_mean, colors='k', linewidths=0.5)
        ax_mean.set_title(f'{title} Mean, a={d}')
        fig.colorbar(contour_mean, ax=ax_mean)
        ax_mean.set_xlabel('s1')
        ax_mean.set_ylabel('s2')

        # Standard deviation contour plot
        ax_std = axes[1, d]
        contour_std = ax_std.contourf(points_2d[:, 0], points_2d[:, 1], Z_std, cmap='viridis', alpha=0.8)
        ax_std.contour(points_2d[:, 0], points_2d[:, 1], Z_std, colors='k', linewidths=0.5)
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
