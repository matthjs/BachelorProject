import os

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


def plot_complexities2(datasets, xlabel='X-axis', ylabel='Y-axis', plot_dir=None, filename=None):
    """
    Plot a generic dataset of values [v1, v2, v3]
    and additionally displays the mean and fits a linear regression
    and degree 2 polynomial regression model.
    :param filename:
    :param datasets:
    :param xlabel:
    :param ylabel:
    :param plot_dir:
    :return:
    """
    plt.figure(figsize=(10, 6))

    for i, data in enumerate(datasets):
        # Define data points implicitly based on the length of the dataset
        data_points = np.arange(len(data))

        # Reshape data points for model fitting
        data_points_reshaped = data_points.reshape(-1, 1)

        # Fit a linear regression model
        linear_regressor = LinearRegression()
        linear_regressor.fit(data_points_reshaped, data)
        linear_pred = linear_regressor.predict(data_points_reshaped)

        # Fit a polynomial regression model (degree 2)
        poly = PolynomialFeatures(degree=2)
        data_points_poly = poly.fit_transform(data_points_reshaped)
        poly_regressor = LinearRegression()
        poly_regressor.fit(data_points_poly, data)
        poly_pred = poly_regressor.predict(data_points_poly)

        # Calculate the mean
        mean_value = np.mean(data)

        # Plot the data and the models
        plt.scatter(data_points, data, label=f'Dataset {i + 1} Actual Data')
        plt.plot(data_points, linear_pred, label=f'Dataset {i + 1} Linear Regression')
        plt.plot(data_points, poly_pred, label=f'Dataset {i + 1} Polynomial Regression (degree 2)')
        plt.axhline(y=mean_value, color='gray', linestyle='--', label=f'Dataset {i + 1} Mean')

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title('Data Points vs. Values with Regression Fits')
    plt.legend()

    if plot_dir is not None:
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        plt.savefig(plot_dir + filename)

    plt.show()


def plot_complexities(data, dataset_name='Dataset', xlabel='X-axis', ylabel='Y-axis', plot_dir=None):
    """
    Plot a generic dataset of values [v1, v2, v3]
    and additionally displays the mean and fits a linear regression
    and degree 2 polynomial regression model.
    :param filename:
    :param data: List of values
    :param dataset_name: Name of the dataset for labeling
    :param xlabel: Label for the x-axis
    :param ylabel: Label for the y-axis
    :param plot_dir: Directory to save the plot, if specified
    :return: None
    """
    # print("D", data)

    plt.figure(figsize=(10, 6))

    # Define data points implicitly based on the length of the dataset
    data_points = np.arange(len(data))

    # Reshape data points for model fitting
    data_points_reshaped = data_points.reshape(-1, 1)

    # Fit a linear regression model
    linear_regressor = LinearRegression()
    linear_regressor.fit(data_points_reshaped, data)
    linear_pred = linear_regressor.predict(data_points_reshaped)

    # Fit a polynomial regression model (degree 2)
    poly = PolynomialFeatures(degree=2)
    data_points_poly = poly.fit_transform(data_points_reshaped)
    poly_regressor = LinearRegression()
    poly_regressor.fit(data_points_poly, data)
    poly_pred = poly_regressor.predict(data_points_poly)

    # Calculate the mean
    mean_value = np.mean(data)

    # Plot the data and the models
    plt.scatter(data_points, data, label=f'{dataset_name} Actual Data')
    plt.plot(data_points, linear_pred, label=f'{dataset_name} Linear Regression')
    plt.plot(data_points, poly_pred, label=f'{dataset_name} Polynomial Regression (degree 2)')
    plt.axhline(y=mean_value, color='gray', linestyle='--', label=f'{dataset_name} Mean')

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f'{dataset_name} vs. Values with Regression Fits')
    plt.legend()

    if plot_dir is not None:
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        plt.savefig(os.path.join(plot_dir, f'{dataset_name}.png'))

    # plt.show()


if __name__ == "__main__":
    # Sample data
    data = [3.1, 3.4, 4.9, 5.3, 6.5, 7.1, 7.4, 9.3, 11.4, 13.4]

    plot_complexities(data, dataset_name='Memory Usage', xlabel='Number of Data Points', ylabel='Memory Usage (GB)')
