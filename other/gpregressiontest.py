import math
import torch
import gpytorch
from matplotlib import pyplot as plt

from gp.gp import GaussianProcessRegressor
from trainers.gptrainer import GaussianProcessTrainer
from util.fetchdevice import fetch_device


def test_gpr():
    # Training data is 11 points in [0,1] inclusive regularly spaced
    train_x = torch.linspace(0, 1, 100)
    # True function is sin(2*pi*x) with Gaussian noise
    train_y = torch.sin(train_x * (2 * math.pi)) + torch.randn(train_x.size()) * 0.2
    train_x = train_x.to(device=fetch_device())
    train_y = train_y.to(device=fetch_device())

    model = GaussianProcessRegressor(train_x, train_y)
    model = model.to(device=fetch_device())

    # optimize kernel parameters using marginal log likelihood
    model.train()
    model.likelihood.train()

    mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)

    trainer = GaussianProcessTrainer(model, learning_rate=0.1)
    trainer.train(train_x, train_y, num_epochs=50)

    # training_iter = 50
    # for i in range(training_iter):
    #     Zero gradients from previous iteration
        # optimizer.zero_grad()
        # Output from model
        # output = model(train_x)
        # Calc loss and backprop gradients
        # loss = -mll(output, train_y)
        # loss.backward()
        # print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
        #     i + 1, training_iter, loss.item(),
        #     model.covar_module.base_kernel.lengthscale.item(),
        #     model.likelihood.noise.item()
        # ))
        # optimizer.step()

    test_x = torch.linspace(0, 1, 51).to(device=fetch_device())

    mean, lower, upper, _ = model.predict(test_x)
    mean = mean.cpu()
    lower = lower.cpu()
    upper = upper.cpu()
    test_x = test_x.cpu()
    train_x = train_x.cpu()
    train_y = train_y.cpu()

    with torch.no_grad():
        # Initialize plot
        f, ax = plt.subplots(1, 1, figsize=(4, 3))

        # Plot training data as black stars
        ax.plot(train_x.numpy(), train_y.numpy(), 'k*')
        # Plot predictive means as blue line
        ax.plot(test_x.numpy(), mean.numpy(), 'b')
        # Shade between the lower and upper confidence bounds
        ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
        ax.set_ylim([-3, 3])
        ax.legend(['Observed Data', 'Mean', 'Confidence'])
        f.show()


if __name__ == "__main__":
    test_gpr()