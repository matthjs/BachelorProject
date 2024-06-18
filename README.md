<br />
<p align="center">
  <h1 align="center">Bachelor Project - Gaussian processes and Reinforcement Learning</h1>

  <p align="center">
  </p>
</p>

## About The Project
This thesis contains the code that was used for my bachelor thesis which focuses on approximation the action-value function using Gaussian process regression.


## Getting started

### Prerequisites
- [Docker v4.25](https://www.docker.com/get-started) or higher (if running docker container).
- [Poetry](https://python-poetry.org/).
## Running
Using docker: Run the docker-compose files to run all relevant services (`docker compose up` or `docker compose up --build`).
It will automatically run the `experiment.py` script, which can be used to reproduce the results from my thesis.

You can also set up a virtual environment using Poetry. Poetry can  be installed using `pip`:
```
pip install poetry
```
Then initiate the virtual environment with the required dependencies (see `poetry.lock`, `pyproject.toml`):
```
poetry config virtualenvs.in-project true    # ensures virtual environment is in project
poetry install
```
The virtual environment can be accessed from the shell using:
```
poetry shell
```
IDEs like Pycharm will be able to detect the interpreter of this virtual environment.

## Usage

The `SimulatorRL` is used to train and evaluate RL agents on a gymnasium environment.
```
sim = SimulatorRL("CartPole-v1", experiment_id="experiment_x")
```

Note that for each agent it is required that a yaml file exists in the configs folder of the form:
```
config_<agent_id>_<environment_str>.yaml
```

You first need to register the agents using the `register_agent` method:
``````
sim.register_agent(<agent_id>, <agent_type>)
``````
where `<agent_type>` can be:
* `sb_dqn` for Stable-Baselines3 DQN implementation.
* `sb_ppo` for Stable-Baselines3 PPO implementation.
* `gpsarsa_agent` for the GP-SARSA algorithm.
* `gpq_agent` for the GP-Q algorithm.

You can then sequentially train all agents for `N` episodes:
```
sim.train_agents(num_episodes=N, callbacks=[...])
```
where callbacks are objects which can be used to keep track of certain metrics during training and/or evaluation. For example the `RewardCallback` keeps track of the average return over episodes (+ variance) and the `UsageCallback` keeps track of time, memory and energy usage.

After training is completed you can evaluate the agents for `M` episodes with:
```
sim.evaluate_agents(M, callbacks=[...])
```
The `RewardCallback` will allow you to determine the average return (return=sum of rewards of an episode) during evaluation.

Additionally, you can plot some relevant collected data using `plot_any_plottable_data()`, save them to a csv file using `.data_to_csv()` and save your agents to stable storage using `.save_agents()`

If you want to visualize how an agent interacts with the environment then you can use the `play` method:
```
sim.play(<agent_id>, <num_episodes>)
```
## TODO
* Fix sparsification scheme if possible.
* ~~Make the kernels and acquistion function more easily configurable.~~
* Implement Hyperparameter to JSON saving.
* ~~Add hyperparameter info into CSV file~~.
* Finish auto hyperparameter tuning code (probably too computationally expensive though).
* ~~General code clean up~~.
* Add more documentation.
* ~~Add variational GPs and Deep GPs.~~

## NOTES
Online updates (Fantasization) for exact GPs works, but the problem is that you cannot cap the dataset size. From testing the time and space complexity does not appear to be too different. However one key difference is that the hyperparameters from the previous fit are retained, which results in a different GP fit. The compromise can be to copy the fitted hyperparameters (kernel parameters + noise variance).
* I have verified that exact GP inference is indeed $O(N^2)$. Time complexity still needs to be verified. TODO: Compare time with fit_gpytorch_mll and space usage with other algorithms.
* Problem: Memory usage for exactGP does not scale properly, variational GPs cope decently and appear to scale more linearly but posterior is an approximation.
* For linear Q-learning, perhaps it is easier and more meaningful to make a comparison to StableBaselines3 DQN algorithm with a linear model.
* Observation noise for posterior yes or no?
* Test dataset size possible for minibatching.
* Balance the batch_size for minibatching and also the batch_size for updating.
* Initialize mean function to nonzero constant?
* UPDATE HYPERPARAMETERS ON ONLY PART OF THE DATASET, THIS MAY ALLOW DEEP GPS (OR VARIATIONAL GPS) TO HAVE SIGNIFICANTLY BIGGER BASE DATASETS TO OPTIMIZE OVER!!!! OPTIMIZING OVER ENTIRE DATASET BECOMES TOO SLOW.
* Reduced speed after reloading deep GP agent in Lunar Lander.
* SVGP consistent low loss on Lunar Lander but limited performance -> need for deep GP?
* For Lunar Lander I guess the number of inducing points needs to be large enough?
### Lunar Lander
* How many layers and inducing points do I need?? 
* Is it better to optimize over random batches or the latest batches??
* SVGP with high fit batch size (1024) and inducing point size (1024) with random batching with low learning rate (0.001) seems to have trouble learning.
* Adding at least one layer has an effect.
* What is the effect of using a discrete kernel for ther actions?
* Maybe do > 1 epoch on each minibatch?
* How many fit batches is actually necessary??
* Decreasing batch fit size does not seem to speed things up that much.
* Random minibatching or N latest minibatches??
* Is it meaningful to increase dataset butches > 100,000?
* Composite kernel seems to cause the agent to not learn? Random_batching=Yes, latest_batching=Yes
* 0.001 to low learning rate?
* Immediate decline in performance in beginning.
* Experiment 200 was able to get > 200 reward! But then performance does not improve that much. Random fitting after all?
* For latest N batches it is actually better to just limit the dataset size instead.
* Look into SVGP performance with relatively high number of inducing points.
* What is the ideal learnign rate? Too low -> slow divergence/drop in performance? High -> Fast spike early.
* Down up trajectory common for SVGP (with ~1000 inducing points) and "high" LR? (0.1, 0.01, 0.001). Lower than this little learning though.
## Information on modules

### Agents
Contains objects that interact with the (gymnasium) environments and also a factory class for the creation of such objects. Note that any computation is delegated to other objects.

### Models
Contains Gaussian process models, Bayesian optimizers, neural networks and objects that are able to fit these models for
value function approximation.

### Simulation
Contains classes that have to do with running RL algorithms on gymnasium environments and keeping track of relevant metrics.

### Statistics
Contains the `metricstracker` class which allows one to keep track of means and variances of values in an online manner.
