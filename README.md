<br />
<p align="center">
  <h1 align="center">Bachelor Project - Interpretable Function Approximation with Gaussian Processes in Value-Based Model-Free Reinforcement Learning</h1>

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

If you want to visualize how an agent interacts with the environment then you can use the `record` method:
```
sim.record(<agent_id>, <num_timesteps>)
```

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


# Simulation Recordings

## Cartpole

### GPQ (Deep Gaussian Process)
![GPQ](/sample_videos/CARTPOLE_GPQ_DGP.gif)
### GPQ (Sparse Variational Gaussian Process)
![GPQ-SVGP](/sample_videos/CARTPOLE_GPQ_SVGP.gif)
### DQN (Multilayer Perceptron)
![DQN-MLP](/sample_videos/CARTPOLE_DQN_MLP.gif)
### DQN (Linear Model)
![DQN-LINEAR](/sample_videos/CARTPOLE_DQN_LINEAR.gif)

## Lunar Lander
### GPSARSA (Deep Gaussian Process)
![GPSARSA1](/sample_videos/LUNAR_LANDER_GPSARSA_DGP_SUCCESS.gif)
![GPSARSA2](/sample_videos/LUNAR_LANDER_GPSARSA_DGP_FAIL.gif)
### GPQ (Sparse Variational Gaussian Process)
![GPQ-SVGP1](/sample_videos/LUNAR_LANDER_GPQ_SVGP_OK.gif)
![GPQ-SVGP2](/sample_videos/LUNAR_LANDER_GPQ_SVGP_FAIL.gif)
### DQN (MLP)
![DQN-MLP1](/sample_videos/LUNAR_LANDER_DQN_MLP_SUCCESS.gif)
![DQN-MLP2](/sample_videos/LUNAR_LANDER_DQN_MLP_SUCCESS2.gif)
### DQN (Linear)
![DQN-LINEAR](/sample_videos/LUJAR_LANDER_DQN_LINEAR_FAIL.gif)

