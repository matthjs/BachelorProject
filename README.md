<br />
<p align="center">
  <h1 align="center"></h1>

  <p align="center">
  </p>
</p>

## About The Project

## Getting started

### Prerequisites
- [Docker v4.25](https://www.docker.com/get-started) or higher (if running docker container).
- [Poetry](https://python-poetry.org/).
## Running
Using docker: Run the docker-compose files to run all relevant services (`docker compose up` or `docker compose up --build`).

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

## TODO
* Fix sparsification scheme if possible.
* Make the kernels and acquistion function more easily configurable.
* Implement Hyperparameter to JSON saving.
* Add hyperparameter info into CSV file.
* Finish auto hyperparameter tuning code.
* General code clean up.
* Add proper documentation.
* Add variational GPs and Deep GPs.