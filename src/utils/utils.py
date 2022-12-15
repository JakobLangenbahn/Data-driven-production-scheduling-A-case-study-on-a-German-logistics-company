""" Utility functions for simulation and modelling """
from math import floor
from random import randrange, seed
from scipy.stats import truncnorm
from random import sample


def transform_to_real_time(number, simulation_start):
    """
    Transform simulation time to real clock time
    :param number: Simulation time from simpy simulation
    :param simulation_start: Planned real world start date of simulation in seconds from 1970-01-01
    :returns: Real world timestamp in seconds from 1970-01-01
    """
    return number * 1000 + simulation_start


def generate_random_deviation(min_value, max_value, mean, var, random_state):
    """
    Generate random variable of a truncated normal distribution
    :param min_value: Lower bound for range
    :param max_value: Higher bound for range
    :param mean: Mean value of normal distribution
    :param var: Variance of normal distribution
    :param random_state: Random seed for reproducibility
    :returns: Sample of truncated normal distribution
    """
    # Calculate values for truncated normal distribution
    a = (min_value - mean) / var,
    b = (max_value - mean) / var
    return floor(truncnorm.rvs(a, b, mean, var, 1, random_state=random_state))


def random_range(begin, end, random_state):
    """
    Provide a random number in a given range based on a random seed
    :param begin: Minimal value of range
    :param end: Maximal value of range
    :param random_state: Random seed for reproducibility
    :returns: Random value from range
    """
    seed(random_state)
    return randrange(begin, end)


def get_hyperparameter_number(params):
    """
    Calculates the number of hyperparameter combinations in a parameter grid
    :param params: Hyperparameter grid
    :returns: Number of hyperparameter combinations in a parameter grid
    """
    number_param = 1
    for key in params:
        number_param *= len(params[key])
    return number_param


def get_random_hyperparameter(param_grid, number, random_seed):
    """
    Get a random subset of a parameter grid
    :param param_grid: Grid of hyperparameters
    :param number: Size of hyperparameter subset
    :param random_seed: Random seed for reproducibility
    :returns: Random subset of hyperparameter grid
    """
    param_grid_list = []
    for i in range(number):
        seed(random_seed + i)
        param_dict = {}
        for key in param_grid:
            param_dict[key] = sample(param_grid[key], 1)[0]
        param_grid_list.append(param_dict)
    return param_grid_list
